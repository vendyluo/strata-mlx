"""Builds the MLX model straight from the GGUF files llama.cpp and Strata use.

The module tree is the mlx-lm port of the model (`mlx_lm.models.qwen4_exp`), which follows the original
checkpoint's layout. llama.cpp's converter changed three things on the way to GGUF, undone here:

  * the delta net's value heads are stored tiled (k0 k1 .. k0 k1 ..) instead of grouped by key head;
  * `ssm_a` is `-exp(A_log)`;
  * the indexer's one query-and-key projection is split in two.

The norms need nothing: GGUF and the port both hold the weight with the reference's `1 +` folded in. The 28.8 GB
n-gram table is not loaded: its rows are read from the file as tokens ask for them.

The experts, nine tenths of the file, are not converted at all: their blocks go to the GPU as they are and
Metal kernels compute on them (blocks.py). Every other quantized tensor becomes MLX's own quantized form, but
where experts are read from the SSD: there the Q3_K and IQ4_XS ones, which MLX can only hold at 8 bits, stay
blocks too (dense.py), and the memory that spares is the experts'.
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models import qwen4_exp

from .blocks import BLOCK, BlockSwitchGLU, BlockSwitchLinear
from .dense import AS_BLOCKS, BlockEmbedding, BlockLinear, ScaledLinear, fits, kept_floats
from .dense import as_blocks as stays_blocks
from .paged import ExpertStore, PagedSwitchGLU, all_fit, sizes
from .paged import machine_memory as paged_machine_memory
from .gguf import KVALUES_IQ4NL, Tensor, pack, q6k_parts, shards, to_affine, to_float

_EXACT_FLOAT = ("F32", "F16", "BF16", "Q6_K")


class PleTable(nn.Module):
    """The n-gram table, left in its file: IQ4_NL rows of 160 values, 90 bytes each."""

    def __init__(self, t: Tensor):
        super().__init__()
        if t.type != "IQ4_NL":
            raise ValueError(f"the n-gram table is {t.type}; only IQ4_NL is read")
        self.dim = t.shape[-1]
        self._rows = t.data.reshape(t.shape[0], -1)

    def __call__(self, gid: mx.array) -> mx.array:
        return self.read(np.array(gid))

    def read(self, gid: np.ndarray) -> mx.array:
        b = self._rows[gid.reshape(-1)].reshape(-1, 18)
        d = b[:, :2].copy().view("<f2").astype(np.float32)
        nib = np.concatenate([b[:, 2:] & 15, b[:, 2:] >> 4], axis=1)
        return mx.array((KVALUES_IQ4NL[nib] * d).reshape(*gid.shape, self.dim))


class HostNgram:
    """The n-gram layer's embedding, with the rows worked out on the CPU.

    Which rows a token reads is a hash of the tokens before it. The port computes it as some thirty small GPU
    operations whose result the table lookup then has to wait for, in the middle of building a pass; the tokens
    are known here, and the same arithmetic in numpy needs no GPU at all.
    """

    def __init__(self, emb, table: PleTable):
        self.table, self.port = table, emb  # the port's own embedding is kept to compare against
        self.ngram_size, self.heads_per_ngram, self.eos = emb.ngram_size, emb.heads_per_ngram, emb.eos_token_id
        self.mults, self.sizes, self.offsets = (np.array(a.tolist(), dtype=np.int64)
                                                for a in (emb._mults, emb._sizes, emb._offsets))

    def _shift_right(self, ids: np.ndarray, shift: int) -> np.ndarray:
        """Shifted right by `shift`, without crossing an end-of-text token."""
        if shift == 0:
            return ids
        B, T = ids.shape
        pos = np.arange(T)
        last_end = np.maximum.accumulate(np.where(ids == self.eos, pos, -1), axis=1)
        before = np.concatenate([np.full((B, 1), -1), last_end[:, :-1]], axis=1)
        src = pos - shift
        gathered = np.take_along_axis(ids, np.broadcast_to(np.maximum(src, 0)[None], (B, T)), axis=1)
        ok = (pos[None] - (before + 1) >= shift) & (src[None] >= 0)
        return np.where(ok, gathered, self.eos)

    def rows(self, ids: np.ndarray, context: np.ndarray) -> np.ndarray:
        """(B, T) tokens after (B, n - 1) of context -> (B, T, heads) rows of the table."""
        history = np.concatenate([context, ids], axis=1).astype(np.int64)
        shifted = [self._shift_right(history, s) for s in range(self.ngram_size)]
        parts = []
        with np.errstate(over="ignore"):  # the hash multiplies modulo 2**64
            for ngram in range(2, self.ngram_size + 1):
                lo = (ngram - 2) * self.heads_per_ngram
                hi = lo + self.heads_per_ngram
                mixed = shifted[0] * self.mults[0]
                for p in range(1, ngram):
                    mixed = mixed ^ (shifted[p] * self.mults[p])
                parts.append(mixed[..., None] % self.sizes[lo:hi] + self.offsets[lo:hi])
        return np.concatenate(parts, axis=-1)[:, -ids.shape[1]:]

    def __call__(self, ids: mx.array, prev_context: mx.array) -> mx.array:
        gid = self.rows(np.array(ids), np.array(prev_context))
        return self.table.read(gid).reshape(*gid.shape[:2], -1)


POOL = 4 << 30  # of freed buffers MLX may keep; a pass frees less, a chunk of prompt reuses about this much


def v_heads_grouped(n_k: int, per_k: int, head_dim: int) -> np.ndarray:
    """Index that takes value heads from GGUF's tiled order back to the checkpoint's grouped order."""
    return np.arange(n_k * per_k * head_dim).reshape(per_k, n_k, head_dim).transpose(1, 0, 2).reshape(-1)


def _row_chunks(t: Tensor, rows: np.ndarray | None = None, weights: int = 8 << 20):
    """The tensor a few rows at a time, in the order `rows` if given.

    Converting a tensor takes several arrays of its size on the way, and what numpy frees stays with the process:
    done whole, the 248,320-row embedding alone left 4 GB behind, and the trunk without its experts 13 GB for 5.6
    of weights. A few million weights at a time leave nothing worth counting.
    """
    n_rows, n_cols = t.shape
    raw = t.data.reshape(n_rows, -1)
    step = max(1, weights // n_cols)
    for i in range(0, n_rows if rows is None else len(rows), step):
        part = raw[i : i + step] if rows is None else raw[rows[i : i + step]]
        yield Tensor(t.name, (len(part), n_cols), t.type, part.reshape(-1))


def _joined(parts: list) -> list:
    return [mx.concatenate(column) if len(column) > 1 else column[0] for column in zip(*parts)]


def _affine(t: Tensor, rows: np.ndarray | None = None, cols: np.ndarray | None = None):
    """(packed codes, scales, biases) as MLX arrays, and (bits, group); `rows` and `cols` reorder the weights."""
    parts = []
    for sub in _row_chunks(t, rows):
        a = to_affine(sub)
        codes, scales, biases = a.codes, a.scales, a.biases
        if cols is not None:
            if (cols.reshape(-1, a.group) - cols[:: a.group, None] != np.arange(a.group)).any():
                raise ValueError(f"{t.name}: the input reorder splits a quantization group")
            groups = cols[:: a.group] // a.group
            codes, scales, biases = codes[:, cols], scales[:, groups], biases[:, groups]
        parts.append((mx.array(pack(codes, a.bits)), mx.array(np.ascontiguousarray(scales)),
                      mx.array(np.ascontiguousarray(biases))))
        mx.eval(parts[-1])
    return _joined(parts), (a.bits, a.group)


def scaled(t: Tensor, rows: np.ndarray | None = None, cols: np.ndarray | None = None) -> ScaledLinear:
    """A Q6_K layer as its codes, scales and d (dense.py); `rows` and `cols` reorder its outputs and inputs."""
    step = 256  # weights to a d: fewer where the reorder moves less than a block at a time
    if cols is not None:
        if (cols.reshape(-1, 16) - cols[::16, None] != np.arange(16)).any():
            raise ValueError(f"{t.name}: the input reorder splits a run of 16 weights")
        while (cols.reshape(-1, step) - cols[::step, None] != np.arange(step)).any():
            step //= 2
    parts = []
    for sub in _row_chunks(t, rows):
        codes, scales, d = q6k_parts(sub)
        if cols is not None:
            codes, scales, d = codes[:, cols], scales[:, cols[::16] // 16], d[:, cols[::step] // 256]
        parts.append(tuple(mx.array(np.ascontiguousarray(a)) for a in (codes, scales, d)))
        mx.eval(parts[-1])
    whole = _joined(parts)
    mx.eval(whole)
    return ScaledLinear(*whole)


def linear(t: Tensor, rows: np.ndarray | None = None, cols: np.ndarray | None = None, as_blocks: tuple = ()):
    """A layer holding the tensor's weights, called like a Linear; `rows` and `cols` reorder its outputs and
    inputs. A tensor of a type in `as_blocks` stays the file's blocks."""
    if t.type == "Q6_K" and fits(*t.shape):
        return scaled(t, rows, cols)
    if stays_blocks(t, as_blocks):
        raw = t.data.reshape(t.shape[0], -1)
        return BlockLinear(mx.array(np.ascontiguousarray(raw if rows is None else raw[rows]).reshape(-1)), t.type,
                           (t.shape[0] if rows is None else len(rows), t.shape[1]), cols)
    if t.type in _EXACT_FLOAT:
        parts = []
        for sub in _row_chunks(t, rows):
            w = to_float(sub)
            parts.append((mx.array(np.ascontiguousarray(w if cols is None else w[:, cols])),))
            mx.eval(parts[-1])
        lin = nn.Linear(1, 1, bias=False)
        (lin.weight,) = _joined(parts)
        return lin
    (weight, scales, biases), (bits, group) = _affine(t, rows, cols)
    q = nn.QuantizedLinear(group, 1, bias=False, group_size=group, bits=bits)
    q.weight, q.scales, q.biases = weight, scales, biases
    return q


def experts(t: Tensor) -> BlockSwitchLinear:
    """A stack of experts, kept as the file's blocks."""
    if t.type not in BLOCK:
        raise ValueError(f"{t.name}: {t.type} experts are not supported")
    return BlockSwitchLinear(mx.array(t.data), t.type, t.shape)


def embedding(t: Tensor, dtype: mx.Dtype = mx.float16, as_blocks: tuple = ()):
    if stays_blocks(t, as_blocks):
        return BlockEmbedding(mx.array(t.data), t.type, t.shape, dtype)
    (weight, scales, biases), (bits, group) = _affine(t)
    q = nn.QuantizedEmbedding(1, group, group_size=group, bits=bits)
    q.weight, q.scales, q.biases = weight, scales, biases
    q.num_embeddings, q.dims = t.shape
    return q


def conv(t: Tensor, order: np.ndarray | None = None) -> mx.array:
    w = to_float(t)  # (channels, kernel)
    return mx.array(np.ascontiguousarray(w if order is None else w[order])[:, :, None])


def load(path: str, dtype: mx.Dtype = mx.float16, expert_memory: float | str | None = None,
         machine_memory: int | None = None, dense_blocks: tuple | None = None):
    """The model from shard 1 of a GGUF (shard 2, beside it, holds the n-gram table). Returns (model, metadata).

    `expert_memory`, in GB: keep only that much of the experts in memory and read the others from the file as
    tokens ask for them (paged.py); the store is `model.expert_store`. None: all of them in memory. "auto": all
    of them if the machine (`machine_memory` bytes, by default this one's) holds the model, else a store whose
    budget the caller sets with `store.fit()` once everything else is loaded.

    `dense_blocks`: the types of dense tensors to keep as the file's blocks. By default those of AS_BLOCKS
    when experts are read from the file, none when all are in memory: the blocks are 1.7 GB less with the Q2_0
    file, and a millisecond a pass slower once a pass has four tokens.

    `dtype` is the type of the activations and of every float weight and scale. float32 holds each weight of the
    file exactly; float16, the default, rounds the scales to 11 bits and runs multi-token steps 1.5-2 times faster.
    """
    files = shards(path)
    g = files[0]
    tensors = {name: t for f in files for name, t in f.tensors.items()}
    meta = g.meta
    if meta.get("general.architecture") != "qwen4exp":
        raise ValueError(f"{Path(path).name} is a {meta.get('general.architecture')} model, not qwen4exp")
    used = set()
    machine, (n_experts, n_dense) = machine_memory or paged_machine_memory(), sizes(files)
    if expert_memory == "auto":
        expert_memory = None if all_fit(machine, n_experts, n_dense) else 0.0
    store = ExpertStore(int(expert_memory * 1e9)) if expert_memory is not None else None
    # #3: the Q6_K layers' floats for prompts are kept only where the whole model and they fit. Elsewhere they
    # would take memory the rule gave to macOS or to the experts; made anew for each chunk, prompts are read a
    # tenth slower (320 against 355-369 tok/s with the IQ3_S file on the M4 Max) and answers are as fast.
    keep = store is None and all_fit(machine, n_experts, n_dense, kept_floats(g.tensors, dtype.size))
    if dense_blocks is None:
        dense_blocks = AS_BLOCKS if store is not None else ()

    def take(name: str) -> Tensor:
        used.add(name)
        return tensors[name]

    def lin(name: str, rows: np.ndarray | None = None, cols: np.ndarray | None = None):
        layer = linear(take(name), rows, cols, dense_blocks)
        if isinstance(layer, ScaledLinear):
            layer.keep = keep
        return layer

    def vec(name: str) -> mx.array:
        return mx.array(np.ascontiguousarray(to_float(take(name))))

    k = "qwen4exp."
    interval = meta[k + "full_attention_interval"]
    n_layers = meta[k + "block_count"]
    text = {
        "hidden_size": meta[k + "embedding_length"],
        "num_hidden_layers": n_layers,
        "num_attention_heads": meta[k + "attention.head_count"],
        "num_key_value_heads": meta[k + "attention.head_count_kv"],
        "head_dim": meta[k + "attention.key_length"],
        "vocab_size": tensors["token_embd.weight"].shape[0],
        "full_attention_interval": interval,
        "num_experts": meta[k + "expert_count"],
        "num_experts_per_tok": meta[k + "expert_used_count"],
        "moe_intermediate_size": meta[k + "expert_feed_forward_length"],
        "shared_expert_intermediate_size": meta[k + "expert_shared_feed_forward_length"],
        "linear_num_key_heads": meta[k + "ssm.group_count"],
        "linear_num_value_heads": meta[k + "ssm.time_step_rank"],
        "linear_key_head_dim": meta[k + "ssm.state_size"],
        "linear_value_head_dim": meta[k + "ssm.inner_size"] // meta[k + "ssm.time_step_rank"],
        "linear_conv_kernel_dim": meta[k + "ssm.conv_kernel"],
        "hc_count": meta[k + "hyper_connection.count"],
        "hc_lowrank": meta[k + "hyper_connection.low_rank"],
        "indexer_n_heads": meta[k + "attention.indexer.head_count"],
        "indexer_head_dim": meta[k + "attention.indexer.key_length"],
        "indexer_budget": meta[k + "attention.indexer.top_k"],
        "indexer_compress_ratio": max(meta[k + "attention.compress_ratios"]),
        "ngram_size": meta[k + "ple.ngram_size"],
        "heads_per_ngram": meta[k + "ple.heads_per_ngram"],
        "ple_layer_ids": [i + 1 for i in meta[k + "ple.layers"]],
        "ple_conv_kernel_size": meta[k + "ple.conv_kernel"],
        "eos_token_id": meta[k + "ple.eos_token_id"],
        "rope_theta": meta[k + "rope.freq_base"],
        "partial_rotary_factor": meta[k + "rope.dimension_count"] / meta[k + "attention.key_length"],
        "rms_norm_eps": meta[k + "attention.layer_norm_rms_epsilon"],
    }
    model = qwen4_exp.Model(qwen4_exp.ModelArgs(text_config=text))
    a = model.args.text
    # held until the end, so that an array still in the model at that point is known to be one of these
    start = [v for _, v in tree_flatten(model.parameters())]
    placeholders = {id(v) for v in start}

    n_k, n_v, d_k, d_v = a.linear_num_key_heads, a.linear_num_value_heads, a.linear_key_head_dim, a.linear_value_head_dim
    qk = 2 * n_k * d_k
    v_rows = v_heads_grouped(n_k, n_v // n_k, d_v)
    v_one = v_heads_grouped(n_k, n_v // n_k, 1)
    qkv_rows = np.concatenate([np.arange(qk), qk + v_rows])

    def residual(gr, prefix: str, inject: bool = True):
        gr.hc_norm.weight = vec(prefix + "_norm.weight")
        gr.input_mix_weight_down = lin(prefix + "_down.weight")
        gr.input_mix_weight_up = lin(prefix + "_up.weight")
        if inject:
            gr.block_inject_weight = lin(prefix + "_inject.weight")

    for i, layer in enumerate(model.layers):
        b = f"blk.{i}."
        if layer.layer_type == "linear_attention":
            la = layer.linear_attn
            la.in_proj_qkv = lin(b + "attn_qkv.weight", rows=qkv_rows)
            la.in_proj_z = lin(b + "attn_gate.weight", rows=v_rows)
            la.in_proj_a = lin(b + "ssm_alpha.weight", rows=v_one)
            la.in_proj_b = lin(b + "ssm_beta.weight", rows=v_one)
            la.out_proj = lin(b + "ssm_out.weight", cols=v_rows)
            la.conv1d.weight = conv(take(b + "ssm_conv1d.weight"), qkv_rows)
            la.A_log = mx.array(np.log(-to_float(take(b + "ssm_a"))[v_one]))
            la.dt_bias = mx.array(np.ascontiguousarray(to_float(take(b + "ssm_dt.bias"))[v_one]))
            la.norm.weight = vec(b + "ssm_norm.weight")
        else:
            sa = layer.self_attn
            sa.q_proj = lin(b + "attn_q.weight")
            sa.k_proj = lin(b + "attn_k.weight")
            sa.v_proj = lin(b + "attn_v.weight")
            sa.o_proj = lin(b + "attn_output.weight")
            sa.q_norm.weight = vec(b + "attn_q_norm.weight")
            sa.k_norm.weight = vec(b + "attn_k_norm.weight")
            both = np.concatenate([to_float(take(b + "indexer.q_proj.weight")),
                                   to_float(take(b + "indexer.k_proj.weight"))])
            sa.indexer.index_qk_proj.weight = mx.array(both)
            sa.indexer.q_layernorm.weight = vec(b + "indexer.q_norm.weight")
            sa.indexer.k_layernorm.weight = vec(b + "indexer.k_norm.weight")

        mlp = layer.mlp
        mlp.gate.weight = vec(b + "ffn_gate_inp.weight")
        mlp.shared_expert_gate.weight = vec(b + "ffn_gate_inp_shexp.weight")[None]
        routed = [take(b + f"ffn_{part}_exps.weight") for part in ("gate", "up", "down")]
        if store is not None:
            mlp.switch_mlp = PagedSwitchGLU(i, store, *routed)
        else:
            mlp.switch_mlp = BlockSwitchGLU(*(experts(t) for t in routed))
        mlp.shared_expert.gate_proj = lin(b + "ffn_gate_shexp.weight")
        mlp.shared_expert.up_proj = lin(b + "ffn_up_shexp.weight")
        mlp.shared_expert.down_proj = lin(b + "ffn_down_shexp.weight")
        residual(layer.attn_hyper_connection, b + "hc_attn")
        residual(layer.mlp_hyper_connection, b + "hc_ffn")

        if layer.ple is not None:
            ple = layer.ple
            ple.key_proj = lin(b + "ple_key.weight")
            ple.value_proj = lin(b + "ple_value.weight")
            ple.norm_key.weight = vec(b + "ple_norm_key.weight")
            ple.norm_query.weight = vec(b + "ple_norm_query.weight")
            ple.norm_conv.weight = vec(b + "ple_norm_conv.weight")
            ple.conv1d.weight = conv(take(b + "ple_conv1d.weight"))
            emb = ple.ple_embedding
            # the hash must land on the rows the file was built with
            for ours, key in ((emb._mults, "ple.layer_multipliers"), (emb._sizes, "ple.head_vocab_sizes"),
                              (emb._offsets, "ple.head_offsets")):
                if ours.tolist() != list(meta[k + key]):
                    raise ValueError(f"the n-gram hash constants do not match the file's {key}")
            emb.ngram_embedding = PleTable(take("per_layer_token_embd.weight"))
            ple.ple_embedding = HostNgram(emb, emb.ngram_embedding)
        layer.set_dtype(dtype)
        mx.eval(layer.parameters())

    model.model.embed_tokens = embedding(take("token_embd.weight"), dtype, dense_blocks)
    model.lm_head = linear(take("output.weight"))  # never blocks: a guess takes some of its rows (rows.py)
    residual(model.model.hyper_connection_mixer, "output_hc", inject=False)
    model.set_dtype(dtype)
    mx.eval(model.parameters())

    missed = sorted(set(tensors) - used)
    if missed:
        raise ValueError(f"{len(missed)} tensors of the file were not loaded, e.g. {missed[:5]}")
    constants = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")
    left = [name for name, v in tree_flatten(model.parameters())
            if id(v) in placeholders and not name.endswith(constants)]
    if left:
        raise ValueError(f"{len(left)} parameters of the model were left at their random start, e.g. {left[:5]}")
    mx.clear_cache()
    if store is None:
        # MLX keeps the buffers of freed arrays for the next ones, without limit unless told. A chunk of prompt
        # frees buffers of sizes the next layer does not ask for again: 37 GB after 2,048 tokens, 65 GB after
        # 7,000, beside 41 GB of weights. Reading then went at 60-100 tok/s, not 400, once the machine had no
        # memory to spare; and with a limit the first long prompt after a start is as fast as the next.
        mx.set_cache_limit(POOL)
    model.eval()
    model.expert_store, model.keeps_floats = store, keep
    return model, meta
