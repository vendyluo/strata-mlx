"""Draft-and-verify must write what plain greedy decoding writes, whatever the guesses are.

A small random model of the real architecture (delta-net and attention layers, experts, hyper-connections, the
n-gram layer), in float32 so that a window and single tokens round the same way.
"""

import random

import mlx.core as mx
import pytest
from mlx_lm.models import qwen4_exp

from strata_mlx.engine import Engine
from strata_mlx.mtp import Mtp

TEXT = dict(
    hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=128,
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32,
    linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
    hc_count=4, hc_lowrank=8, indexer_n_heads=2, indexer_head_dim=16,
    ngram_vocab_size_base=1000, split_ngram_parts=4, ple_embed_dim=64, ple_layer_ids=[2], eos_token_id=5,
)
PROMPT = [7, 3, 99, 41, 12, 64, 8]
N = 48


@pytest.fixture(scope="module")
def model():
    mx.random.seed(3)
    model = qwen4_exp.Model(qwen4_exp.ModelArgs(text_config=TEXT))
    mx.eval(model.parameters())
    model.eval()
    return model


@pytest.fixture(scope="module")
def answer(model):
    return Engine(model).generate(PROMPT, N)


@pytest.mark.parametrize("window, wrong", [(1, 0.0), (3, 0.0), (3, 0.4), (7, 0.25), (3, 1.0)])
def test_any_guesses_give_the_plain_answer(model, answer, window, wrong):
    rng = random.Random(window)
    engine = Engine(model)
    out = [engine.prefill(PROMPT)]
    while len(out) < N:
        truth = answer[len(out) : len(out) + window]
        engine.guesses = [t if rng.random() >= wrong else (t + 1) % TEXT["vocab_size"] for t in truth]
        out += engine.step()
    assert out[:N] == answer
    kept = engine.stats.by_position
    if wrong == 0.0:
        assert engine.stats.accepted == engine.stats.drafted
    elif wrong == 1.0:
        assert engine.stats.accepted == 0
    else:
        assert sum(1 for k in kept[: window + 1] if k) >= 3  # short, middling and whole windows all happened


def test_the_draft_head_changes_speed_not_the_answer(model, answer):
    mx.random.seed(4)
    mtp = Mtp(model.args.text)
    mx.eval(mtp.parameters())
    engine = Engine(model, mtp, drafts=3)
    assert engine.generate(PROMPT, N) == answer
    assert engine.mtp_cache.offset == len(PROMPT) + engine.stats.tokens  # one real cell per decoded position


@pytest.mark.parametrize("keep", ["always", "never", "by turns", "as the policy says"])
def test_however_many_guesses_the_head_makes_the_answer_is_the_plain_one(model, answer, keep):
    mx.random.seed(4)
    mtp = Mtp(model.args.text)
    mx.eval(mtp.parameters())
    engine = Engine(model, mtp)
    calls = iter(range(10**6))
    if keep == "always":
        engine.policy.worth_guessing = engine.policy.worth_keeping = lambda sure, guesses: True
    elif keep == "never":
        engine.policy.worth_keeping = lambda sure, guesses: False
    elif keep == "by turns":
        engine.policy.worth_guessing = lambda sure, guesses: next(calls) % 3 != 0
        engine.policy.worth_keeping = lambda sure, guesses: next(calls) % 4 != 0
    assert engine.generate(PROMPT, N) == answer
    assert engine.mtp_cache.offset == len(PROMPT) + engine.stats.tokens  # one real cell per decoded position
    if keep == "always":
        assert engine.stats.drafted >= 6 * (engine.stats.steps - 3)  # the last passes are cut to what is left to write
        assert len(engine.guess_sure) == 7 and all(a >= b for a, b in zip(engine.guess_sure, engine.guess_sure[1:]))
    elif keep == "never":
        assert engine.stats.drafted <= 3 * engine.stats.steps  # the first three are made in any case


# ---- sampling, lookup, and a conversation read only once

from strata_mlx.sampler import Sampling  # noqa: E402

TURN = 9
SAMPLED = dict(temperature=1.0, top_k=20, top_p=0.95, seed=17)


def test_a_seed_writes_the_same_text_with_and_without_drafts(model):
    plain = Engine(model).generate(PROMPT, N, sampling=Sampling(**SAMPLED))
    assert plain != Engine(model).generate(PROMPT, N)  # it does sample
    mx.random.seed(4)
    mtp = Mtp(model.args.text)
    mx.eval(mtp.parameters())
    assert Engine(model, mtp, drafts=3).generate(PROMPT, N, sampling=Sampling(**SAMPLED)) == plain
    other = Engine(model).generate(PROMPT, N, sampling=Sampling(**{**SAMPLED, "seed": 18}))
    assert other != plain


def test_penalties_apply_in_a_window_as_they_do_token_by_token(model, answer):
    s = dict(penalty_last_n=16, penalty_repeat=1.3, penalty_freq=0.2, penalty_present=0.1)
    plain = Engine(model).generate(PROMPT, N, sampling=Sampling(**s))
    assert plain != answer
    rng = random.Random(1)
    engine = Engine(model)
    engine.sampling = Sampling(**s)
    out = [engine.prefill(PROMPT)]
    while len(out) < N:
        truth = plain[len(out) : len(out) + 3]
        engine.guesses = [t if rng.random() >= 0.3 else (t + 1) % TEXT["vocab_size"] for t in truth]
        out += engine.step()
    assert out[:N] == plain


def test_lookup_guesses_do_not_change_the_answer(model):
    prompt = [7, 3, 99, 41, 12, 64, 8, 20, 21] * 5  # text that repeats: lookup has something to copy
    plain = Engine(model).generate(prompt, N)
    engine = Engine(model, lookup=True)
    assert engine.generate(prompt, N) == plain


def test_reading_in_chunks_is_reading_at_once(model, answer):
    assert Engine(model, chunk=3).generate(PROMPT, N) == answer


@pytest.mark.parametrize("second", ["continues", "edits the last message", "edits the first message", "repeats",
                                    "is a prefix"])
def test_a_follow_up_reads_only_what_is_new_and_writes_what_a_fresh_start_writes(model, second):
    first = [TURN, 30, 31, 32, 33, TURN, 40, 41, 42, 43, 44, TURN, 50]
    engine = Engine(model, chunk=4, turn_token=TURN)
    reply = engine.generate(first, 12)
    prompt = {
        "continues": first + reply + [TURN, 60, 61, 62, TURN, 50],
        "edits the last message": first[:8] + [77, 78, TURN, 50],
        "edits the first message": [TURN, 30, 99] + first[3:],
        "repeats": first,
        "is a prefix": first[:7],
    }[second]
    read = []
    engine.sampling = Sampling()
    start = engine.resume(prompt)
    out = [engine.read(prompt, on_chunk=lambda done, total: read.append(done))]
    while len(out) < 12:
        out += engine.step()
    assert out[:12] == Engine(model, turn_token=TURN).generate(prompt, 12)
    assert engine.tokens[: len(prompt)] == prompt
    expected_start = {"continues": len(first) + len(reply) - 1, "edits the last message": 5,
                      "edits the first message": 0, "repeats": len(first) - 1, "is a prefix": 5}[second]
    assert start == expected_start


def test_going_back_takes_the_draft_head_back_too(model):
    first = [TURN, 30, 31, 32, 33, TURN, 40, 41, 42, 43, 44, TURN, 50]
    edited = first[:8] + [77, 78, TURN, 50]
    mx.random.seed(4)
    mtp = Mtp(model.args.text)
    mx.eval(mtp.parameters())
    engine = Engine(model, mtp, drafts=3, turn_token=TURN)
    engine.generate(first, 12)
    assert engine.generate(edited, 12) == Engine(model, turn_token=TURN).generate(edited, 12)
    assert engine.mtp_cache.offset == len(engine.tokens)


def test_lookup_windows_are_checked_like_any_other(model, answer):
    """Lookup guesses forced on every step, a third of them wrong."""
    rng = random.Random(2)
    engine = Engine(model, lookup=True)
    engine.policy.choose = lambda head_window, lookup_len, match: lookup_len
    out = [engine.prefill(PROMPT)]
    while len(out) < N:
        truth = answer[len(out) : len(out) + 5]
        found = [t if rng.random() >= 0.33 else (t + 1) % TEXT["vocab_size"] for t in truth]
        engine.lookup.propose = lambda pending, max_k, found=found: (found, 12)
        out += engine.step()
    assert out[:N] == answer
    assert engine.stats.lookup_steps == engine.stats.steps > 0
    assert 0 < engine.stats.accepted < engine.stats.drafted


def test_the_fused_pass_writes_what_the_models_own_operations_write(model, answer):
    assert Engine(model, fused=False).generate(PROMPT, N) == answer


@pytest.mark.parametrize("fuse_upto", [0, 8])
def test_either_form_of_the_hyper_connections_writes_the_same(model, answer, fuse_upto):
    assert Engine(model, fuse_upto=fuse_upto).generate(PROMPT, N) == answer


def test_warming_up_leaves_the_conversation_alone(model, answer):
    engine = Engine(model)
    first = engine.prefill(PROMPT)
    engine.warm()
    out = [first]
    while len(out) < N:
        out += engine.step()
    assert out[:N] == answer


# ---- experts kept as the file's blocks (blocks.py)

import numpy as np  # noqa: E402
from mlx_lm.models.switch_layers import SwitchLinear  # noqa: E402

from strata_mlx.blocks import BlockSwitchLinear, dequantize  # noqa: E402


def _with_iq4_nl_experts(as_blocks: bool):
    """The small model with its experts drawn as random IQ4_NL blocks: kept as blocks, or unpacked to floats."""
    mx.random.seed(3)
    model = qwen4_exp.Model(qwen4_exp.ModelArgs(text_config=TEXT))
    rng = np.random.default_rng(11)
    for layer in model.model.layers:
        glu = layer.mlp.switch_mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            experts, rows, cols = getattr(glu, name).weight.shape
            b = rng.integers(0, 256, (experts * rows * cols // 32, 18), dtype=np.uint8)
            b[:, :2] = rng.uniform(0.0005, 0.002, (b.shape[0], 1)).astype("<f2").view(np.uint8)
            data = mx.array(b.reshape(-1))
            if as_blocks:
                setattr(glu, name, BlockSwitchLinear(data, "IQ4_NL", (experts, rows, cols)))
            else:
                plain = SwitchLinear(cols, rows, experts, bias=False)
                plain.weight = dequantize(data, "IQ4_NL", (experts, rows, cols))
                setattr(glu, name, plain)
    mx.eval(model.parameters())
    model.eval()
    return model


@pytest.fixture(scope="module")
def block_model():
    return _with_iq4_nl_experts(True)


def test_experts_as_blocks_write_what_their_weights_as_floats_write(block_model):
    assert Engine(block_model).generate(PROMPT, N) == Engine(_with_iq4_nl_experts(False)).generate(PROMPT, N)


@pytest.mark.parametrize("window, wrong", [(3, 0.4), (7, 0.25)])
def test_windows_over_block_experts_give_the_plain_answer(block_model, window, wrong):
    answer = Engine(block_model).generate(PROMPT, N)
    rng = random.Random(window)
    engine = Engine(block_model)
    out = [engine.prefill(PROMPT)]
    while len(out) < N:
        truth = answer[len(out) : len(out) + window]
        engine.guesses = [t if rng.random() >= wrong else (t + 1) % TEXT["vocab_size"] for t in truth]
        out += engine.step()
    assert out[:N] == answer


def test_a_long_prompt_over_block_experts_reads_as_token_by_token(block_model):
    prompt = [(7 * i + 3) % TEXT["vocab_size"] for i in range(90)]
    whole = Engine(block_model)
    BlockSwitchLinear.unpack_from, before = 100, BlockSwitchLinear.unpack_from  # 180 pairs: the unpacked path
    try:
        first = whole.read(prompt)
    finally:
        BlockSwitchLinear.unpack_from = before
    assert first == Engine(block_model, chunk=1).read(prompt)


# ---- a conversation on disk

def _draft_engine(model):
    mx.random.seed(4)
    mtp = Mtp(model.args.text)
    mx.eval(mtp.parameters())
    return Engine(model, mtp, drafts=3, lookup=True, chunk=4, turn_token=TURN)


@pytest.mark.parametrize("second", ["continues", "edits the last message"])
def test_a_restored_conversation_goes_on_as_the_saved_one_would(model, tmp_path, second):
    first = [TURN, 30, 31, 32, 33, TURN, 40, 41, 42, 43, 44, TURN, 50]
    saved = _draft_engine(model)
    reply = saved.generate(first, 12)
    path = str(tmp_path / "talk.bin")
    assert saved.save(path) > 0
    prompt = first + reply + [TURN, 60, 61, 62, TURN, 50] if second == "continues" else first[:8] + [77, 78, TURN, 50]

    restored = _draft_engine(model)
    restored.generate([TURN, 99, 98, 97], 5)  # it held something else
    restored.restore(path)
    assert restored.tokens == saved.tokens
    assert restored.resume(prompt) == saved.resume(prompt) > 0  # reads only what is new
    assert restored.generate(prompt, 12) == saved.generate(prompt, 12) == _draft_engine(model).generate(prompt, 12)


def test_saving_twice_replaces_the_file_and_an_empty_conversation_can_be_saved(model, tmp_path):
    engine, path = Engine(model), str(tmp_path / "talk.bin")
    engine.save(path)
    engine.generate(PROMPT, 6)
    engine.save(path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["talk.bin"]
    other = Engine(model)
    other.restore(path)
    assert other.tokens == engine.tokens and len(other.tokens) >= len(PROMPT)


def test_a_file_of_another_model_or_no_session_at_all_is_refused_and_changes_nothing(model, tmp_path):
    engine, path = Engine(model), str(tmp_path / "talk.bin")
    engine.identity = "one"
    engine.generate(PROMPT, 6)
    engine.save(path)
    other = Engine(model)
    other.identity = "another"
    held = other.generate([7, 8, 9], 4), list(other.tokens)
    with pytest.raises(ValueError, match="other model files"):
        other.restore(path)
    (tmp_path / "junk.bin").write_bytes(b"not a session" * 100)
    with pytest.raises(ValueError):
        other.restore(str(tmp_path / "junk.bin"))
    with pytest.raises(OSError):
        other.restore(str(tmp_path / "missing.bin"))
    assert other.tokens == held[1]
    assert other.generate([7, 8, 9], 4) == held[0]


# ---- experts read from a file as tokens ask for them (paged.py)

from strata_mlx import paged  # noqa: E402
from strata_mlx.gguf import Tensor  # noqa: E402


def _paged(folder, budget: int):
    """The model of `_with_iq4_nl_experts`, its experts in a file and at most `budget` bytes of them in memory."""
    mx.random.seed(3)
    model = qwen4_exp.Model(qwen4_exp.ModelArgs(text_config=TEXT))
    rng = np.random.default_rng(11)
    store = paged.ExpertStore(budget, threads=2)
    path, at = str(folder / "experts.bin"), 0
    with open(path, "wb") as f:
        for i, layer in enumerate(model.model.layers):
            glu, tensors = layer.mlp.switch_mlp, {}
            for name in ("gate_proj", "up_proj", "down_proj"):  # the order the other model draws them in
                experts, rows, cols = getattr(glu, name).weight.shape
                b = rng.integers(0, 256, (experts * rows * cols // 32, 18), dtype=np.uint8)
                b[:, :2] = rng.uniform(0.0005, 0.002, (b.shape[0], 1)).astype("<f2").view(np.uint8)
                f.write(b"\0" * 7)  # tensors do not start where the file does
                at += 7
                tensors[name] = Tensor(name, (experts, rows, cols), "IQ4_NL", b.reshape(-1), path, at)
                f.write(b.tobytes())
                at += b.size
            layer.mlp.switch_mlp = paged.PagedSwitchGLU(i, store, tensors["gate_proj"], tensors["up_proj"],
                                                        tensors["down_proj"])
    mx.eval(model.parameters())
    model.eval()
    model.expert_store = store
    return model


ONE_EXPERT = 3 * 32 * 64 // 32 * 18  # bytes of one of the small model's experts


@pytest.mark.parametrize("budget", [0, 5 * ONE_EXPERT, 1 << 30])
def test_experts_read_from_a_file_write_what_experts_in_memory_write(block_model, tmp_path, budget):
    model = _paged(tmp_path, budget)
    store = model.expert_store
    assert Engine(model).generate(PROMPT, N) == Engine(block_model).generate(PROMPT, N)
    assert store.misses > 0 and store.held <= max(budget, 2 * ONE_EXPERT)
    if budget >= 1 << 30:
        assert store.hits > store.misses  # eight experts a layer: soon all of them are in memory
    elif budget == 0:
        assert store.hits == 0


def test_windows_over_experts_read_from_a_file_give_the_plain_answer(block_model, tmp_path):
    answer = Engine(block_model).generate(PROMPT, N)
    rng = random.Random(5)
    engine = Engine(_paged(tmp_path, 6 * ONE_EXPERT))
    out = [engine.prefill(PROMPT)]
    while len(out) < N:
        truth = answer[len(out) : len(out) + 5]
        engine.guesses = [t if rng.random() >= 0.3 else (t + 1) % TEXT["vocab_size"] for t in truth]
        out += engine.step()
    assert out[:N] == answer


def test_a_prompt_reads_whole_layers_and_keeps_none_of_them(block_model, tmp_path, monkeypatch):
    monkeypatch.setattr(paged, "WHOLE_FROM", 8)  # the small model has eight experts a layer
    monkeypatch.setattr(paged, "KEEP_UPTO", 40)
    model = _paged(tmp_path, 1 << 30)
    prompt = [(7 * i + 3) % TEXT["vocab_size"] for i in range(90)]
    assert Engine(model).read(prompt) == Engine(block_model, chunk=1).read(prompt)
    # the prompt's last token is read by itself (a checkpoint goes before it): its two experts a layer stay
    assert model.expert_store.held == len(model.model.layers) * 2 * ONE_EXPERT
    assert model.expert_store.bytes_read == len(model.model.layers) * (8 + 2) * ONE_EXPERT  # each layer whole, once


def test_a_prompts_next_layer_is_read_while_this_one_computes(block_model, tmp_path, monkeypatch):
    monkeypatch.setattr(paged, "WHOLE_FROM", 8)
    monkeypatch.setattr(paged, "KEEP_UPTO", 40)
    monkeypatch.setattr(paged, "READ_AHEAD_FROM", 41)
    prompt = [(7 * i + 3) % TEXT["vocab_size"] for i in range(90)]
    want = Engine(block_model, chunk=1).read(prompt)
    for ahead in (True, False):
        folder = tmp_path / str(ahead)
        folder.mkdir()
        model = _paged(folder, 1 << 30)
        store = model.expert_store
        store.read_ahead, pools = ahead, []
        fill = store._fill
        monkeypatch.setattr(store, "_fill", lambda *a: pools.append(a[-1]) or fill(*a))
        assert Engine(model).read(prompt) == want
        layers = len(model.model.layers)
        # the first layer is read when it asks; with read_ahead every other one while the layer before computes
        assert pools == [store.pool] + [store.ahead_pool if ahead else store.pool] * (layers - 1)
        assert store.bytes_read == layers * (8 + 2) * ONE_EXPERT and store.ahead is None


def test_a_short_chunk_over_experts_in_memory_reads_nothing_ahead(block_model, tmp_path, monkeypatch):
    # a chunk that keeps nothing and is shorter than READ_AHEAD_FROM: the layers it uses are all in memory, and the
    # next layer's other experts must not be read for it (a chunk of 65 tokens once read 502 of them a layer)
    monkeypatch.setattr(paged, "KEEP_UPTO", 40)
    monkeypatch.setattr(paged, "SEPARATE_UPTO", 10)
    model = _paged(tmp_path, 1 << 30)
    store, engine = model.expert_store, Engine(model)
    prompt = [(3 * i + 2) % TEXT["vocab_size"] for i in range(30)]
    engine.generate(prompt, N)  # its experts in memory
    started = []
    read_next = store.read_next
    monkeypatch.setattr(store, "read_next", lambda layer: started.append(layer) or read_next(layer))
    engine.reset()
    store.reset_stats()
    engine.read(prompt)
    assert not started and store.misses == 0 and store.bytes_read == 0 and store.ahead is None


def test_a_layer_read_whole_is_the_files_bytes_in_pieces_of_any_alignment(tmp_path):
    # tensors larger than a piece, starting off a page, with experts in memory between runs read from the file
    rng = np.random.default_rng(4)
    data = rng.integers(0, 256, 9 * paged.PIECE, dtype=np.uint8)
    (tmp_path / "f").write_bytes(data.tobytes())
    one = 3 * paged.PIECE // 7 + 5  # bytes of one expert of one tensor
    tensors = tuple(Tensor(f"t{i}", (6, one), "IQ4_NL", np.zeros(6 * one, np.uint8), str(tmp_path / "f"), at)
                    for i, at in enumerate((12345, 3 * paged.PIECE + 1, 6 * paged.PIECE + 4096)))
    store = paged.ExpertStore(1 << 30, threads=3)
    for e in (0, 3, 4):  # in memory: what is copied from there must be what is there
        store.lru[(7, e)] = (tuple(mx.array(np.full(one, e + 1, np.uint8)) for _ in tensors), 3 * one)
    got = store.whole(tensors, 7)
    for t, g in zip(tensors, got):
        want = data[t.offset : t.offset + 6 * one].reshape(6, one).copy()
        want[[0, 3, 4]] = np.array([1, 4, 5], np.uint8)[:, None]
        np.testing.assert_array_equal(np.array(g).reshape(6, one), want)
    assert store.bytes_read == 3 * 3 * one


def test_a_prompt_reads_only_the_experts_not_in_memory(block_model, tmp_path, monkeypatch):
    monkeypatch.setattr(paged, "WHOLE_FROM", 8)
    monkeypatch.setattr(paged, "KEEP_UPTO", 40)
    prompt = [(5 * i + 1) % TEXT["vocab_size"] for i in range(90)]
    want = Engine(block_model, chunk=1).read(prompt)
    model = _paged(tmp_path, 1 << 30)
    store, engine = model.expert_store, Engine(model)
    engine.generate(PROMPT, N)  # an answer leaves its experts in memory
    layers = len(model.model.layers)
    for layer, e in [(layer, e) for layer in range(layers) for e in (0, 3, 4, 7) if layer % 2 or e < 4]:
        store.held -= store.lru.pop((layer, e))[1]  # runs held and not, at either end of a layer and inside
    held = sum(1 for layer in range(layers) for e in range(8) if (layer, e) in store.lru)
    assert 0 < held < 8 * layers
    engine.reset()
    store.reset_stats()
    assert engine.read(prompt) == want
    # the chunk reads what was not in memory; the prompt's last token by itself reads its own missing experts
    assert (8 * layers - held) * ONE_EXPERT <= store.bytes_read < 8 * layers * ONE_EXPERT


def test_a_window_spread_over_several_kernel_calls_gives_the_plain_answer(block_model, tmp_path, monkeypatch):
    # with ten experts a token, as in the real model, a kernel binds two tokens' worth: a window is several calls
    monkeypatch.setattr(paged, "MAX_SEPARATE", 3)  # here: one token a call
    answer = Engine(block_model).generate(PROMPT, N)
    rng = random.Random(6)
    engine = Engine(_paged(tmp_path, 6 * ONE_EXPERT))
    out = [engine.prefill(PROMPT)]
    while len(out) < N:
        truth = answer[len(out) : len(out) + 4]
        engine.guesses = [t if rng.random() >= 0.3 else (t + 1) % TEXT["vocab_size"] for t in truth]
        out += engine.step()
    assert out[:N] == answer
    assert engine.stats.by_position[4] > 0  # whole windows of five were kept


def test_the_most_used_experts_are_in_memory_before_the_first_request(block_model, tmp_path):
    model = _paged(tmp_path, 5 * ONE_EXPERT)
    store = model.expert_store
    counts = np.zeros((len(model.model.layers), 8), dtype=np.uint16)
    used = {(0, 3): 9, (2, 7): 8, (1, 1): 7, (3, 0): 6, (0, 5): 5, (2, 2): 4, (1, 6): 1}
    for (layer, e), n in used.items():
        counts[layer, e] = n
    assert store.preload(counts) == 5
    assert list(store.lru) == [(0, 5), (3, 0), (1, 1), (2, 7), (0, 3)]  # the least used is the first to go
    assert store.held == 5 * ONE_EXPERT
    assert store.preload(counts) == 0  # no room left
    assert store.preload(np.zeros((2, 8), dtype=np.uint16)) == 0  # counts of another model
    assert Engine(model).generate(PROMPT, N) == Engine(block_model).generate(PROMPT, N)


def test_a_machine_counts_what_its_answers_use(block_model, tmp_path):
    model = _paged(tmp_path, 1 << 30)
    store = model.expert_store
    Engine(model).generate(PROMPT, 12)
    layers = len(model.model.layers)
    counts = store.profile()
    assert counts.shape == (layers, 8) and counts.sum() == sum(store.use.values()) > 0
    before = np.full((layers, 8), 7, dtype=np.uint32)
    assert (store.profile(before) == counts + 3).all()  # what was counted before weighs half
    path = str(tmp_path / "cache" / "use.npy")
    paged.save_profile(path, counts)
    assert (paged.load_profile(path) == counts).all()
    assert paged.load_profile(str(tmp_path / "none.npy")).shape == (48, 512)  # the profile that comes with the engine
    store.clear()
    assert store.held == 0 and store.preload(counts) == int((counts > 0).sum())


# ---- dense layers kept as the file's blocks (dense.BlockLinear, BlockEmbedding)

import itertools  # noqa: E402

import mlx.nn as nn  # noqa: E402

from strata_mlx.blocks import BLOCK  # noqa: E402
from strata_mlx.dense import AS_BLOCKS  # noqa: E402
from strata_mlx.gguf import Tensor  # noqa: E402
from strata_mlx.load import embedding, linear, v_heads_grouped  # noqa: E402

# rows of 256 weights, a block of these types, wherever a dense layer of the real file is one of them
WIDE = dict(TEXT, hidden_size=256, num_attention_heads=8, linear_value_head_dim=64)
_D = {"Q3_K": (108, 5e-4, 1e-3), "IQ4_XS": (0, 1.5e-5, 3e-5)}  # where a block's scale is, and a range for it


def _with_dense(as_blocks: bool):
    """A wider small model with its projections and its embedding drawn as random Q3_K and IQ4_XS blocks, the
    delta net's reordered as the loader reorders them: kept as blocks, or unpacked to floats."""
    mx.random.seed(3)
    model = qwen4_exp.Model(qwen4_exp.ModelArgs(text_config=WIDE))
    a, rng, kinds = model.args.text, np.random.default_rng(13), itertools.cycle(AS_BLOCKS)
    n_k, n_v, d_v = a.linear_num_key_heads, a.linear_num_value_heads, a.linear_value_head_dim
    v_rows = v_heads_grouped(n_k, n_v // n_k, d_v)
    qk = 2 * n_k * a.linear_key_head_dim
    qkv_rows = np.concatenate([np.arange(qk), qk + v_rows])

    def drawn(rows, cols):
        kind = next(kinds)
        at, low, high = _D[kind]
        b = rng.integers(0, 256, (rows * cols // BLOCK[kind][0], BLOCK[kind][1]), dtype=np.uint8)
        b[:, at : at + 2] = rng.uniform(low, high, (len(b), 1)).astype("<f2").view(np.uint8)
        return Tensor(kind, (rows, cols), kind, b.reshape(-1))

    def floats(t, rows=None, cols=None):
        w = np.array(dequantize(mx.array(t.data), t.type, t.shape))
        w = w if rows is None else w[rows]
        return mx.array(np.ascontiguousarray(w if cols is None else w[:, cols]))

    def put(owner, name, rows=None, cols=None):
        t = drawn(*getattr(owner, name).weight.shape)
        if as_blocks:
            setattr(owner, name, linear(t, rows, cols, AS_BLOCKS))
        else:
            plain = nn.Linear(1, 1, bias=False)
            plain.weight = floats(t, rows, cols)
            setattr(owner, name, plain)

    for layer in model.layers:
        if layer.layer_type == "linear_attention":
            put(layer.linear_attn, "in_proj_qkv", rows=qkv_rows)
            put(layer.linear_attn, "in_proj_z", rows=v_rows)
            put(layer.linear_attn, "out_proj", cols=v_rows)
        else:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                put(layer.self_attn, name)
        put(layer.mlp.shared_expert, "gate_proj")
        put(layer.mlp.shared_expert, "up_proj")
    t = drawn(a.vocab_size, a.hidden_size)
    if as_blocks:
        model.model.embed_tokens = embedding(t, mx.float32, AS_BLOCKS)
    else:
        model.model.embed_tokens.weight = floats(t)
    mx.eval(model.parameters())
    model.eval()
    return model


@pytest.fixture(scope="module")
def dense_model():
    return _with_dense(True)


def test_dense_layers_as_blocks_write_what_their_weights_as_floats_write(dense_model):
    answer = Engine(dense_model).generate(PROMPT, N)
    assert answer == Engine(_with_dense(False)).generate(PROMPT, N)
    assert len(set(answer)) > 4  # an answer of one token over and over would say little


@pytest.mark.parametrize("window, wrong", [(3, 0.4), (7, 0.25)])
def test_windows_over_dense_blocks_give_the_plain_answer(dense_model, window, wrong):
    answer = Engine(dense_model).generate(PROMPT, N)
    rng = random.Random(window)
    engine = Engine(dense_model)
    out = [engine.prefill(PROMPT)]
    while len(out) < N:
        truth = answer[len(out) : len(out) + window]
        engine.guesses = [t if rng.random() >= wrong else (t + 1) % TEXT["vocab_size"] for t in truth]
        out += engine.step()
    assert out[:N] == answer


def test_a_long_prompt_over_dense_blocks_reads_as_token_by_token(dense_model):
    prompt = [(7 * i + 3) % TEXT["vocab_size"] for i in range(90)]
    assert Engine(dense_model).read(prompt) == Engine(dense_model, chunk=1).read(prompt)
    assert Engine(dense_model).read(prompt) == Engine(_with_dense(False)).read(prompt)
