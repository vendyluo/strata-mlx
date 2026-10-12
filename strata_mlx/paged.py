"""Experts read from the SSD as tokens ask for them, for a machine whose memory does not hold them all.

A token uses ten of a layer's 512 experts, and over a conversation they are not the same few (bench/expert_use.py),
so a small memory means reading from the disk all the time, and what matters is how. Here:

  * every expert in memory is three arrays of its own (its gate, up and down tensors, as the file's blocks), in
    one store shared by all layers, the least recently used dropped first when the budget is full;
  * a layer waits for its router's choice to reach the CPU (0.2 ms), reads what is missing with a few parallel
    reads past the page cache, and hands each (token, expert) pair's arrays to the kernels as they are;
  * a prompt, whose tokens use nearly every expert of a layer, reads the layer's three tensors whole instead, in
    file order, and keeps none of it;
  * at the start the budget is filled with the experts used most: by this machine's own answers so far, and
    before there are any, by a profile of seven conversations that comes with the engine (expert_use.npy).

Measured on the way, a token at a time on an M4 Max with nothing to read: 23 ms a pass with every expert in one
array; 34 with the wait at every layer; 42 if the ten experts are first joined into a stack, which is why the
kernels take them separately. Replacing one expert inside a 1.4 GB array copies the array (2 ms), which is why
each expert is its own array.
"""

import fcntl
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
from mlx_lm.models.activations import swiglu

from .blocks import BLOCK, MAX_SEPARATE, BlockSwitchLinear, matvec_separate, switch_glu
from .gguf import Tensor

WHOLE_FROM = 384  # experts of a layer in one call from which its tensors are read whole
KEEP_UPTO = 640  # (token, expert) pairs in one call up to which what was read stays in memory: answers, not prompts
PAGE, PIECE = 16384, 1 << 20
SEPARATE_UPTO = 80  # pairs up to which each brings its own arrays to the kernels: a window of an answer


# What a machine's memory is set aside for before experts get the rest, in bytes: macOS and whatever else runs;
# and what this process takes beyond its weights. Measured with Q2_0: 1.3 GB that is not arrays, 2.1 GB at the peak
# of reading a 1,500-token prompt (a layer's tensors read whole, the activations), MLX's pool of 0.5.
# #6: macOS had 4.5 GB until a 32 GB MacBook Air was heard from: with nothing else open it holds 6.1 GB itself, and
# the machine swapped until the experts had 16 GB and not 20.5, which is what 8.5 GB gives them. One machine.
SYSTEM, WORKING = int(8.5e9), int(4.0e9)
DRAFT_HEAD = int(1.6e9)
# The dense tensors as MLX's arrays take this many times their size in the file: several are held at 8 bits. So a
# machine that holds the model whole has them. One that reads experts from the SSD keeps two of those types as
# the file's blocks (dense.BlockLinear), 1.7 GB less with the Q2_0 file; its store finds that out when it fits
# itself to what is loaded (`fit`), and `check.py` counts it.
DENSE_AS_ARRAYS = 1.6
LEAST = int(0.5e9)  # of experts in memory, however little is left for them


def machine_memory() -> int:
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def sizes(files: list) -> tuple:
    """(experts, dense tensors) of a model's shards, in bytes as the files have them. The dense tensors are
    those of the first shard: the n-gram table, in the second, stays on the disk."""
    experts = sum(t.data.nbytes for f in files for name, t in f.tensors.items() if name.endswith("_exps.weight"))
    dense = sum(t.data.nbytes for name, t in files[0].tensors.items() if not name.endswith("_exps.weight"))
    return experts, dense


def all_fit(machine: int, experts: int, dense: int, floats: int = 0) -> bool:
    """Whether a machine of that much memory holds the whole model, sizes as in the file; with `floats`, and the
    weights of its Q6_K layers as floats beside it, kept for prompts (dense.ScaledLinear, #3)."""
    return machine - SYSTEM - WORKING >= experts + DENSE_AS_ARRAYS * dense + DRAFT_HEAD + floats


class ExpertStore:
    """The experts in memory, within a budget of bytes."""

    def __init__(self, budget: int, threads: int = 8):
        self.budget, self.held = budget, 0
        # MLX keeps the buffers of freed arrays for the next ones, without limit unless told: 4 GB after a hundred
        # tokens. Enough of them to take the experts that come and go, no more.
        mx.set_cache_limit(512 << 20)
        self.lru = OrderedDict()  # (layer, expert) -> (arrays, bytes)
        self.layers = {}  # layer -> its three tensors
        self.use = {}  # (layer, expert) -> times an answer asked for it since the start
        self.fds = {}
        self.pool = ThreadPoolExecutor(threads)
        self.reset_stats()

    def fit(self, machine: int) -> int:
        """Set the budget to what a machine of that much memory has left, everything else being loaded now."""
        self.budget = max(LEAST, machine - SYSTEM - WORKING - (mx.get_active_memory() - self.held))
        return self.budget

    def reset_stats(self):
        self.hits = self.misses = self.bytes_read = 0
        self.read_s = 0.0

    def _fd(self, path: str) -> int:
        if path not in self.fds:
            self.fds[path] = os.open(path, os.O_RDONLY)
            fcntl.fcntl(self.fds[path], fcntl.F_NOCACHE, 1)  # the page cache is memory this machine does not have
        return self.fds[path]

    def _read(self, t: Tensor, first: int, count: int) -> mx.array:
        one = t.data.nbytes // t.shape[0]
        fd = self._fd(t.path)
        data = os.pread(fd, count * one, t.offset + first * one)
        if len(data) != count * one:
            raise OSError(f"{t.name}: short read")
        return mx.array(np.frombuffer(data, dtype=np.uint8))

    def whole(self, tensors: tuple) -> list:
        """A layer's tensors, all experts, read in file order and not kept.

        In pieces of 1 MiB that start on a page: measured on macOS, a read of 2 MB or more goes through the page
        cache whatever F_NOCACHE says (240 MB read in 4 MB pieces left 189 MB there), one of 1 MiB does not.
        """
        t0 = time.perf_counter()
        out = []
        for t in tensors:
            fd, size = self._fd(t.path), t.data.nbytes
            start = t.offset // PAGE * PAGE
            buf = bytearray(t.offset + size - start)
            view = memoryview(buf)
            got = self.pool.map(lambda at: os.preadv(fd, [view[at : at + PIECE]], start + at), range(0, len(buf), PIECE))
            if sum(got) != len(buf):
                raise OSError(f"{t.name}: short read")
            out.append(mx.array(np.frombuffer(view[t.offset - start :], dtype=np.uint8)))
            self.bytes_read += size
        self.read_s += time.perf_counter() - t0
        return out

    def clear(self):
        """Let go of every expert and of the counts of their use."""
        self.lru.clear()
        self.use.clear()
        self.held = 0

    def preload(self, counts: np.ndarray, group: int = 512) -> int:
        """Fill what is free of the budget with the experts `counts` (layers, experts) says are used most.
        Returns how many were read. The most used are the last the store will drop."""
        if not self.layers or counts.ndim != 2 or counts.shape[0] <= max(self.layers) \
                or any(t[0].shape[0] != counts.shape[1] for t in self.layers.values()):
            return 0
        size = {layer: sum(t.data.nbytes // t.shape[0] for t in tensors) for layer, tensors in self.layers.items()}
        order = sorted(((int(counts[layer, e]), layer, e) for layer in self.layers for e in range(counts.shape[1])
                        if counts[layer, e] > 0 and (layer, e) not in self.lru), reverse=True)
        chosen, room = [], self.budget - self.held
        for _, layer, e in order:
            if size[layer] > room:
                break
            chosen.append((layer, e))
            room -= size[layer]
        chosen.reverse()  # the least used first: the first to go
        for a in range(0, len(chosen), group):
            part = chosen[a : a + group]
            jobs = [(e, t) for layer, e in part for t in self.layers[layer]]
            arrays = list(self.pool.map(lambda job: self._read(job[1], job[0], 1), jobs))
            for i, (layer, e) in enumerate(part):
                self.lru[(layer, e)] = (tuple(arrays[i * 3 : i * 3 + 3]), size[layer])
                self.held += size[layer]
        return len(chosen)

    def profile(self, before: np.ndarray | None = None) -> np.ndarray | None:
        """How often each expert was used, (layers, experts): this run's count, and half of what was counted
        before it, so that what the machine is used for now weighs more than what it was used for once."""
        if not self.layers:
            return None
        shape = (max(self.layers) + 1, next(iter(self.layers.values()))[0].shape[0])
        out = np.zeros(shape, dtype=np.uint32)
        if before is not None and before.shape == shape:
            out += before.astype(np.uint32) // 2
        for (layer, e), n in self.use.items():
            out[layer, e] += n
        return out

    def get(self, layer: int, tensors: tuple, experts: list, keep: bool = True) -> list:
        """The arrays of each of `experts`, read if they are not in memory."""
        if keep:  # an answer's tokens, not a prompt's
            for e in experts:
                self.use[(layer, e)] = self.use.get((layer, e), 0) + 1
        missing = [e for e in experts if (layer, e) not in self.lru]
        self.hits += len(experts) - len(missing)
        self.misses += len(missing)
        got = {}
        if missing:
            t0 = time.perf_counter()
            jobs = [(e, t) for e in missing for t in tensors]
            arrays = list(self.pool.map(lambda job: self._read(job[1], job[0], 1), jobs))
            self.read_s += time.perf_counter() - t0
            n = len(tensors)
            for i, e in enumerate(missing):
                got[e] = tuple(arrays[i * n : (i + 1) * n])
                size = sum(a.nbytes for a in got[e])
                self.bytes_read += size
                if keep:
                    self.lru[(layer, e)] = (got[e], size)
                    self.held += size
        out = []
        for e in experts:
            if e in got:
                out.append(got[e])
            else:
                self.lru.move_to_end((layer, e))
                out.append(self.lru[(layer, e)][0])
        # what this call uses is at the recent end; drop from the other one
        while self.held > self.budget and len(self.lru) > len(experts):
            _, (_, size) = self.lru.popitem(last=False)
            self.held -= size
        return out


class PagedSwitchGLU:
    """A layer's routed experts, called like mlx-lm's `SwitchGLU`, their weights in an `ExpertStore`."""

    def __init__(self, layer: int, store: ExpertStore, gate: Tensor, up: Tensor, down: Tensor):
        for t in (gate, up, down):
            if t.type not in BLOCK:
                raise ValueError(f"{t.name}: {t.type} experts are not supported")
        self.layer, self.store, self.tensors = layer, store, (up, gate, down)
        store.layers[layer] = self.tensors

    def _linear(self, blocks: mx.array, t: Tensor, experts: int) -> BlockSwitchLinear:
        lin = BlockSwitchLinear(blocks, t.type, (experts, *t.shape[1:]))
        lin.unpack_from = 1 << 62  # never unpacked to floats: 1.7 GB for a moment is more than there may be
        return lin

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        idx = np.array(indices)  # the layer waits here for its router
        used, local = np.unique(idx, return_inverse=True)
        if idx.size <= SEPARATE_UPTO:
            return self._separately(x, idx, used)
        if len(used) >= WHOLE_FROM:
            stacks, n = self.store.whole(self.tensors), self.tensors[0].shape[0]
        else:
            each = self.store.get(self.layer, self.tensors, used.tolist(), keep=idx.size <= KEEP_UPTO)
            stacks, n = [mx.concatenate(col) if len(col) > 1 else col[0] for col in zip(*each)], len(used)
            indices = mx.array(local.reshape(idx.shape).astype(np.uint32))
        up, gate, down = (self._linear(b, t, n) for b, t in zip(stacks, self.tensors))

        return switch_glu(x, indices, up, gate, down)

    def _separately(self, x: mx.array, idx: np.ndarray, used: np.ndarray) -> mx.array:
        """A few tokens: no stack is built, the kernels read each pair's expert where it lies."""
        k = idx.shape[-1]
        by = dict(zip(used.tolist(), self.store.get(self.layer, self.tensors, used.tolist())))
        pairs = [by[e] for e in idx.reshape(-1).tolist()]
        (_, hidden, wide), kinds = self.tensors[0].shape, [t.type for t in self.tensors]
        x = x.reshape(-1, wide)
        tokens = max(1, MAX_SEPARATE // k)  # as many as one kernel can bind
        out = []
        for t in range(0, len(x), tokens):
            ups, gates, downs = zip(*pairs[t * k : (t + tokens) * k])
            xs = x[t : t + tokens]
            h = swiglu(matvec_separate(gates, kinds[1], hidden, wide, xs, k),
                       matvec_separate(ups, kinds[0], hidden, wide, xs, k))
            out.append(matvec_separate(downs, kinds[2], wide, hidden, h, 1))
        out = mx.concatenate(out) if len(out) > 1 else out[0]
        return out.reshape(*idx.shape, wide)


SEED = os.path.join(os.path.dirname(__file__), "expert_use.npy")  # bench/expert_use.py --counts, with Q2_0


def load_profile(path: str | None) -> np.ndarray | None:
    """The counts in `path`, or the ones that come with the engine if there is no such file or it cannot be read."""
    for p in (path, SEED):
        try:
            if p and os.path.exists(p):
                return np.load(p)
        except (OSError, ValueError):
            pass
    return None


def save_profile(path: str, counts: np.ndarray) -> None:
    """Beside its place, then renamed: a start never finds half a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        np.save(f, counts)
    os.replace(tmp, path)
