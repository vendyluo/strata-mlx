"""Experiment 38: where a short piece of a prompt loses its time after a rest (ISSUES.md, issue 1).

e34 found a piece of 64 tokens two to three times as slow after two seconds of nothing; e37 that the GPU and
the CPU both run slower for a while after a rest, without a model. Here a piece's time is split in two: the
CPU building its graph (`_forward` returns arrays not yet computed) and the GPU computing it (`mx.eval`). With
`--beat S --work N` the GPU is given N small products one after another every S seconds of the rest, to
see how much work keeps it at speed, which a server waiting for its next request would have to do; with
`--stream GB` it sums that much memory instead. With `--spin` the CPU is kept busy and the GPU idle. With
`--wired` the model's arrays are kept resident. With `--first`, [ms] of one addition on the GPU before each
piece, by itself.

Result, M5 Ultra 96 GB, Q2_0, a piece of 64 tokens (building + computing, ms):

    one after another                               4 +  92
    after 2-8 s                                    15-24 + 223-379
    after 2-8 s, the GPU given work every 0.05-0.5 s (8 MB of products or 1-2 GB summed, 2-40% of the time)
                                                   12-20 + 221-336     no better
    after 0.5-1 s, the CPU kept busy                5 + 101-115
    after 2-8 s, the CPU kept busy                  8 + 223-306
    after 2-8 s, wired                             16-21 + 150-163     but after 4-8 s the first thing on the
                                                                       GPU waits 250-320 ms ([252], [317])
    after 2-8 s, wired, one addition every 1-2.5 s  5-20 + 103-161

So not the GPU's clock: work during the rest did not help, whatever it was. Two things instead. The arrays MLX
does not keep resident are let go of by the GPU after about two seconds of nothing, and every piece after that
computes at less than half speed; resident (`mx.set_wired_limit`, as mlx-lm does), they are not. And after
about four seconds of nothing the GPU sleeps, and the first thing given to it waits 250-320 ms; one addition a
second keeps it awake. What is left, 60-80 ms after a rest of half a second or more, is the CPU's: kept busy,
it goes. serve.py does the first two (#1).

    uv run python bench/experiments/e38_idle_parts.py --model <shard 1> [--lengths 64 256] [--rests 0.5 2 4 8]
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from spec_decode import EDIT_ME, tokenize  # noqa: E402
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--lengths", nargs="+", type=int, default=[64, 256])
p.add_argument("--rests", nargs="+", type=float, default=[0.5, 2, 4, 8])
p.add_argument("--repeat", type=int, default=2)
p.add_argument("--beat", type=float, default=0, help="during a rest, some work every this many seconds (0: none)")
p.add_argument("--work", type=int, default=50, help="small products one after another, each time")
p.add_argument("--stream", type=float, default=0, help="or: GB of memory summed each time (memory, not arithmetic)")
p.add_argument("--spin", action="store_true", help="during a rest, keep this thread busy on the CPU, the GPU idle")
p.add_argument("--first", action="store_true", help="before each piece, time one addition on the GPU by itself")
p.add_argument("--wired", action="store_true", help="MLX keeps its arrays resident (mx.set_wired_limit), as mlx-lm does")
args = p.parse_args()

if args.wired:
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

model, _ = load(args.model)
prompt = tokenize(args.model, "<|im_start|>user\nSay what this is for.\n\n```python\n" + EDIT_ME * 9 + "```<|im_end|>\n<|im_start|>assistant\n")
engine = Engine(model)
engine.warm()


p_first = []  # the first thing on the GPU after a rest, with --first: ms


def read(T: int) -> tuple:
    """One piece of T tokens into an empty conversation: (building, computing, all) in ms."""
    if args.first:
        t = time.perf_counter()
        mx.eval(mx.array(1.0) + 1)
        p_first.append((time.perf_counter() - t) * 1e3)
    engine.reset()
    t0 = time.perf_counter()
    logits, w = engine._forward(mx.array([prompt[:T]]), last_only=True)
    t1 = time.perf_counter()
    mx.eval(logits)
    engine._commit(w, T)
    mx.eval(*[a for v in engine.gdn.values() for a in v])
    t2 = time.perf_counter()
    return (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t2 - t0) * 1e3


_v = mx.random.normal((64, 2048)).astype(mx.float16)
_w = mx.random.normal((2048, 2048)).astype(mx.float16) / 64
_big = mx.ones((int(args.stream * 2**29),), dtype=mx.float16) if args.stream else None
mx.eval(_v, _w, *([_big] if args.stream else []))


def work():
    if args.stream:
        return mx.eval(_big.sum())
    x = _v
    for _ in range(args.work):
        x = x @ _w
    mx.eval(x)


def rest(seconds: float):
    """Nothing for `seconds`, or with --beat, nothing but some work now and then."""
    end = time.perf_counter() + seconds
    if args.spin:
        while time.perf_counter() < end:
            pass
        return
    if not args.beat:
        return time.sleep(seconds)
    while (left := end - time.perf_counter()) > 0:
        time.sleep(min(args.beat, left))
        work()


def show(parts: tuple) -> str:
    return ("[{:3.0f}] ".format(p_first[-1]) if args.first else "") + "{:4.0f}+{:4.0f}={:4.0f}".format(*parts)


t0 = time.perf_counter()
work()
print(f"ms: building the graph + computing it = the piece; the work of a beat takes {(time.perf_counter() - t0) * 1e3:.1f} ms"
      + (f", every {args.beat:g} s" if args.beat else ""), flush=True)
for T in args.lengths:
    read(T), read(T)
    for _ in range(args.repeat):
        row = f"{T:4} tokens: one after another " + " ".join(show(read(T)) for _ in range(3))
        for seconds in args.rests:
            rest(seconds)
            row += f" | after {seconds:g} s {show(read(T))}"
        print(row, flush=True)
