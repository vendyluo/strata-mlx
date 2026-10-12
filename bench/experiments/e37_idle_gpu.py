"""Experiment 37: does the GPU itself slow down after a rest? (ISSUES.md, issue 1, its next step)

e34 found a piece of 64 tokens two to three times as slow after two seconds of nothing, a piece of 256 tokens a
little slower, one token of an answer not at all. Here no model: the same work one time after another, then
after rests of growing length.

- `cube`: one 2048-cube float16 product (compute, a few ms).
- `chain`: 400 small products one after another in one graph (many short kernels, like a layer's).
- `tiny`: one addition of two numbers (what a dispatch and a wait cost, nothing more).

    uv run python bench/experiments/e37_idle_gpu.py [--rests 0.5 2 4 8] [--repeat 3]
"""

import argparse
import time

import mlx.core as mx

p = argparse.ArgumentParser()
p.add_argument("--rests", nargs="+", type=float, default=[0.5, 2, 4, 8])
p.add_argument("--repeat", type=int, default=3)
p.add_argument("--beat", type=float, default=0, help="during a rest, one `tiny` every this many seconds (0: none)")
p.add_argument("--only", nargs="+", default=["cube", "chain", "tiny"])
args = p.parse_args()

a = mx.random.normal((2048, 2048)).astype(mx.float16)
v = mx.random.normal((64, 2048)).astype(mx.float16)
w = mx.random.normal((2048, 2048)).astype(mx.float16) / 64
one = mx.array(1.0)
mx.eval(a, v, w, one)


def cube():
    return a @ a


def chain():
    x = v
    for _ in range(400):
        x = x @ w
    return x


def tiny():
    return one + one


def timed(fn) -> float:
    t0 = time.perf_counter()
    mx.eval(fn())
    return (time.perf_counter() - t0) * 1e3


def rest(seconds: float):
    """Nothing for `seconds`, or with --beat, nothing but a `tiny` now and then."""
    if not args.beat:
        return time.sleep(seconds)
    end = time.perf_counter() + seconds
    while (left := end - time.perf_counter()) > 0:
        time.sleep(min(args.beat, left))
        mx.eval(tiny())


for name, fn in (("cube", cube), ("chain", chain), ("tiny", tiny)):
    if name not in args.only:
        continue
    for _ in range(5):
        timed(fn)
    for r in range(args.repeat):
        row = f"{name:5} one after another " + " ".join(f"{timed(fn):7.2f}" for _ in range(5))
        for rest_s in args.rests:
            rest(r_s := rest_s)
            row += f" | after {r_s:g} s {timed(fn):7.2f}"
        print(row + " ms", flush=True)
