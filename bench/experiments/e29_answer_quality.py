"""Experiment 29: is an answer written with drafts, in float16, any worse than the model's own?

In float16 a window of several tokens and a single token round differently, so where two tokens are nearly
level the faster ways of decoding can write the other one, and from there on another text. Whether that costs
anything is asked of the model itself, with nothing rounded and nothing guessed: float32, a token a pass. It
reads each answer and says, at every token, whether that token was its own first choice after the same text,
and if not how much less probable than its first choice it found it.

The answers: float16 a token a pass; with the draft head; with the head's guesses over the listed rows
(rows.py), which is what the server runs. And float32 with the draft head and the list, which the tests hold to
be the float32 answer exactly.

Result, tokens that were the judge's first choice, of four answers (and the widest gap elsewhere: its first
choice's probability, the written token's):

                                        Q2_0, 1,024 tokens              IQ3_S, 986 tokens
    float16, a token a pass             1,020  (49.5%, 46.7%)           982  (49.5%, 47.7%)
    float16, draft head                 1,018  (26.1%, 25.1%)           981  (47.4%, 44.3%)
    float16, draft head, listed rows    1,020  (36.2%, 34.5%)           981  (44.6%, 38.0%)
    float32, draft head, listed rows    1,024                           986

Float16 writes a token other than the model's first choice about once in 250, with or without drafts and
with or without the list, always where two are nearly level; the text after it is then another one. Float32
with drafts and the list wrote the model's own answer every time, at 44-49 tok/s with the Q2_0 file (80-103 in
float16: its table lookup is float16 only) and 53-73 with IQ3_S (69-80).

    uv run python bench/experiments/e29_answer_quality.py --model <shard 1> --mtp ~/.cache/strata-mlx/mtp
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# #5: the judge is float32, and on an M5 MLX multiplies float32 at reduced precision unless told not to first
os.environ.setdefault("MLX_ENABLE_TF32", "0")
import mlx.core as mx  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from spec_decode import PROMPTS, decode, tokenize  # noqa: E402

from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402
from strata_mlx.mtp import load_mtp  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--mtp", required=True)
p.add_argument("--tokens", type=int, default=256)
p.add_argument("--json")
args = p.parse_args()
prompts = {name: tokenize(args.model, f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
           for name, text in PROMPTS.items()}

answers = {name: {} for name in prompts}  # prompt -> way of decoding -> (tokens, tok/s)
model, _ = load(args.model)
mtp = load_mtp(args.mtp, model.args.text, mx.float16)
ways = {"float16, a token a pass": Engine(model), "float16, draft head": Engine(model, mtp, guess_rows=False),
        "float16, draft head, listed rows": Engine(model, mtp)}
for engine in ways.values():
    engine.warm()
for _ in range(2):  # the second round: the list as a server has it after a while
    for name, prompt in prompts.items():
        for way, engine in ways.items():
            out, _, tps = decode(engine, prompt, args.tokens, {})
            answers[name][way] = (out, tps)
del ways, model, mtp
mx.clear_cache()

model, _ = load(args.model, mx.float32)
mtp = load_mtp(args.mtp, model.args.text, mx.float32)
fast = Engine(model, mtp)
fast.warm()
for name, prompt in prompts.items():
    answers[name]["float32, draft head, listed rows"] = decode(fast, prompt, args.tokens, {})[::2]
for name, prompt in prompts.items():  # again, the list longer now and the windows other ones: the same answer?
    again = decode(fast, prompt, args.tokens, {})[0]
    if again != answers[name]["float32, draft head, listed rows"][0]:
        print(f"{name}: float32 with drafts wrote another answer the second time", flush=True)
judge = Engine(model)
judge.warm()
rows = []
for name, prompt in prompts.items():
    for way, (out, tps) in answers[name].items():
        judge.reset()
        judge.read(prompt[:-1])
        first, behind = 0, []
        t0 = time.perf_counter()
        for i, (given, written) in enumerate(zip(prompt[-1:] + out[:-1], out)):
            logits, w = judge._forward(mx.array([[given]]))
            logp = logits[0, -1].astype(mx.float32)
            logp = logp - mx.logsumexp(logp)
            top = mx.argmax(logp)
            mx.eval(logp, top)
            judge._commit(w, 1)
            gap = logp[top].item() - logp[written].item()
            first += top.item() == written
            if top.item() != written:
                behind.append((gap, i, math.exp(logp[top].item()), math.exp(logp[written].item())))
        worst = max(behind, default=(0.0, -1, 0.0, 0.0))
        ref = answers[name]["float16, a token a pass"][0]
        parts = next((i for i, (a, b) in enumerate(zip(out, ref)) if a != b), len(out))
        rows.append({"prompt": name, "way": way, "tokens": len(out), "tok_s": round(tps, 1), "first_choice": first,
                     "not_first": [{"at": i, "behind": round(g, 4), "p_first": round(a, 4), "p_written": round(b, 4)} for g, i, a, b in behind],
                     "parts_from_float16_single_at": parts})
        print(f"{name:10} {way:34} {tps:5.1f} tok/s  the judge's first choice at {first:3}/{len(out)}"
              + (f"; elsewhere its first choice had {worst[2]:.1%} and the token written {worst[3]:.1%} at the most apart (token {worst[1]})"
                 if behind else "")
              + ("" if parts == len(out) else f" | parts from float16 a token a pass at {parts}"), flush=True)
if args.json:
    Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")
