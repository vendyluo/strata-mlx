"""A prompt read a layer at a time (bylayer.py) leaves what the engine's read leaves, bit for bit; reads a
layer's experts once; and, stopped inside a piece, is as if the piece had not been begun."""

import mlx.core as mx
import pytest

from strata_mlx import bylayer, engine, paged, serve
from strata_mlx.engine import Engine
from strata_mlx.mtp import Mtp
from test_engine import ONE_EXPERT, TEXT, TURN, _paged

BODY = [(7 * i + 3) % TEXT["vocab_size"] for i in range(150)]
PROMPT = [TURN, *BODY[:30], TURN, *BODY[30:110], TURN, *BODY[110:]]
PROMPT = [p if (p != TURN or j in (0, 31, 112)) else 11 for j, p in enumerate(PROMPT)]
FOLLOW = [TURN, 60, 61, 62, TURN, 50]
LAYERS = TEXT["num_hidden_layers"]


class Stopped(Exception):
    pass


@pytest.fixture(scope="module", params=["float32", "float16"])
def made(request, tmp_path_factory):
    """(the small model with its experts in a file, a draft head) of a dtype: every path the real one takes."""
    before = paged.WHOLE_FROM, paged.KEEP_UPTO, paged.SEPARATE_UPTO
    # eight experts a layer: a chunk that uses them all reads the layer whole, one that does not reads those it
    # uses and keeps none, and only the prompt's last token, a chunk by itself, goes the way an answer's do
    paged.WHOLE_FROM, paged.KEEP_UPTO, paged.SEPARATE_UPTO = 8, 4, 4
    model = _paged(tmp_path_factory.mktemp("experts"), 1 << 30)
    mx.random.seed(4)
    mtp = Mtp(model.args.text)
    if request.param != "float32":
        model.set_dtype(getattr(mx, request.param))
        mtp.set_dtype(getattr(mx, request.param))
    mx.eval(model.parameters(), mtp.parameters())
    yield model, mtp
    paged.WHOLE_FROM, paged.KEEP_UPTO, paged.SEPARATE_UPTO = before


def make(made, chunk=16, **kw):
    model, mtp = made
    model.expert_store.clear()
    model.expert_store.reset_stats()
    return Engine(model, mtp, drafts=3, lookup=True, chunk=chunk, turn_token=TURN, root=20, **kw)


def state_of(e: Engine) -> dict:
    """Every array of the conversation an engine holds, by name."""
    out = {"context": e.context, "ple_conv": e.ple_conv}
    for i, (conv, recurrent) in e.gdn.items():
        out[f"gdn.{i}.conv"], out[f"gdn.{i}.state"] = conv, recurrent
    caches = {f"attn.{i}": c for i, c in e.attn.items()}
    caches["draft head"] = e.mtp_cache
    for name, c in caches.items():
        assert c.offset == c.indexer.offset == len(e.tokens) and c.indexer.pooled_tokens <= c.offset, name
        if c.keys is not None:
            out[f"{name}.keys"], out[f"{name}.values"], out[f"{name}.index"] = c.state
    for j, cp in enumerate(e.checkpoints):
        out[f"checkpoint.{j}.marks"] = mx.array([cp.at, cp.pinned, cp.tail])
        out[f"checkpoint.{j}.context"], out[f"checkpoint.{j}.ple_conv"] = cp.context, cp.ple_conv
        for i, (conv, recurrent) in cp.gdn.items():
            out[f"checkpoint.{j}.gdn.{i}.conv"], out[f"checkpoint.{j}.gdn.{i}.state"] = conv, recurrent
    return {k: v for k, v in out.items() if v is not None}


def differ(a: Engine, b: Engine) -> list:
    """The names of what two engines hold differently: no array may differ in a bit."""
    sa, sb = state_of(a), state_of(b)
    out = sorted(sa.keys() ^ sb.keys())
    out += [k for k in sorted(sa.keys() & sb.keys()) if sa[k].shape != sb[k].shape or not mx.array_equal(sa[k], sb[k]).item()]
    if a.tokens != b.tokens or a.lookup.seen != b.lookup.seen:
        out.append("tokens")
    return out


def answer(e: Engine, first: int, n: int = 16) -> list:
    out = [first]
    while len(out) < n:
        out += e.step(n - len(out))
    return out


def both(made, prompt=PROMPT, chunk=16, piece=bylayer.PIECE, ahead=True, **read):
    """(an engine that read the prompt chunk by chunk, one that read it a layer at a time, their first tokens)."""
    a, b = make(made, chunk, piece=0), make(made, chunk, piece=piece, ahead=ahead)
    return a, b, a.read(prompt, **read), b.read(prompt, **read)


@pytest.mark.parametrize("how", [{}, {"keep": False}, {"pin": 45}, {"piece": 48}, {"piece": 32, "pin": 45},
                                 {"ahead": False}, {"piece": 48, "ahead": False}, {"chunk": 7}, {"chunk": 7, "piece": 21},
                                 {"chunk": 3, "ahead": False}, {"chunk": 3, "piece": 30, "ahead": False}])
def test_a_layer_at_a_time_leaves_what_chunk_by_chunk_leaves(made, how):
    a, b, ta, tb = both(made, **how)
    assert a.checkpoints and len(a.tokens) == len(PROMPT)
    assert differ(a, b) == []
    assert ta == tb and a.guesses == b.guesses
    out = answer(a, ta)
    assert answer(b, tb) == out
    nxt = PROMPT + out + FOLLOW  # and a request that goes on from the answer
    assert a.resume(nxt) == b.resume(nxt)
    assert answer(a, a.read(nxt)) == answer(b, b.read(nxt))
    assert differ(a, b) == []


def test_it_is_the_read_that_was_tested_and_only_for_more_than_a_chunk(made, monkeypatch):
    seen = []
    inner = bylayer.read
    monkeypatch.setattr(bylayer, "read", lambda *a, **kw: seen.append(1) or inner(*a, **kw))
    make(made).read(PROMPT)
    assert seen == [1]
    make(made, piece=0).read(PROMPT)
    make(made, chunk=len(PROMPT)).read(PROMPT)  # a chunk: as before
    e = make(made)
    e.read(PROMPT[:140])
    e.read(PROMPT[:156])  # 16 tokens more, a chunk of them
    assert seen == [1, 1]


def test_a_layers_experts_are_read_once_for_a_piece(made):
    store = made[0].expert_store
    prompt = PROMPT[:113]  # one message's end inside, and the token before the last: chunks of 16, 15, 1
    make(made, piece=0).read(prompt)
    before = store.bytes_read
    assert before > LAYERS * 3 * 8 * ONE_EXPERT
    # read ahead, a layer is read whole and once. Two chunks are a token each (the one before a message's
    # end, the prompt's last) and go the way an answer's tokens do: their two experts each are read and stay
    make(made).read(prompt)
    assert LAYERS * (8 + 2) * ONE_EXPERT <= store.bytes_read <= LAYERS * (8 + 4) * ONE_EXPERT
    assert store.bytes_read - store.held == LAYERS * 8 * ONE_EXPERT
    assert store.kept is None and store.coming is None
    # not read ahead, a layer is read whole when a chunk uses that much of it and before that expert by expert:
    # at most twice
    make(made, ahead=False).read(prompt)
    assert LAYERS * 8 * ONE_EXPERT <= store.bytes_read - store.held <= LAYERS * (7 + 8) * ONE_EXPERT
    make(made, ahead=False, chunk=3).read(prompt)  # chunks that use few experts: each read once, or the layer
    assert store.bytes_read - store.held <= LAYERS * (7 + 8) * ONE_EXPERT
    # three pieces: 48, 48 and the rest. Each reads a layer once, but not what memory holds of it: the two experts
    # a layer the token before the message's end left there are not read again by the two pieces after it
    make(made, piece=48).read(prompt)
    assert store.bytes_read - store.held == LAYERS * (3 * 8 - 2 * 2) * ONE_EXPERT


@pytest.mark.parametrize("small", [False, True])
@pytest.mark.parametrize("how", [{}, {"piece": 48}, {"ahead": False}, {"piece": 0}])
def test_what_memory_holds_of_a_layer_is_not_read_again(made, how, small, monkeypatch):
    import numpy as np

    if small:  # pages and pieces smaller than this model's runs of experts: a run read in several pieces
        monkeypatch.setattr(paged, "PAGE", 256)
        monkeypatch.setattr(paged, "PIECE", 512)
    store, prompt = made[0].expert_store, PROMPT[:113]
    a = make(made, **how)
    ta = a.read(prompt)
    read = store.bytes_read
    b = make(made, **how)
    counts = np.zeros((LAYERS, 8), dtype=np.uint32)
    counts[:, [0, 3, 4, 7]] = 1  # half of each layer in memory, in two runs: what is read lies between them
    assert store.preload(counts) == LAYERS * 4
    store.reset_stats()
    tb = b.read(prompt)
    # a layer whole is half read, or less: a token's chunk may find its experts in memory too
    assert store.bytes_read <= read - LAYERS * 4 * ONE_EXPERT
    assert differ(a, b) == [] and ta == tb
    assert answer(b, tb) == answer(a, ta)


def in_file(tensors: tuple) -> list:
    import numpy as np

    out = []
    for t in tensors:
        with open(t.path, "rb") as f:
            f.seek(t.offset)
            out.append(np.frombuffer(f.read(t.data.nbytes), dtype=np.uint8))
    return out


def test_experts_let_go_of_while_their_layer_is_read_ahead_are_still_in_it(made):
    import numpy as np

    store = made[0].expert_store
    make(made)
    layer = min(store.layers)
    counts = np.zeros((LAYERS, 8), dtype=np.uint32)
    counts[layer, [0, 3, 4, 7]] = 1
    assert store.preload(counts) == 4
    store.reset_stats()
    store.ahead(layer)
    store.lru.clear()  # a token's chunk lets them go before the read is taken
    store.held = 0
    store.hold(layer)
    for got, want in zip(store.whole(store.layers[layer], layer), in_file(store.layers[layer])):
        np.testing.assert_array_equal(np.array(got), want)
    assert store.bytes_read == 4 * ONE_EXPERT
    store.release()


def test_a_short_read_ahead_fails_at_hold_once_every_read_of_it_has_ended(made, monkeypatch):
    import itertools
    import os
    import time

    import numpy as np

    store = made[0].expert_store
    make(made)
    first, then = sorted(store.layers)[:2]
    real, calls, started, ended = os.preadv, itertools.count(), [], []

    def preadv(fd, buffers, at):
        k = next(calls)
        started.append(k)
        if k:
            time.sleep(0.05)
        n = real(fd, buffers, at)
        ended.append(k)
        return n - 1 if k == 0 else n

    monkeypatch.setattr(paged.os, "preadv", preadv)
    store.ahead(first)
    with pytest.raises(OSError, match="short read"):
        store.hold(first)
    assert len(started) >= len(store.layers[first]) and len(ended) == len(started)
    monkeypatch.setattr(paged.os, "preadv", real)
    store.ahead(then)  # and the next layer is read as if nothing had failed
    store.hold(then)
    for got, want in zip(store.whole(store.layers[then], then), in_file(store.layers[then])):
        np.testing.assert_array_equal(np.array(got), want)
    store.release()


def test_pieces():
    def of(bounds, piece, chunk=4, periodic=1 << 30):
        return [(p[0][0], p[-1][1]) for p in bylayer.pieces(list(zip(bounds, bounds[1:])), piece, chunk, periodic)]

    assert of([0, 4, 8, 12, 16], 8) == [(0, 8), (8, 16)]
    assert of([0, 4, 8, 12, 16], 100) == [(0, 16)]
    assert of([0, 4, 8], 2) == [(0, 4), (4, 8)]  # at least a chunk
    assert of([0, 4, 7, 8, 9], 8) == [(0, 9)]  # a last piece of less than a chunk goes with the one before
    assert of([0, 4, 8, 12, 15, 16], 8) == [(0, 8), (8, 16)]
    assert of([3, 4, 8, 12, 16, 20], 16, periodic=8) == [(3, 8), (8, 16), (16, 20)]  # ends at a periodic checkpoint


def steps_of(made, prompt, e=None, **kw) -> int:
    count = [0]

    def step():
        count[0] += 1

    e = e or make(made, **kw)
    e.read(prompt, on_step=step)
    return count[0]


def stop_at(at: int):
    count = [0]

    def step():
        count[0] += 1
        if count[0] == at:
            raise Stopped

    return step


def held(made, **kw):
    """Two engines that hold the same conversation and are asked for the same longer prompt."""
    first = PROMPT[:60]
    a, b = make(made, piece=0), make(made, **kw)
    out = answer(a, a.read(first), 6)
    assert answer(b, b.read(first), 6) == out
    nxt = first + out[:-1] + [TURN, 60, 61] + PROMPT[60:]
    assert a.resume(nxt) == b.resume(nxt) == len(a.tokens)
    return a, b, nxt


def test_a_piece_that_is_stopped_is_as_if_not_begun(made):
    a, x, nxt = held(made)
    total = steps_of(made, nxt, x)
    assert total > 2 * LAYERS * 5
    out = None
    for at in sorted({*range(1, total + 1, 7), 2, total - 1, total}):  # first layer and last, either half, the cells
        a, x, nxt = held(made)
        before = len(x.tokens)
        with pytest.raises(Stopped):
            x.read(nxt, on_step=stop_at(at))
        assert len(x.tokens) == before and differ(a, x) == [], at
        store = made[0].expert_store
        assert store.kept is None and store.coming is None
        ta, tb = a.read(nxt), x.read(nxt)  # read again: as if never stopped
        assert differ(a, x) == [] and ta == tb, at
        if out is None:
            out = answer(a, ta)
            assert answer(x, tb) == out


def test_stopped_from_nothing_it_holds_nothing(made):
    blank = make(made)
    for at in (1, 9, 40):
        x = make(made)
        with pytest.raises(Stopped):
            x.read(PROMPT, on_step=stop_at(at))
        assert differ(blank, x) == [] and x.tokens == []
        assert all(c.keys is None for c in [*x.attn.values(), x.mtp_cache])  # nothing for a session file to hold
    a, b, ta, tb = both(made)
    x.read(PROMPT)
    assert differ(a, x) == []


def test_stopped_by_what_hears_the_progress(made, monkeypatch):
    monkeypatch.setattr(bylayer, "SAID_UPTO", 0)
    a, x, nxt = held(made)
    said = []

    def progress(done, total):
        said.append(done)
        if len(said) == 3:
            raise Stopped

    with pytest.raises(Stopped):
        x.read(nxt, on_chunk=progress)
    assert said == sorted(said) and len(a.tokens) < said[0] and said[-1] < len(nxt)  # in proportion to the layers done
    assert differ(a, x) == []


def test_stopped_where_attention_goes_by_pooled_keys(made):
    model, mtp = made
    indexers = [l.self_attn.indexer for l in model.model.layers if l.layer_type != "linear_attention"]
    budgets = [i.token_budget for i in indexers]
    try:
        for i in indexers:
            i.token_budget = 16
        a, b, ta, tb = both(made)
        assert differ(a, b) == [] and ta == tb
        for at in (3, 22, 51, 70):
            a, x, nxt = held(made)
            with pytest.raises(Stopped):
                x.read(nxt, on_step=stop_at(at))
            assert differ(a, x) == []
            assert a.read(nxt) == x.read(nxt) and differ(a, x) == []
    finally:
        for i, budget in zip(indexers, budgets):
            i.token_budget = budget


def test_stopped_in_a_later_piece_the_pieces_before_it_stay(made):
    x = make(made, piece=48)
    one = steps_of(made, PROMPT[:48], piece=48)
    with pytest.raises(Stopped):
        x.read(PROMPT, on_step=stop_at(one + 10))
    assert len(x.tokens) == 48
    a = make(made, piece=0)
    a.read(PROMPT[:49])  # what a request for that much and a token leaves, the token not decoded
    assert x.tokens == a.tokens[:48]
    assert x.resume(PROMPT) == 48
    ref = make(made, piece=0)
    assert answer(x, x.read(PROMPT)) == answer(ref, ref.read(PROMPT))


# ---- through the line protocol

def session_of(made, monkeypatch, **kw):
    monkeypatch.setattr(serve, "END_TOKENS", (5,))
    lines = []
    s = serve.Session(make(made, **kw), lines.append, 4096)
    s.lines = lines
    return s


def ids(tokens):
    return ",".join(map(str, tokens))


def test_progress_is_said_layer_by_layer_and_ends_at_the_prompts_length(made, monkeypatch):
    monkeypatch.setattr(bylayer, "SAID_UPTO", 4)
    s = session_of(made, monkeypatch, piece=96)
    s.handle(f"GEN 4 {ids(PROMPT)}")
    said = [int(l.split()[1]) for l in s.lines if l.startswith("PP ")]
    assert len(said) == 2 * LAYERS + 2 and said == sorted(said)  # two pieces: a line a layer, and one at each end
    assert said[LAYERS] == 96 and said[-1] == len(PROMPT)
    assert all(len(PROMPT) - t > 4 for t in said[:-1])
    assert s.lines[-1].split()[14] == str(len(PROMPT))


def test_stop_inside_a_piece_says_what_stays_read(made, monkeypatch):
    monkeypatch.setattr(bylayer, "SAID_UPTO", 4)
    s = session_of(made, monkeypatch, piece=96)
    seen = []

    def write(line):
        seen.append(line)
        if len([l for l in seen if l.startswith("PP ")]) == LAYERS + 3:  # in the second piece
            s.stop.set()

    s.write = write
    s.handle(f"GEN 200 {ids(PROMPT)}")
    done = seen[-1].split()
    assert done[0] == "DONE" and done[5] == "cancel" and done[1] == "0"
    assert int(done[14]) == len(s.engine.tokens) == 96  # the first piece, not what the line of progress said
    s.stop.clear()
    del seen[:]
    s.handle(f"GEN 4 {ids(PROMPT)}")
    assert seen[0] == "RESUME 96" and int(seen[-1].split()[14]) == len(PROMPT) - 96
    ref = make(made, piece=0)
    assert [int(l.split()[1]) for l in seen if l.startswith("T ")] == answer(ref, ref.read(PROMPT), 4)
