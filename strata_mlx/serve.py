"""The engine as a process that speaks Strata's line protocol, so `serve/server.py` can drive it unchanged.

The server starts `<exe> --serve <args>` and talks over stdin and stdout. Prompts arrive as token ids.

    engine  INFO engine=... backend=mlx ...          once, then
            READY <max context> stop
    server  GEN <max new> [key=value ...] <id>,<id>,...
    engine  RESUME <tokens reused from the conversation held>
            PP <read> <total> <ms> <tokens/s>          after each chunk of the prompt
            T <id>                                     each token of the answer, the end token included
            DONE <tokens> <prompt tokens> <prompt ms> <decode ms> <stop|length|cancel> <guesses kept>
                 <guesses made> <reused> 0 <lookup passes> 0 0 0.0 <prompt tokens read> 0
            ERR <reason>                               instead, for a request that cannot run; no DONE follows
    server  STOP                                       ends the request under way; its DONE says cancel
            SAVE <path> | RESTORE <path>               between requests: the conversation held, to or from a file
    engine  SWAIT transfer <seconds>                   how long the one blocking step may take, then
            SAVED | RESTORED <tokens> <bytes> <ms>
            SERR <invalid|storage|memory|io> 0 <reason>  instead; the conversation held is as it was
    server  QUIT

    strata-mlx --serve --native <shard 1 of the GGUF> --mtp <folder of the draft head>

On a machine whose memory does not hold the model, only part of the experts stays in memory and the others are
read from the SSD as tokens ask for them (paged.py). How much is worked out from the machine's memory;
`--expert-memory <GB>` sets it by hand, and `--machine-memory <GiB>` runs as a machine of that size would.
Such a machine counts which experts its answers use, in `~/.cache/strata-mlx/expert-use-<model>.npy`
(`--expert-profile <path>` for another place, `off` for none), and starts with the most used ones in memory.

The draft head's guesses are worked out over the rows of the output layer that have been in sight (rows.py);
`--guess-rows off` has them go over all rows.

Between requests the GPU is not left idle for long (#1): a machine that holds the model keeps its arrays resident
(MLX's wired limit), and for `--keep-awake <seconds>` after a request (600 by default, 0 for never) the engine
gives the GPU one addition a second. Without the first, a prompt read after two seconds of nothing ran at
about half speed; without the second, the GPU took 250-320 ms to come back after four seconds of nothing
(bench/experiments/e38, M5 Ultra: a piece of 64 tokens in 250-400 ms after a rest so, 120-180 ms with both).

`strata-mlx --check [--native <shard 1>] [--machine-memory <GiB>]` loads nothing and says which of these a Mac
of that memory would do with a file (check.py).
"""

import errno
import hashlib
import os
import queue
import sys
import threading
import time

from .engine import Engine, Stats
from .paged import load_profile, machine_memory, save_profile
from .sampler import Sampling

VERSION = "0.0.1"
END_TOKENS = (248044, 248046)  # <|endoftext|>, <|im_end|>
_FLOATS = ("temperature", "top_p", "min_p", "penalty_repeat", "penalty_freq", "penalty_present")
_INTS = ("top_k", "penalty_last_n", "seed")
_IGNORED = ("cvec", "ckpt", "pin", "pcie_frac", "spec_min_p")  # for the CUDA engine's expert cache and its options
AWAKE_EVERY = 1.0  # seconds between the additions that keep the GPU awake: one in 2.5 s was enough as well (e38)


class Session:
    """One conversation slot: turns protocol lines into engine calls and writes the answers."""

    def __init__(self, engine: Engine, write, max_context: int, stop: threading.Event | None = None):
        self.engine, self.write, self.max_context = engine, write, max_context
        self.stop = stop or threading.Event()

    def handle(self, line: str):
        if line.startswith("GEN "):
            self.gen(line[4:])
        elif line.startswith("GENI "):
            self.write("ERR this engine was started without --vision")
        elif line.startswith(("SAVE ", "RESTORE ")):
            self.session_file(*line.split(" ", 1))
        elif line.startswith("VRAM"):
            self.write("ERR the MLX engine has no expert cache to resize")
        elif line.strip():
            self.write(f"ERR unknown command: {line[:40]}")

    def session_file(self, command: str, path: str):
        """`SAVE <path>` / `RESTORE <path>`: the conversation the engine holds, to or from a file."""
        e, t0 = self.engine, time.perf_counter()
        if not path:
            self.write("SERR invalid 0 missing path")
            return
        try:
            # one blocking step: tell the server how long it may take before it counts the engine as silent
            size = _state_bytes(e) if command == "SAVE" else os.path.getsize(path)
            self.write(f"SWAIT transfer {int(min(3600, 60 + size / 2e7))}")
            size = e.save(path) if command == "SAVE" else e.restore(path)
        except ValueError as err:
            self.write(f"SERR invalid 0 {' '.join(str(err).split())}")
        except MemoryError:
            self.write("SERR memory 0 not enough memory for the session")
        except OSError as err:
            kind = "storage" if err.errno in (errno.ENOSPC, errno.EDQUOT) else "io"
            self.write(f"SERR {kind} 0 {' '.join(str(err).split())}")
        else:
            done = "SAVED" if command == "SAVE" else "RESTORED"
            self.write(f"{done} {len(e.tokens)} {size} {(time.perf_counter() - t0) * 1e3:.1f}")

    def gen(self, rest: str):
        try:
            fields = rest.split()
            max_new, options = int(fields[0]), {}
            for f in fields[1:-1]:
                key, _, value = f.partition("=")
                if key in _FLOATS:
                    options[key] = float(value)
                elif key in _INTS:
                    options[key] = int(value)
                elif key not in _IGNORED:
                    raise ValueError(key)
            prompt = [int(t) for t in fields[-1].split(",")]
            vocab = self.engine.vocab
            if max_new < 0 or not prompt or len(fields) < 2 or any(not 0 <= t < vocab for t in prompt):
                raise ValueError
        except (ValueError, IndexError):
            self.write("ERR bad request")
            return
        if len(prompt) >= self.max_context:
            self.write(f"ERR the prompt ({len(prompt)} tokens) exceeds the context ({self.max_context})")
            return

        e = self.engine
        e.sampling = Sampling(**options)
        t0 = time.perf_counter()
        reused = e.resume(prompt)
        self.write(f"RESUME {reused}")
        e.stats = Stats()

        def progress(done, total):
            ms = (time.perf_counter() - t0) * 1e3
            self.write(f"PP {done} {total} {ms:.0f} {(done - reused) / max(ms, 1e-3) * 1e3:.1f}")
            if self.stop.is_set():
                raise _Cancelled

        produced, finish = 0, "length"
        try:
            token = e.read(prompt, on_chunk=progress)
        except _Cancelled:
            # the prompt is read as far as the last chunk; what was read stays for the next request
            self._done(0, prompt, reused, (time.perf_counter() - t0) * 1e3, 0.0, "cancel")
            return
        t1 = time.perf_counter()
        budget = min(max_new, self.max_context - len(prompt))
        pending = [token]
        while produced < budget:
            if self.stop.is_set():
                finish = "cancel"
                break
            if not pending:
                pending = e.step(budget - produced, END_TOKENS)
            token = pending.pop(0)
            self.write(f"T {token}")
            produced += 1
            if token in END_TOKENS:
                finish = "stop"
                break
        self._done(produced, prompt, reused, (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3, finish)

    def _done(self, produced, prompt, reused, prompt_ms, decode_ms, finish):
        s = self.engine.stats
        self.write(f"DONE {produced} {len(prompt)} {prompt_ms:.1f} {decode_ms:.1f} {finish} {s.accepted} {s.drafted} "
                   f"{reused} 0 {s.lookup_steps} 0 0 0.0 {len(prompt) - reused} 0")


class _Cancelled(Exception):
    pass


def _state_bytes(e: Engine) -> int:
    """About what a session file of this conversation takes."""
    caches = [*e.attn.values(), *([e.mtp_cache] if e.mtp_cache is not None else [])]
    kv = sum(a.nbytes for c in caches if c.keys is not None for a in c.state)
    small = sum(a.nbytes for pair in e.gdn.values() for a in pair)
    return kv + small * (1 + len(e.checkpoints))


def model_identity(path: str, dtype: str) -> str:
    """Names a model file without reading it all: its size and a hash of its first megabytes, where a GGUF
    file keeps its metadata and its table of tensors."""
    with open(path, "rb") as f:
        head = hashlib.sha256(f.read(8 << 20)).hexdigest()[:32]
    return f"{os.path.getsize(path)}:{head}:{dtype}"


def _options(argv: list) -> dict:
    """The arguments this engine knows; the CUDA engine's others (expert cache, K/V placement, ...) are skipped."""
    known = {"--native": None, "--mtp": None, "--spec": "4", "--max-context": "131072", "--dtype": "float16",
             "--prefill": "2048", "--suffix-draft": "auto", "--expert-memory": "auto", "--machine-memory": None,
             "--expert-profile": "auto", "--guess-rows": "on", "--keep-awake": "600"}
    out, skipped, i = dict(known), [], 0
    while i < len(argv):
        a = argv[i]
        if a in known:
            out[a] = argv[i + 1]
            i += 2
        else:
            skipped.append(a)
            i += 2 if a.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--") else 1
    return out, skipped


def main():
    argv = [a for a in sys.argv[1:] if a != "--serve"]
    if "--check" in argv:
        from .check import main as check

        return check([a for a in argv if a != "--check"])
    opts, skipped = _options(argv)
    # the protocol owns stdout: anything a library prints goes to the log instead
    wire = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)

    def write(line: str):
        wire.write(line + "\n")
        wire.flush()

    if not opts["--native"]:
        write("ERR --native <shard 1 of the GGUF> is required")
        return 1
    if skipped:
        print("strata-mlx: arguments of the CUDA engine that do nothing here:", " ".join(skipped), file=sys.stderr)

    if opts["--dtype"] == "float32":
        # #5: on chips with Neural Accelerators (M5) MLX multiplies float32 matrices at reduced precision unless
        # told not to, before its first computation; float32 is asked for to get the exact model's tokens
        os.environ.setdefault("MLX_ENABLE_TF32", "0")

    import mlx.core as mx

    from .load import load
    from .mtp import load_mtp

    t0 = time.perf_counter()
    dtype = getattr(mx, opts["--dtype"])
    machine = int(float(opts["--machine-memory"]) * 2**30) if opts["--machine-memory"] else machine_memory()
    wanted = opts["--expert-memory"]
    model, _ = load(opts["--native"], dtype, expert_memory=wanted if wanted == "auto" else float(wanted),
                    machine_memory=machine)
    store = model.expert_store
    paged = store is not None
    if not paged:
        # #1: resident, the arrays are not let go of by the GPU after a few seconds of nothing; not where experts
        # are read from the SSD, whose machine may need macOS to be able to move memory
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    if not paged and not model.keeps_floats:
        print("strata-mlx: the Q6_K layers' weights as floats do not fit beside the model; they are made anew for "
              "each chunk of a prompt, which reads prompts about a tenth slower", file=sys.stderr)
    mtp = load_mtp(opts["--mtp"], model.args.text, dtype) if opts["--mtp"] else None
    chunk = int(opts["--prefill"]) if opts["--prefill"].isdigit() else 2048
    # --spec is the CUDA engine's window; here anything above 1 turns the draft head on, and how many guesses it
    # makes for a pass is its own business (drafting.py)
    # Guesses copied from earlier text: by themselves when there is no draft head. With the head guessing as
    # many tokens as it is sure of they added nothing, measured (rewriting a file 100 against 101 tok/s, code 71
    # against 74), so they are then only on if asked for.
    head = mtp if int(opts["--spec"]) > 1 else None
    copies = opts["--suffix-draft"]
    engine = Engine(model, head, lookup=head is None if copies == "auto" else int(copies) > 0,
                    chunk=chunk, guess_rows=opts["--guess-rows"] != "off")
    engine.identity = model_identity(opts["--native"], opts["--dtype"])
    if paged and wanted == "auto":
        store.fit(machine)  # now that everything else is in memory: what is left is the experts'

    max_context = int(opts["--max-context"])
    # compile the kernels of every window size and of the sampler now, not during the first request
    engine.warm()
    for sampling in (Sampling(), Sampling(temperature=1.0, seed=1)):
        engine.generate(list(range(1000, 1024)), 12, sampling=sampling)
    # and a prompt long enough to go the way long prompts go, so that the kernels of that way are compiled too.
    # Not with experts read from the SSD: there it would read every layer whole.
    engine.reset()
    profile_path, before = None, None
    if not paged:
        engine.read(list(range(1000, 1600)))
        engine.reset()
    elif opts["--expert-profile"] != "off":
        # the experts this machine's answers used most, in memory before the first request: without them the
        # first answer after a start is the slowest. The warm-up's own experts and counts are let go first.
        profile_path = opts["--expert-profile"]
        if profile_path == "auto":
            name = hashlib.sha256(engine.identity.encode()).hexdigest()[:16]
            profile_path = os.path.expanduser(f"~/.cache/strata-mlx/expert-use-{name}.npy")
        before = load_profile(profile_path)
        store.clear()
        if before is not None:
            t1 = time.perf_counter()
            n = store.preload(before)
            print(f"strata-mlx: {n} experts ({store.held / 1e9:.1f} GB) read ahead in {time.perf_counter() - t1:.1f} s",
                  file=sys.stderr)
        store.reset_stats()
    print(f"strata-mlx: session is up in {time.perf_counter() - t0:.0f} s, "
          f"{mx.get_active_memory() / 2**30:.1f} GiB of weights", file=sys.stderr)

    stop, commands = threading.Event(), queue.Queue()

    def reader():
        for line in sys.stdin:
            line = line.rstrip("\n")
            if line == "STOP":
                stop.set()
            else:
                if line.startswith("GEN"):
                    stop.clear()  # a STOP that arrived after its request had ended
                commands.put(line)
        commands.put("QUIT")

    threading.Thread(target=reader, daemon=True).start()
    session = Session(engine, write, max_context, stop)
    write(f"INFO engine={VERSION}-mlx backend=mlx mlx={mx.__version__} context={max_context} "
          f"floats={opts['--dtype']} drafts={engine.drafts} lookup={int(engine.lookup is not None)} "
          f"guess_rows={int(engine.rows is not None)}"
          + (f" expert_memory_gb={store.budget / 1e9:.1f}" if paged else ""))
    write(f"READY {max_context} stop")
    awake, last = float(opts["--keep-awake"]), time.monotonic()
    while True:
        try:
            line = commands.get(timeout=AWAKE_EVERY if awake > 0 else None)
        except queue.Empty:
            if time.monotonic() - last < awake:
                mx.eval(mx.array(1.0) + 1)  # #1: the GPU kept from the sleep it takes 250-320 ms to come back from
            continue
        last = time.monotonic()
        if line == "QUIT":
            return 0
        session.handle(line)
        last = time.monotonic()  # any line from the server, not only a request: someone is using it
        if profile_path and line.startswith("GEN") and store.use:
            try:
                save_profile(profile_path, store.profile(before))
            except OSError as e:
                print(f"strata-mlx: the expert profile was not saved: {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
