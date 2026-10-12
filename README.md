# strata-mlx

Qwen3.8-Flash-Next on Apple Silicon: an MLX engine that reads the GGUF files
[Strata](https://github.com/Niko1221/Strata) uses and speaks Strata's line protocol, so Strata's server and
web page can drive it.

This is an independent, unofficial project. It is not affiliated with Strata or its authors, who have said
that macOS is out of scope for Strata itself. It is experimental and has been run on two machines, an M4 Max and
an M5 Ultra.

## What it is

Strata runs a 125B-parameter mixture-of-experts model on a 12 GB graphics card by keeping most experts in
host memory. A Mac with unified memory does not need that half of it. This engine takes the other half:

- **The file as it is.** The GSQ-RCO GGUF files load without a weight being changed. Experts stay in the
  file's own blocks, nine block types, and Metal kernels multiply from them. The 28.8 GB n-gram table stays
  on disk.
- **Draft and verify.** The model's own draft head guesses the next tokens and one pass of the model checks
  them. The model decides every token that is written.
- **Prompts.** Read in chunks, the experts a tile at a time under Apple's matrix product for shaders. A
  conversation is read once: a follow-up reads only what is new, and a conversation can be saved and restored.
- **Long conversations**, measured to 128K tokens of context.
- **Less memory than the model.** Experts that do not fit are read from the SSD as tokens ask for them, and
  the dense tensors MLX would hold at twice their size stay the file's blocks, which leaves the experts more.
- **Strata's line protocol** on stdin and stdout (`strata_mlx/serve.py`).

## Measured

M4 Max, 128 GB, macOS 27, MLX 0.32.3. Greedy decoding. Every number is from this one machine, which slows
down under sustained load, so read them as ranges. The two model files are Q2_0 (66.4 GB) and IQ3_S (83.6 GB).

Answers are written 1.6 to 2.9 times as fast as by llama.cpp or LM Studio, and a long prompt is read 4 to 36%
faster.

**Writing an answer**, tokens a second over four prompts of 256 tokens each (more is better):

| Engine | Q2_0 | IQ3_S |
| :--- | ---: | ---: |
| **strata-mlx (this project)** | **68-111** | **65-95** |
| strata-mlx, draft head off | 44-67 | 43-56 |
| llama.cpp `llama-server` 0.4.1 | 38-39 | 35-36 |
| LM Studio 0.4.25 (llama.cpp runtime) | 43 | 40 |

**Reading a prompt** of 1,180 tokens, tokens a second (more is better):

| Engine | Q2_0 | IQ3_S |
| :--- | ---: | ---: |
| **strata-mlx (this project)** | **570-578** | **510-517** |
| strata-mlx, draft head off | 595-612 | 504-515 |
| llama.cpp `llama-server` 0.4.1 | 494-495 | 488-489 |
| LM Studio 0.4.25 (llama.cpp runtime) | 426-446 | 421-424 |

With the draft head off the engine still copies guesses from earlier in the conversation where the answer
repeats it; one token a pass with no guesses at all is 46-48 tok/s (Q2_0) and 41-43 (IQ3_S).

These tables are from before two things were found, on an M5 Ultra: after a few seconds of nothing the GPU
let go of the model's arrays and then slept, so that a prompt of a few dozen tokens took about 0.6 s to the
first token where llama.cpp took 0.3; and this engine's reading speeds were from a later session than the other
engines'. Ollama 0.35.1 loads neither file.

On an M5 Ultra with 96 GB, every engine in one session and the server keeping the GPU awake, tok/s (llama.cpp
0.6.0; `--spec 1` is the draft head off):

| File | Answers | `--spec 1` | llama.cpp | Reading 1,180 tokens | `--spec 1` | llama.cpp |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q2_0 | **132-189** | 75-102 | 60-61 | **2,346-2,360** | 2,449-2,675 | 1,291-1,292 |
| IQ3_S | **103-152** | 71-90 | 57-58 | **2,364** | 2,384-2,483 | 1,288-1,304 |
| IQ3_XXS | **122-161** | 71-86 | 58-59 | **2,413-2,514** | 2,554-2,574 | 1,317-1,326 |
| IQ2_XS | **118-166** | 72-102 | 60-61 | **2,500-2,501** | 2,551-2,561 | 1,346-1,348 |

There a prompt of 27 tokens waits 0.14 s for the first token, 0.25-0.97 s with llama.cpp. The M4 Max has not been
run again.

In 16-bit floats, the default, about one token in 250 is not the one the exact model would pick, at places
where two candidates are nearly tied; that is so with and without the draft head. `--dtype float32` writes
the exact model's tokens (checked on eight answers) and is slower. On an M5, MLX by default multiplies float32
matrices only about as accurately as float16 ones, so `--dtype float32` and the tests turn that off
(`MLX_ENABLE_TF32=0`); on an M5 Ultra float32 so wrote the model's own token at every step of four answers
([issue 5](https://github.com/yibie/strata-mlx/issues/5)).

With part of the experts in memory and the rest on the SSD, limited on the same machine to stand for smaller
ones, Q2_0 only: answers at 11-12 tok/s with 3.3 GB of experts in memory, 18-27 with 11.9 GB, 23-38 with
20.5 GB; a prompt of 1,180 tokens read at 107-145 tok/s, since a chunk of a prompt reads 26 to 30 GB of
experts from the disk ([issue 4](https://github.com/yibie/strata-mlx/issues/4)). Those were what a 16, 24 and
32 GB machine kept when the engine left macOS 4.2 GB. A real 32 GB Mac, a MacBook Air M5, swapped with 20.5 GB
and wrote answers at 6-19 tok/s; with 16 GB it wrote them at 23-29
([issue 6](https://github.com/yibie/strata-mlx/issues/6)). So the engine now leaves macOS 7.9 GB, and keeps one
layer's experts free for reading the next layer while a prompt is read, which gives a 32 GB machine's experts
15.5 GB; the three figures above have not been measured again with it. Reading a prompt on such machines was
then made faster: on an M5 Ultra, 1,468 tokens as a 16, 24 and 32 GB machine at 390-403, 382-386 and 410-433
tok/s, where it had been 217-247, 214-216 and 128-183 ([issue 4](https://github.com/yibie/strata-mlx/issues/4)).

The full account, with what was tried and did not help, is in [ROADMAP.md](ROADMAP.md) (English, as it went)
and [docs/OPTIMIZATION_RECORD.md](docs/OPTIMIZATION_RECORD.md) (Chinese, by topic).

## Which Mac

`--check` loads nothing and says what a Mac would do with each file. It works before anything is downloaded,
and for a Mac other than the one it runs on:

```sh
./strata-mlx --check                         # this Mac
./strata-mlx --check --machine-memory 48     # a Mac with 48 GB
./strata-mlx --check --native <shard 1>      # a file on the disk, by its own header
```

By the same arithmetic the engine starts with:

| Mac's memory | Q2_0 (66 GB on disk) | IQ3_S (84 GB on disk) |
| :--- | :--- | :--- |
| 16 GB | too small | too small |
| 24 GB | 20% of the experts in memory | 7% of the experts in memory |
| 32 GB | 46% of the experts in memory | 24% of the experts in memory |
| 36 GB | 58% of the experts in memory | 33% of the experts in memory |
| 48 GB | 96% of the experts in memory | 59% of the experts in memory |
| 64 GB | whole | 93% of the experts in memory |
| 96 GB and more | whole | whole |

"Whole" is the path the speeds under [Measured](#measured) are for, on an M4 Max; other chips have not been
run. Where only part of the experts is in memory the rest is read from the SSD as tokens ask for it, which is
the slow path above. IQ3_S has not been run that way at all. This table is arithmetic, not a test: the only
Mac its author has run the engine on has 128 GB. It leaves 7.9 GB to macOS and other programs and does not count the conversation itself, about 1 GB for 36,000 tokens. That figure is
from the one small Mac the engine has been reported from, a MacBook Air M5 with 32 GB: with 4.2 GB left to
macOS it swapped, and with what the table now gives it it hardly did and was about twice as fast
([issue 6](https://github.com/yibie/strata-mlx/issues/6)). Other sizes are not known: a 16 GB Mac may do with
less than is left here, and `--expert-memory GB` sets the experts' share by hand.

## Running it

Needs an Apple Silicon Mac, Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/). Tested on macOS 27
only; the prompt kernel uses Metal 4's Metal Performance Primitives.

```sh
git clone https://github.com/yibie/strata-mlx && cd strata-mlx
uv sync
uv run pytest          # 332 tests, no model file needed
```

The model, about 66 GB for Q2_0: both shards of one size from
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
side by side in one folder (the second shard is the same file in all four sizes). Q2_0 and IQ3_S have been run on
both machines, IQ3_XXS and IQ2_XS on the M5 Ultra, all four checked against llama.cpp's answer from the same file.

The draft head is not in those files. Its 31 tensors, 4.9 GB, are fetched out of the original checkpoint by
range requests, without downloading its 360 GB, and each is checked against a SHA-256. The script is Strata's
own, here unchanged and where Strata has it, in [tools](tools):

```sh
python tools/mtp_fetch.py fetch --out ~/.cache/strata-mlx/mtp     # resumable
```

Without the head the engine still runs, at the "draft head off" speeds above.

```sh
./strata-mlx --serve --native <shard 1 of the GGUF> --mtp ~/.cache/strata-mlx/mtp
```

The process then speaks Strata's line protocol: it prints `INFO ...` and `READY`, takes
`GEN <max new tokens> <id>,<id>,...` and answers with `T <id>` lines and a `DONE` line. Prompts are token
ids; tokenizing, chat templates and the HTTP APIs are the server's part. Strata's `serve/server.py`
(0.1.40) has driven it unchanged, with its config's `exe` set to the `strata-mlx` script and the options above
in `args`; Strata's `setup.py` does not run on macOS, so that config is written by hand.

Options: `--spec 1` (draft head off), `--dtype float32`, `--max-context N`, `--expert-memory GB` (how much of
the experts to keep in memory; by default all of them if the machine holds the model), `--machine-memory GiB`
(run as a machine of that size would), `--guess-rows off`, `--expert-profile off`, `--keep-awake SECONDS` (how long
after a request the GPU is kept from sleeping, 600 by default, 0 for never).

The benchmarks in `bench/` tokenize with llama.cpp's `llama-tokenize`, and `bench/compare_servers.py` starts
`llama-server` and LM Studio's `lms` if they are installed.

## Not there

Image input, several conversations decoded together, and the files other than Q2_0 and IQ3_S. Conversation
caches are 16-bit floats (3.9 GB at 128K tokens). Nothing here has run on a Mac with less than 128 GB.

## License

MIT, see [LICENSE](LICENSE). The projects named below have their own; the one file taken from Strata,
`tools/mtp_fetch.py`, is Strata's and keeps Strata's MIT license, which is beside it.

## Thanks to

- [Strata](https://github.com/Niko1221/Strata) (MIT): the design this follows, the line protocol, the order
  of its sampler and the idea of its lookup drafter, all written anew here for MLX; and its script that
  fetches the draft head, included as it is.
- [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp) (MIT): the GGUF format and the block types,
  whose unpacking the kernels here follow; the codebook tables come from its `gguf` package at run time.
- [MLX and mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT), and the port of this model to mlx-lm in
  [pull request 1788](https://github.com/ml-explore/mlx-lm/pull/1788), which this engine builds on and pins.
- ISTA-DASLab for the GSQ-RCO files, and Qwen for the model. The model's weights have their own license and
  are not part of this repository.
