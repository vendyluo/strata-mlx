# Roadmap

## Done

- The model straight from Strata's GGUF files, without changing a weight (`strata_mlx/gguf.py`, `load.py`).
- Experts computed on the file's own blocks (`blocks.py`): nine block types, each checked against gguf-py's
  dequantizer on random blocks. Nothing is converted at start (22 s to load Q2_0, was 34-57), and in a pass the
  kernels are as fast as MLX's own 2-bit gather or a little faster (43 tok/s a token at a time, was 38-40), so
  Q2_0 uses them too. A chunk of a prompt multiplies from the blocks too, a tile at a time (e33, below); it
  unpacked each stack to floats for the length of one product before.
  When comparing with llama.cpp, use a prompt the model answers: forced past its own end token (`ignore_eos`),
  three implementations of the same file disagree by half a nat and the comparison says nothing.
- Draft-and-verify with the model's draft head, for greedy and for sampled answers (`engine.py`, `mtp.py`,
  `sampler.py`): a seed writes the same text with and without drafts.
- Guesses copied from earlier in the conversation, used where they settle more tokens per millisecond
  (`drafting.py`).
- A conversation read once: checkpoints at message boundaries, a follow-up reads only what is new.
- Prompts read in chunks, the output layer for the last token only.
- Strata's line protocol (`serve.py`, `./strata-mlx`): its server, web app and OpenAI / Anthropic APIs run on this
  engine unchanged.
- The sparse-attention path past 2,048 tokens, checked against llama.cpp at 2,100 and 6,000 tokens.
- Long conversations (`attention.py`, `bench/long_context.py`). The port attended to every cached cell under a
  mask and pooled the indexer's keys of the whole conversation again each pass; with several queries MLX's
  attention kernel then spreads the two key heads over the 24 query heads, 1.6 GB a layer at 128K cells. Now a
  pass gathers the 2,052 cells each query chose, and a block's pooled key is computed once. With the caches
  filled to a length (Q2_0, M4 Max; before -> now):

      cells cached    one token a pass    draft head, 4 a pass    a 2,048-token chunk     its peak above rest
          8K           26-31 -> 29 ms        69-71 -> 59 ms        260-409 -> 405 tok/s      3.8 -> 3.8 GB
         32K              32 -> 27             105 -> 56               288 -> 321            5.7 -> 3.8
         64K              36 -> 28             166 -> 59               219 -> 306            8.9 -> 4.1
        128K              45 -> 29             285 -> 60               115 -> 280           15.7 -> 5.0

  With the draft head a pass of four tokens at 128K was slower than four passes of one; now the head pays at
  every length. (A later run of the same, a browser busy on the GPU: 30-32 ms, 63-83 ms and 172-233 tok/s, at
  every length alike.) A chunk gathers from 12K cells on (8.6 GB of keys and values copied for 2,048 queries, a
  layer; under the mask the cost grows with the length, and inside a real read faster than measured alone). The
  queries of a chunk do not choose the same blocks, so there is no smaller set to gather once.
  Against llama.cpp on chat prompts of 3,161, 6,945 and 32,299 tokens: its token at 64 of 64 steps, a token or
  4 or 8 a pass. The 32,299-token prompt is read at 154-246 tok/s over five runs (llama.cpp on Metal: 267; the
  6,945-token one at 320-436) and the passes after it start at 31 ms, as with caches filled; llama.cpp writes
  11.5 tok/s there, 33-35 after 3K and 7K tokens. That evening other programs kept the GPU busy: 400 passes in
  a row took 29-65 ms by twenties at 2K cells and 37-51 at 32K, so the slower 32K answers measured (22-34 tok/s
  a token at a time) say nothing about the length.
- MLX's pool of freed buffers has a limit of 4 GiB (`load.POOL`). It had none: a chunk of prompt frees buffers
  of sizes the next layer does not ask for again, 37 GB after 2,048 tokens and 65 GB after 7,000, beside 41 GB
  of weights. With other programs' memory that left the machine swapping: a 7K-token prompt read at 60-100
  tok/s and a six-token chunk took six seconds. With the limit it reads at 320-415 tok/s, and the first long
  prompt after a start as fast as the next (it took twice as long: the pool being filled). Passes are the same
  at 1, 4 and 16 GiB and with no limit.
  Not done: the caches are 30 KB a token (3.9 GB at 128K) in 16-bit floats; at 8 bits they would be about half,
  which matters to the machines that page experts, whose budget does not count the conversation at all.
- A conversation on disk (`SAVE` / `RESTORE`, the server's `/slots/0?action=save|restore`): one file, written
  beside its place and renamed when whole, refused if it was saved with other model files. 1,515 tokens are
  264 MiB, saved in 0.6 s.

- `strata-mlx --check`: what a Mac of some memory would do with a file (held whole, read from the SSD in
  part, too small), by the rule the engine starts with and without loading anything; for a Mac that has no file
  yet the sizes of Q2_0 and IQ3_S are kept in `check.py`. By it Q2_0 is held whole from 46.7 GB of memory and
  IQ3_S from 66.1 (50.4 and 69.8 since macOS is left 8.5 GB, below). It counts the 3.3 GB of floats the IQ3_S file keeps for prompts, which the rule does not
  (ISSUES.md, issue 3).

## Now: what is left of Strata that is worth having on a Mac

- **The larger model sizes.** IQ3_S runs: 56 GB of weights (61 with the draft head), 38-45 tok/s a token at a
  time, 52-81 with the draft head (see "The IQ3_S file" below), about 440 tok/s reading a 1,500-token prompt. On a chat prompt it picks llama.cpp's token at 63 of 64 steps
  and its log-probabilities are closer to either llama.cpp backend (median 0.05 and 0.09) than those are to each
  other (0.09); llama.cpp on Metal writes 33 tok/s from the same file. All six expert types of the file unpack
  to gguf-py's weights exactly. IQ3_XXS and IQ2_XS will not be run: there is no disk space for more model
  files. Their tensors all have a path through the loader and their block types pass the same tests.
- **The same answer with and without drafts at full speed.** True in float32; in float16 a window and a single
  token round differently and the answers can part at a near-tie. What that costs, judged by the model itself
  in float32 a token a pass (e29): float16 writes a token other than the model's first choice about once in
  250, with drafts or without, with the listed rows or without, each time one the model found nearly as
  probable (at the widest 44.6% against 38.0%); float32 with drafts and the list wrote the model's own answer
  in all eight answers. `--dtype float32` costs half the speed with the Q2_0 file (44-49 tok/s for 80-103, its
  table lookup being float16 only) and a tenth to a fifth with IQ3_S (53-73 for 69-80).
- **Prompt reading speed.** About 440 tok/s for a 1,515-token prompt with the draft head (was 220; the first
  prompt after a start took four times as long until the loader's leftover buffers were released), 240-330 for
  8,192 random tokens. For the ceiling see the section on the GPU below.
- **The fixed cost of a pass.** A single-token pass is now 24 ms (was 28), a four-token window about 40 (was 43-47):
  layer halves compiled, hyper-connections as two Metal kernels for windows of one or two tokens. What was learned
  on the way: a pass is about 3,300 kernels in 229 command buffers, the GPU is busy 77% of it, and MLX's
  command-buffer limits change nothing. Cutting the kernel count by 43% bought 15%, because the kernels that are
  easy to merge are the cheap ones; hand-written kernels were slower than MLX's matrix products from three tokens
  up. No single large target is left: the expert gather already reads at 130-310 GB/s inside a pass. Getting
  near Strata's numbers would take a pass of a few hundred large kernels, each as good as MLX's own. The block
  kernels showed the same thing again: three times faster than MLX's gather alone, 6-10% faster in a pass.
  (Superseded: see "Faster answers" below. The pass was near the floor of the bytes it reads, and "no single
  large target" was wrong: the dense weights are seven times the experts' bytes.)
- Pictures (`GENI`), several conversations decoded together (`BGEN`).

## Faster answers: experiments (bench/experiments, results in bench/results/e*.json)

The yardstick is `e05_pass_rate.py`: passes of one token and of four, alternated, lower quartile. On a quiet
machine it repeats within 1%; other programs on the GPU move it by a half within minutes, so every change is
compared in one process, the two ways alternated pass by pass. Tokens a second are from `spec_decode.py`
(256 tokens, greedy, with thinking) on a quiet machine, with the draft head.

    step                                             one token   four tokens   prose / code / reasoning / edit
    start                                             22.3 ms      35.8 ms      61-63 / 69-70 / 77-82 / 77 tok/s
    hyper-connections read once for a window (e07)    22.1         34.0         63 / 73 / 82 / 82
    Q2_0 experts by table lookup (e12)                22.1         31.5         72 / 75 / 82 / 85
    n-gram rows on the CPU (e14), routers' own
      product, the experts' activation folded in      20.8         30.3
    one wait for the GPU a step; three guesses,
      then more while the head is sure (e13)                                    88 / 81 / 97 / 97

    a token a pass, no draft head: 43 tok/s at the start, 48 now.

What the measurements said, in the order they corrected each other:

- **A pass is not thousands of small kernels that cost by their number.** A token reads 5.5 GB of weights: 4.87
  of dense tensors (hyper-connections 1.27, delta-net projections 2.1, attention 0.6, output layer 0.5) and 0.66
  of experts. Summing those arrays once takes 13.9 ms; a one-token pass waits 20 ms for the GPU. An earlier note
  here put the bytes at 2 GB and the floor at 5 ms, and three rounds of merging kernels bought 15%.
- **Parts timed alone do not add up to a pass** (e01, e04, e05). MLX commits its command buffer after every
  50 MB of inputs, 23 us each time, so a kernel with a 236 MB stack of experts as input costs 25 us alone; in a
  pass the GPU is what is waited for and the limits (20 to 500 operations, any size) change nothing. Parts are
  measured by leaving them out of a real pass (e06): of 23.8 / 38.1 ms for one / four tokens at the start,
  routed experts 4.1 / 12.6, delta-net projections 6.0 / 8.7, hyper-connections 5.1 / 8.1, attention projections
  2.3 / 3.9, output layer 2.3 / 3.1, routers 2.0 / 2.7, shared experts 1.8 / 2.5.
- **One GPU thread is slow; a kernel takes as long as its longest thread** (e08, e09). A matrix of 324 by
  10,240 times four tokens: 38 us with 32 threads to a row, 21 with 1,024. A norm kernel of 128 threads looping
  160 times: 21 us for nothing. An array local to a thread: six times slower. From this the hyper-connections'
  three kernels and the routers' product (fused.py).
- **The expert kernel is bound by arithmetic, not by bytes or loads** (e02, e03, e10, e11): about eight
  operations a weight at all this GPU has, whatever the inner loop, the threads to a row, four values to a load
  or rows sharing their activations. A table of the 16 sums two weights make of two inputs is three operations
  a weight; a table of 256 sums for four weights does not stay in a cache and is four times slower (blocks.py,
  `lookup_matvec`; e12).
- **Waits for the GPU that nobody needed** cost 3.6 ms of a 41 ms step: the context tokens were kept as a GPU
  array (a slice and a concatenation waited for at every commit, and again by the n-gram lookup), and the states
  a pass leaves behind were waited for although only the next pass needs them.
- **How many guesses** (e13). The head's sureness of a run of guesses (the product of the probabilities it
  gave them) predicts whether they are kept, well for answers that repeat or reason step by step, poorly for
  prose. Three guesses in any case and more while it is sure: replayed, 6% over three every pass; measured, 2%
  on prose, 5% on code, 6% on reasoning, 10% rewriting a file. Deciding every guess by sureness lost on prose.

Tried, no gain:

- MLX's own switches (`MLX_MAX_OPS_PER_BUFFER`, `MLX_MAX_MB_PER_BUFFER`, `MLX_METAL_FAST_SYNCH`,
  `MLX_BFS_MAX_WIDTH`).
- The expert kernel loading four values at once (e10), or several rows to a thread (e11).
- A dense projection on the file's Q3_K blocks (e15): 74 / 80 / 91 us for 1 / 2 / 4 tokens against MLX's 77 /
  78 / 81 on its 8-bit form, 2.6 times the bytes. MLX's quantized product is flat up to four tokens and reads
  at 370-400 GB/s; the block kernel is held back by its million threads and by what each must do to undo the
  format. (With the layer's scales left in float32, MLX's product takes 134 us for four tokens: the first run
  of this experiment had it so, and the block kernel looked a third faster.) Where experts are read from the
  SSD the blocks are used since, for the memory they spare and with another kernel: e36.
- A copy of the output layer at fewer bits for the head's guesses (e16): 0.2 to 0.4 ms less a guess, and 3%,
  8%, 14% of the guesses change at 4, 3, 2 bits.
- A kernel for picking the largest logit: in a chain MLX's argmax of 248,320 takes 30 us (e17); the 0.4 ms it
  seemed to take alone was the wait for the GPU.

- Guesses copied from earlier text on top of the head's own (400 tokens, quiet machine): with three guesses a
  pass they add 7% rewriting a file (94 to 101 tok/s) and nothing on code; with the head guessing as many as it
  is sure of, nothing (101 to 100, 74 to 71). The server now turns them on by itself only without a draft head.

Not yet tried: the ten largest of a router's logits and their softmax as one kernel (about 18 us a layer
now); a block kernel for IQ4_XS, whose format is simpler than Q3_K's; the head's own layer, which is MLX
operations throughout. Each is under a millisecond a pass.

Changes weights, so not done, only measured (e18): the hyper-connections' 1.27 GB of 16-bit floats at 8 bits
would spare a pass 0.6 GB, about 1.6 ms. Rounded to 8 bits and compared with llama.cpp's answers: its token at
64 of 64 steps on the chat prompt as before, at 62 of 64 on the 7K-token one (64 before); at 6 bits 62 and 61,
at 4 bits 60 and 54.

### The IQ3_S file (e20 to e24)

Of the changes above, all but the table lookup apply to any file and were already in what IQ3_S ran with. What
that file has of its own is not in its experts but in its dense layers: 1.88 GB of them are Q6_K (the Q2_0
file has 84 MB), which scales every 16 weights, and MLX takes no scale to fewer than 32 (`quantize` refuses, the
product has no kernel). The loader unpacked Q6_K to floats: 4.58 GB of float16, so that a token read 7.16 GB of
dense arrays where the Q2_0 file's token reads 4.87, and the output layer's 1.27 GB again for every guess of the
draft head.

    step                                         one token   four tokens   a token reads   prose / code / reasoning / edit
    start (Q6_K as float16)                       26.6 ms      38.8 ms      7.16 GB         40 / 54 / 66 / 74 tok/s
    Q6_K as a byte a weight (dense.py, e23)       22.2         35.2         5.03            52 / 67 / 81 / 81

    a token a pass, no draft head: 28 / 33 / 36 / 37 tok/s before, 38 / 42 / 44 / 45 after.

Tokens a second: before and after a prompt at a time, in turns, a process each (the machine sped up through
that run, so a column compares, a row does not). llama.cpp's token on the chat prompt at 63 of 64 steps as
before. With the Q2_0 file, whose six Q6_K layers go the same way now: 64 of 64 on the chat prompt as before,
and on the 7K-token one 63 of 64 a token at a time (64 before), 64 of 64 four at a time. Its answers are as
fast as before or a little faster (in turns as above: 35-46 tok/s a token at a time against 35-45, 53-97 with
the head against 54-95).

- **On the file's blocks** (e20, e21): 6.56 bits a weight, and faster than MLX over float16 for one and two
  tokens (72 and 83 us against 130 for a 10,240 x 2,560 tensor), level at four, 1.7 to 3 times slower at six and
  eight: MLX's product stays at 130 us whatever the window, the block kernel undoes the format's interleaving
  for every weight and then multiplies for every token.
- **A byte a weight** (e22, e23): the code as a signed byte, the scales and d as the file has them, the same
  number at 8.56 bits; a kernel laid out as MLX's quantized product. In a pass, against float16: 4.0 ms less for
  one token, 3.5 for two, 3.2 for four, 1.4 for six, 1.7 more for eight. One form in memory, 2.45 GB for 4.58.
- **The draft head's guesses** (e24): three guesses cost a step 7.5 ms of 43, of which 4.1 are the output layer
  (2.9 of 6.2 with the Q2_0 file): every guess is all 248,320 rows for one token. See the next section.
- **The experts** were left: a four-token window reads 3.9 GB of them, which is most of the 11 ms they take.

Timings of answers on this machine move by up to a half for minutes at a time, and nothing but the GPU's own
speed shows it (bench/quiet.py). At least part of it is the machine: a 14-inch MacBook Pro whose GPU does a
4096-cube product at 14.5 T operations a second at rest does it at 12 after twenty seconds of full load, with
the thermal state then "fair"; in the middle of long benchmarks it was at 9 and below. So a benchmark's later
prompts run slower than its first, and before and after are compared a prompt at a time, in turns, a process
each. The first IQ3_S numbers of this round (25-27 tok/s a token at a time, 36-49 with the head) came from the
slow stretches and are void.

### Guesses over the rows that have been in sight (e25 to e28, rows.py)

A guess that comes out wrong costs speed, never the answer, so a guess need not look at every row of the output
layer if the row that wins is seldom one it skips. Recorded in four answers (1,190 guesses): of the guesses the
trunk kept, 72% were tokens of the conversation itself and 98% were that or among the 256 largest of some row
of logits the trunk had worked out by then, which are there for nothing. So those rows are listed, about
13,000 after an answer and 30,000 after four, a twentieth to an eighth of the layer, and a guess multiplies
only them. The list is kept between conversations; the largest of a pass's rows are picked out by the GPU
while the CPU makes the draft and are listed a step later.

In turns with guesses over all rows, one process, two rounds a prompt (e28):

    three guesses a pass, Q2_0            +5.4 +4.8 | +2.0 +7.6 | +3.8 +6.7 | +9.7 +7.9 %     tokens a pass as before
    three guesses a pass, IQ3_S           -0.6 +12.2 | +6.7 +8.7 | +3.6 +8.3 | +6.4 +9.1 %    as before but for the first
    as many as the head is sure of, Q2_0  -1.2 +7.4 | -3.6 +7.5 | +4.5 +13.0 | +14.0 -1.2 %   (5% on average)
    as many as the head is sure of, IQ3_S +8.1 +6.9 | +6.4 +11.0 | +4.1 +5.7 | +7.4 +10.9 %    (8% on average)

The first answer after a start has a short list and loses a few guesses to it (69% kept where 70% are, with
the Q2_0 file). With IQ3_S that first answer also parted from the other engine's at token 107, a near-tie in
float16 that the different windows tipped, and was no faster. With as many guesses as the head is sure of, a
run differs from the next by more than the list changes it:
the same engine settles 3.8 and then 4.4 tokens a pass on the same prompt, the policy having learned in
between. The head's probability of a guess is now taken among the listed rows; that makes it too sure by under
0.1% for nine guesses in ten.

Tried on the way: a sketch of the layer (its rows squeezed to the directions they vary most in) to pick the
rows instead of a list: the rows vary in all directions alike, and it takes a fifth of the layer to find 99% of
the winners (e25). Listing the rows within some distance of the largest instead of a fixed number: a row with
no clear winner lists tens of thousands (e27). Picking the largest out and waiting for them: 1.1 ms a step,
a third of what the list spares (e27).

### Against llama.cpp's server and LM Studio (bench/compare_servers.py)

One file, the same prompts as text with their chat markers, each engine loading the model alone and reporting
its own timings; 256 tokens an answer, the most likely token every time; "reading" is a prompt of 1,180 tokens.
Tokens a second, prose / code / reasoning / edit, then reading:

    Q2_0    llama-server 0.4.1                38-39 / 38 / 38 / 38                494-495
            LM Studio 0.4.25 (llama.cpp)      43 / 43 / 43 / 43                   426-446
            this engine, --spec 1             44-48 / 50-53 / 48-51 / 61-67       595-612   (320-385 before e30-e33)
            this engine as it serves          68-91 / 70-86 / 105-107 / 95-111    570-578   (370-386)
    IQ3_S   llama-server 0.4.1                35 / 35-36 / 35-36 / 35             488-489
            LM Studio 0.4.25 (llama.cpp)      40 / 40 / 40 / 40                   421-424
            this engine, --spec 1             43-45 / 49 / 43-45 / 53-56          504-515   (349-377)
            this engine as it serves          65-66 / 79-81 / 88-95 / 87-88       510-517   (334-349)

Answers come about twice as fast as from either (1.6 to 2.9 times). Prompts were read slower than by both
when this was first run; since the experts of a prompt go in tiles from the blocks (below) the 1,180 tokens are
read faster than by either, a prompt of 340 tokens about as fast as by llama.cpp with the Q2_0 file (419-451
against 427-433) and slower with IQ3_S (319-367), and prompts of 27 to 49 tokens slower (31-112 against 92-148).
Ollama 0.35.1 loads neither file: its GGUF reader does not know the Q2_0 tensors, which the IQ3_S file has too.
`--spec 1` is not a token a pass: with no draft head the server copies guesses from the context. The other
engines' rows are from one run of all engines in turn, twice in opposite orders; this engine's are from a run
of its own afterwards, on the code of 25cf76e.

That comparison found a fault: with Q6_K as a byte a weight, the IQ3_S file's prompts were read at 246 tok/s
(the Q2_0 file's at 390-400). A chunk of a prompt multiplied those layers as floats worked out anew by MLX
operations, half a second for every piece a prompt is read in. Now one kernel makes the floats (12 ms), at the
first prompt, and they are kept: 3.3 GB more held with the IQ3_S file, 0.2 GB with Q2_0. Made anew for each
prompt they cost a tenth of the reading speed (320 against 355-369 tok/s), the fresh memory and not the work.

### Why prompts are read slower: the ten chosen experts (e30)

Asked after the comparison, and not measured before: the loop above was all about answers. A chunk of a prompt
is few kernels, each milliseconds long, so its parts can be timed in place (each made to finish at its edges;
that costs a fifth more time with the Q2_0 file). A piece of 1,024 tokens, Q2_0, 2.4-2.5 s without the stops:

    the ten chosen experts                         1570-1630 ms   52-54%    4.83 T operations   3.0-3.1 T a second
    delta-net projections                           400-430       14%       4.27                10.0-10.6
    hyper-connections                               230-270        8%
    choosing the ten, weighting and adding them     210-230        7%
    the shared expert                               140-170        5%       0.48                2.9-3.5
    attention projections                           117-118        4%       1.22                10.4-10.5
    delta-net convolution and norms 70, its recurrence 53, attention itself 42-44, the routers' product 25-29,
    the n-gram table 18, glue 17-20, keeping the states 9, the output layer 5-6

The stops flatter the experts' share: without the stops, and with the ten chosen answering zeros, the piece
takes 850 ms where it takes 2,990 (both files, a warm machine). They are 71% of a prompt's time, at 2.3 T
operations a second, when the dense products of the same pass run at 10 and the GPU does 14.3 in one large
product. All else of a pass would read 1,200 tokens a second. So what an earlier note said, that reading is
near what this GPU can do and the gap to Strata is the hardware's, holds for a third of the pass and not for
the rest.

Why they are slow, as far as measured: 1,024 tokens are 10,240 pairs for 372 experts of a layer, 27.5 each,
so every product is small; in tiles of 16 tokens 26% of the slots are filling; a scale to every 64 weights
makes each tile forty products; and every chunk splits all 512 stacks of every tensor into codes and scales
whatever its length. With the IQ3_S file, which has no 2-bit operand, each stack goes through 1.7 GB of floats.

The two ways for Q2_0, forced at each length (ms a piece: 2-bit product in tiles / table of sums a pair):

    128 tokens   811 / 376      256   1066 / 667      512   1702 / 1492      1024   2610 / 3469      2048   4143 / 6332

The engine changes to tiles from 3,000 pairs (300 tokens), set before the table of sums existed; at 512 tokens
the table is still the faster by an eighth. Not changed yet: where they cross between 512 and 1,024 is not
measured. Not the cause: the cut before the last token (a prompt of 1,180 is read as 1,179 and 1), the draft
head reading along (3,140 ms against 3,113), the n-gram table, the recurrence.

What would help, it seemed, was a kernel that reads an expert's blocks once for all the tokens that chose it.
Measured before building it (e31, e32), the product was not what was slow:

    a layer's ten chosen, their inputs given (e31, ms, the mean of five layers of a real pass, 10,220 pairs):
    splitting the three stacks into codes and scales 4.1, the sums of the activations 2.3, the three products
    4.7 + 4.6 + 4.4, sorting the pairs on the CPU 0.5, the gathers 1.4, silu 0.5: 20.4 in all, 1.0 s a chunk

    one product (gate), T operations a second: the 2-bit product in tiles 7.1, MLX's gather_mm over the stack
    as float16 8.1, MLX's gather_qmm over the stack requantized at 2 bits 6.8, the table of sums 2.8

So the product was already where MLX's own grouped products are, and a chunk paid 1.7-2.1 s for 1.0 s of
experts. The rest was the wait: the pairs were sorted by expert on the CPU, so a pass stopped at every layer
until the GPU had the routers' choice, and what followed each stop ran slower. With the choices known
beforehand (taken from a pass before) the same chunk took 1.8 s instead of 2.7 (e32).

Now the GPU sorts them (blocks.tiles_of): the tiles are laid out by array operations, in as many tiles as the
pairs could need at most, and the kernel leaves out those nothing fills. In turns, one piece of 1,024 tokens:

    sorting on the CPU 2723 ms (376 tok/s)   the choices known beforehand 1821 (562)   sorting on the GPU 1832 (559)

The first token of the answer is the same each way; against llama.cpp's answers after prompts of 3,161 and
6,945 tokens the top token is equal at 64 of 64 and 63 of 64 steps a token at a time, 64 and 64 four a pass,
as before. A prompt of 1,180 tokens as the engine reads it: 2,298 ms (514 tok/s) where it took 2,885-3,113.
By length, tiles / table of sums, tok/s: 192 tokens 281 / 381, 256 330 / 386, 320 380 / 388, 384 410 / 390,
1,024 561 / 361, 2,048 627 / 344; the change of way stays at 3,000 pairs. Peak memory with chunks of 2,048:
4.6 GB over the weights. The ten chosen are now 54% of a chunk (980 ms of 1,810), at 4.9 T operations a
second.

The IQ3_S file's experts did not go this way (no 2-bit operand): each stack was unpacked to 1.7 GB of float16
(5.9 ms) for MLX's gather_mm (5.2 ms), three times a layer: 34 ms a layer step by step, 43 in a pass, 79 when
a layer's three products are made to finish together (three stacks of 1.7 GB at once).

Then the kernel (e33, blocks.tile_matmul), for every type: a threadgroup takes a tile of 16 tokens and 64 rows
of the tile's expert, unpacks 64 columns of those rows into threadgroup memory (`unpack32`, as the other
kernels) and Apple's matmul2d multiplies the tile's tokens by that operand; then the next 64 columns. Nothing
is split, summed or unpacked beforehand. On layers of a real pass, one product of 10,220 pairs:

    IQ2_S 5.3 ms   IQ3_S 5.1   IQ3_XXS 5.1   IQ4_NL 4.8   Q2_0 4.5-5.0        6.3 to 7.4 T operations a second
    a layer's three products: 16.3 ms; before, Q2_0 22.2 (split 4.1, sums 2.3, products 13.7), IQ3_S file 43-79
    tiles of 32 tokens, or 128 columns at a time: a fifth slower; 256 columns (all of a threadgroup's memory):
    wrong results on the real shapes. matmul2d sets its destination, it does not add to it.

So the 2-bit operand of Apple's primitive, the split and the sums are gone again (e30 to e32 are of that code).
A piece of a prompt, in tiles / a pair at a time (the table of sums for Q2_0, the matvec kernel else), tok/s:

    Q2_0     64 tokens 147 / 154   128 283 / 309   192 429 / 362   256 476 / 368   512 593 / 383   1,024 679 / 378   2,048 696 / 337
    IQ3_S    64 tokens  89 /  85   128 166 / 178   192 251 / 241   256 337 / 277   384 467 / 288   1,024 633 (358 unpacked)   2,048 671 (382)

Tiles from 1,600 pairs on. As the engine reads, with the draft head reading along, three prompts each:

    Q2_0     1,180 tokens 555-560 tok/s   2,620 652-653   8,668 420-507   peak 2.5 to 4.0 GB over what is held
    IQ3_S    1,180 tokens 488-513         2,620 633-639   8,668 455-614   peak 5.6 to 7.1 GB (3.3 of them the
             Q6_K layers' floats, kept from the first prompt on)

The comparison's prompt of 1,180 tokens was read at 370-386 (Q2_0) and 334-349 (IQ3_S) before; llama.cpp reads
it at 494 and 488, LM Studio at 426-446 and 421-424. The longest prompts slow down as they go: each read is
15 to 20 s of full load, and the machine slows under it (above).

Checks. Every type on random blocks against the unpacked weights, float16 and float32: 2e-7 of the largest
value. On real layers against a pair at a time: 1e-4 to 5e-4 a product, 1e-3 through all three, as the ways
before. Against llama.cpp's saved answers, top token equal a token at a time / four a pass: Q2_0 after 1,537
tokens 64 / 64, after 3,161 64 / 64, after 6,945 64 / 63 (before 63 / 64), in float32 after 3,161 64 / 64;
IQ3_S after 1,537 tokens 64 / 64 against both its Metal and its CPU answer (before 63). With part of the
experts read from the SSD (1.5 and 18.7 GB of them in memory) the answer starts with the same token. (This
said the 1,537 tokens were then read at 273-287 tok/s where they had been at 136-159. That was measured with
the model's file in the page cache, which answered for the disk: with it dropped it is 146-171. See e36 below.)

The ten chosen are now half of a chunk's time (49-54%), at 4.9 to 6.1 T operations a second; all else of a
pass would read 1,200 tokens a second. What is left in them: a tile costs the same with one pair as with
sixteen, and 26% of the slots are filling at 1,024 tokens.

Open, seen on the way and not looked into: a short piece right after an idle stretch is slow. 64 tokens take
179 ms one after another and 430 to 580 ms after 2 to 8 s of nothing; one token of an answer is 21 ms either
way, 256 tokens show little of it. A question of a few dozen tokens therefore waits half a second for its
first token.

## What the Apple GPU can do, measured (M4 Max, macOS 27)

Looked into because prompts are read three to six times slower here than by Strata on an RTX 5070, and the
question was whether frameworks leave the Apple GPU's speed unused. They do not, on this generation:

- Metal 4 has a matrix product for use inside a shader (`mpp::tensor_ops::matmul2d`, Metal Performance
  Primitives), with int8 x int8 -> int32 and with weights packed at 4 or 2 bits as an operand. It builds and runs
  inside `mx.fast.metal_kernel`. At 4096 x 4096 x 4096 (`bench/gpu_matmul_types.py`, results checked against
  numpy): half 14.3 T operations a second, float 9.0, int8 12.8, half x packed 2-bit 13.4, int8 x packed 2-bit
  11.8. MLX's own matmul: half 14.3-14.7, float 12.3. So the integer forms are no faster than half on this GPU,
  and MLX already runs at what Apple's own primitive reaches.
- An earlier note put this GPU's ceiling for reading prompts at 650-750 tok/s, from 8-9.5 TFLOPS measured at
  2048^3. At the larger size it is 14.3: about 1,100 tok/s if every product ran at that rate, which the small
  per-expert ones do not: they run at 2.3 and are 71% of the time (e30, above), so the engine and not the GPU
  is what holds prompts at 320-440 tok/s.
- The hardware, from the reverse-engineered instruction set (M1 generation; nothing public for M4): scalar,
  32 threads to a SIMD group, fused multiply-add in 32 and 16 bits, integer multiply-add and bit-field
  instructions, no dot-product instruction. Apple's guide says the primitive uses "GPU neural accelerators in
  the Apple M5 chip"; MLX and llama.cpp both ship kernels built on it and switch them on by GPU generation.
  Not tested: there is no M5 here.
- The CPU has SME2 with 512-bit vectors (sysctl). Accelerate's sgemm measures 3.3 TFLOPS here, a quarter of the
  GPU; a paper measures 8 T int8 operations a second from two threads on an M4 Pro.
- For this engine, built (`blocks.packed_matmul`): a prompt's Q2_0 experts go through the primitive as 2-bit
  codes. The scale of a Q2_0 block sits inside its 18 bytes and differs every 64 weights, so the stack is split
  into a plane of codes and one of scales (1.3 ms, 0.24 GB instead of 1.7 unpacked), the product is taken 64
  weights at a time into registers and each part scaled as it is added. In tiles of 16 tokens by 64 outputs it
  runs at 10.5 T operations a second (tiles of 64 tokens: 4.6). Real model, back to back after a warm-up: a
  700-token prompt 346 against 281 tok/s, 1,537 tokens the same 430-446, 4,611 tokens 367-424 against 280-360,
  8,192 random tokens 218-365 against 162-342 (this machine's timings move that much between rounds; in every
  pair the packed way was the faster or equal). The peak of memory above rest while reading: 2.5-4.3 GB against
  7.4-11.4. Same tokens as before against llama.cpp (64/64 and 63/64). Read from the SSD on a small machine it
  replaces the pair-at-a-time kernel: 136, 146 and 159 tok/s for the 1,537-token prompt as a 16, 24 and 32 GB
  machine (were 103, 124, 135); most of that time is reading every layer from the disk.
  Not for float32 (the primitive has no float32 by 2-bit product) nor for the codebook types of the larger files,
  which still unpack to floats.
- None of this touches how fast answers are written: a token's products are matrix-vector, bound by memory and
  by the number of kernels in a pass.

## Macs with 16, 24 and 32 GB: a first version runs

On a machine whose memory does not hold the model, only part of the experts stays in memory, the least recently
used dropped first; a layer waits for its router, reads what is missing from the SSD and goes on
(`strata_mlx/paged.py`). How much stays is worked out from the machine's memory: 4.5 GB are left to macOS and
other programs, 4 GB to what this process needs beyond its weights, the dense weights and the draft head take
7.4 (5.6 since e36 below), the experts get the rest. `--expert-memory <GB>` sets it by hand; `--machine-memory <GiB>` runs as a
machine of that size would, which is how these were measured, on the M4 Max with Q2_0, the engine process driven
over the line protocol with the draft head, the file dropped from the page cache and reads going past it:

    machine    experts in memory    process (peak)    answers           1,537-token prompt
     16 GB          1.5 GB          10-11 GB (12)      6-7 tok/s         136 tok/s
     24 GB         10.1 GB          18-19 GB (21)     19-23              146
     32 GB         18.7 GB          27 GB (29)        28-47              159
    128 GB       all, 34 GB         41 GB             64-83 (116 rewriting a file)   440

(Since e36 below: 3.3, 11.9 and 20.5 GB of experts in memory and answers at 11-12, 18-27 and 23-38 tok/s.)

**A real 32 GB Mac, and 8.5 GB left to macOS** (ISSUES.md, issue 6). The first small machine heard from, a
MacBook Air M5, swapped with those 20.5 GB of experts (106 GB swapped out during a run, answers at 5.8-19.2
tok/s) and was about twice as fast with 16 (23.4-26.8, and 24.3-28.8 with nothing else open). The arrays peaked
at the experts plus 8.5 GB in every run, as the rule counts; macOS itself held 6.1 GB, not 4.5. Capping the
budget on a 128 GB machine could not show this: nothing else is short of memory there. So macOS is left 8.5 GB
now, and a machine of 24 or 32 GB keeps 7.6 or 16.2 GB of experts; a 16 GB one is left the least, 0.5 GB, on
no 16 GB machine's evidence. The table above is of the budgets it names, not measured again.

Tests hold it to the same tokens as the model with every expert in memory, at a budget of nothing, of five
experts and of all, and with a window spread over several kernel calls. By budget, a token at a time and with
the draft head (`bench/paged.py`): 2 GB 7.5 and 10 tok/s, 210 misses a token; 4 GB 7-8 and 12, 148; 8 GB 12 and
20, 63; 16 GB 18 and 34-41, 17. The disk here gives 6.5 GB/s; a slower one lowers every row but the last. A page
cache answering for the disk made the same run 2.4 times faster, which is why the cache is dropped first.

What it took, and what is still wrong:

- One expert swapped into a large array copies the array (2 ms), so each expert is three arrays of its own and the
  kernels take a token's ten as ten inputs; joining them first cost 8 ms a token.
- The wait for each layer's router costs 10 ms a token even when nothing is read: with every expert resident this
  way of running gives 22 tok/s, not 43. A machine that holds everything should not use it.
- The loader left 7 GB behind for 5.6 GB of dense weights (what numpy frees stays with the process) and peaked
  26 GB above the draft head's 1.6: both now convert a few rows at a time, to the same weights bit for bit.
- macOS puts any read of 2 MB or more through the page cache, F_NOCACHE or not: whole tensors are read in 1 MiB
  pieces.
- Tried and closed: reading straight into the buffer of an array that already exists (to spare making arrays).
  The bytes arrive, the GPU goes on seeing the old ones. A new array for every read costs nothing measurable.
- A knob not yet turned for the prompt peak below: smaller chunks (`--prefill 512`) hold less at once and read
  every layer four times as often.
- Reading a prompt held every delta-net layer's q, k and v of every token through the pass, and afterwards
  the whole convolution input behind each few-row slice of it: 4.6 GB at the peak and 1.4 GB left for 1,500
  tokens, now 2.1 and 0.4.
- The budget is filled at the start with the experts used most (`ExpertStore.preload`): by a profile of seven
  conversations that comes with the engine (`strata_mlx/expert_use.npy`), then by the machine's own answers,
  counted in `~/.cache/strata-mlx/expert-use-<model>.npy` (`--expert-profile off` for neither). It reads 1,080,
  7,293 and 13,507 experts in 0.3, 1.8 and 3.3 s as a 16, 24 and 32 GB machine. It buys little: the first answer
  after a start goes from 29.5 to 31.5 tok/s as a 32 GB machine (34.8 started again with the profile the same
  three requests left; 45 for an answer that follows one like it), and stays at 22 and at 6-7 as a 24 and a
  16 GB one. Which experts a conversation needs is not what was used most before: the profile's 13,500 hold 91%
  of all uses, and one miss in a layer is a read that the whole pass waits for.
- Done since (e36 below): the dense weights' 5.6 GB as arrays were 2 GB more than in the file; two types of
  them now stay the file's blocks and the experts have 1.7 GB more.
- Not done: a layer still waits for its router and for
  its reads (asking for them ahead was measured and not built, e35 below; the experts the last token used are
  no guess, they are in memory already); no prompt past 12K tokens has
  been read this way, where attention gathers 0.5 GB at a time and the conversation's caches (30 KB a token)
  come out of memory the budget does not count; and none of this has run on a machine that small or on a
  256-512 GB SSD.

### Reading experts before their router has run: measured, not built (e35)

With part of the experts in memory a layer waits for its router and then for the disk, and the GPU has nothing
to do meanwhile. Others ask the disk early, for experts guessed with the model's own routers on an earlier
state: Eliseev and Mazur (arXiv:2312.17238), slotstream for this model, mlx-stream for DeepSeek. Before
building that here, what it could give was worked out from what is measured and from one recording.

From `bench/results/paged_*.json`, with the draft head, the disk is 65-71% of an answer's time with 2 GB of
experts in memory (a 16 GB Mac), 58-60% with 4, 43-45% with 8 (a 24 GB Mac) and 6-14% with 16 (a 32 GB one).
Reads asked for ahead go on beside the GPU's work and no faster: a token then takes the longer of the two, and
the reads not guessed besides. The rule, set before the recording: build it if that comes to 1.5 times at 2 or
at 8 GB.

Recorded with every expert in memory, a token at a time, seven conversations, 2,083 tokens of answers: the ten
experts each layer routes to, and a router's best N for three earlier inputs. Of the ten, among the best N:

    guess                                                              best 6    best 8    best 10   best 16
    a  layer L's router, on what its hyper-connection reads before
       L's mixer has written (the mixer ahead)                          53.9%     67.9%     78.0%     89.2%
    b  the same on the residual as layer L-1's experts find it
       (a layer ahead)                                                  53.8%     67.2%     77.0%     89.3%
    c  layer L's router on layer L-1's experts' input (a layer ahead)   45.9%     56.3%     64.2%     77.5%
    the ten experts the token before went to at that layer              34.1%

Replayed against a memory of some experts (bench/expert_use.py's: the most used elsewhere, the least recently
used replaced), for b, the best a layer ahead: of the experts that had to be read, the share among its best N,
the reads it would add for nothing as a share of them, and what an answer would gain by the rule above:

    experts in memory          read a token     best 6              best 8              best 10
     1,500   2 GB, 16 GB Mac      228.8      45% +9%   1.35x     59% +19%  1.24x     69% +34%  1.10x
     3,000   4 GB                 150.3      43% +11%  1.34x     57% +22%  1.39x     68% +40%  1.21x
     6,000   8 GB, 24 GB Mac       68.2      38% +15%  1.20x     52% +30%  1.30x     64% +55%  1.39x
    12,000  16 GB, 32 GB Mac       20.3      31% +23%  1.03x     44% +48%  1.05x     56% +88%  1.06x

At most 1.35 to 1.39 times, so by the rule it is not built. And that is the most it could be: it takes every
read asked for ahead to be done in time with a layer's notice; guess b costs a hyper-connection's read more at
every layer (13 MB of weights each, 0.6 GB a token beside the 5.5 a token reads now); and the disk's shares
are those with the draft head while the guesses were recorded a token at a time. slotstream, with this model
on a 48 GB Mac and 20-22 GB for it, measured 1.1 times for its lookahead.

Why so little where the disk matters most: with 2 GB of experts the disk is two thirds of the time, and a guess
only moves reads beside the GPU's work, it does not make them fewer. A wider guess holds more of the misses and
reads more for nothing (best 16: 83% held, 107% added), which a disk at its limit pays in full. The experts the
token before used hold none of the misses: they are in memory already.

What makes the reads fewer is more experts in memory: 229 read a token with 2.1 GB of them, 150 with 4.1, and
with the draft head 10 tok/s at 2 GB against 12.5 at 4 (above). On a 16 GB Mac the dense weights take 2 GB more
as arrays than in the file, which is that step: built next.

### The dense tensors of two types as the file's blocks (e36)

By type, the Q2_0 file's dense tensors (the n-gram table apart, which stays on the disk): 3.64 GB in the file,
5.49 as the loader held them. Of the difference 1.25 GB is in 91 Q3_K tensors (0.44 of it the embedding), 0.49
in 56 IQ4_XS ones and 0.11 in all the others: MLX can hold those two types only at 8 bits. Where experts are
read from the SSD they now stay the file's blocks (`dense.BlockLinear`, `BlockEmbedding`), 1.74 GB less, which
the store gives to the experts when it fits itself to what is loaded. A machine that holds every expert keeps
the arrays.

What it took: an `unpack32` for Q3_K (IQ4_XS had one, for the experts of other files); a kernel that unpacks a
run of weights once and multiplies a window's tokens by it, each token's sum taken in the same order whatever
the window (`blocks.window_matvec`); the delta net's output projection, whose inputs the loader reorders half
a block at a time, reorders its activations instead; the embedding takes a token's row of blocks and unpacks
that.

All of a pass's products of these types, the real model's 146 tensors, as blocks against as arrays
(`bench/experiments/e36_dense_as_blocks.py`, two runs): 3.8 against 4.7 ms for one token, 4.2-4.3 / 4.7-4.8
for two, 5.6-5.7 / 4.8 for four, 9.3-9.4 / 7.4-7.5 for eight. (e15 found a Q3_K projection no faster as blocks
and stopped there; that was one tensor and a kernel of another layout.) A chunk of 1,024 tokens: 310-312 ms as
arrays, 290-307 with the weights unpacked for the chunk and multiplied by MLX's matmul, 423-454 through the
experts' tiles with a layer as a stack of one expert. So a chunk unpacks, 63 MB at a time at most and let go
after: the peak of arrays while a 1,537-token prompt is read is the same either way, 11.4-11.5 GB as a 16 GB
machine.

In a pass (`bench/paged.py --machine-memory <GiB> --dense arrays|blocks`): four conversations of 200 tokens,
twice over, with the draft head, the file dropped from the page cache; first the arrays and then the blocks,
and at 24 and 32 GB again in the other order:

    machine   dense as   experts     answers, tok/s (mean)                  experts read a token
     16 GB    arrays      1.59 GB     6.4-7.7  (7.2)                             352-431
              blocks      3.33 GB     9.5-11.8 (10.9)                            193-211
     24 GB    arrays     10.18 GB    14.9-23.5 (19.2), then 20.3-25.4 (23.3)      40-80
              blocks     11.92 GB    17.5-28.2 (20.3), then 23.7-29.0 (26.4)      32-65
     32 GB    arrays     18.77 GB    21.1-40.1 (33.3), then 30.1-41.7 (36.5)      11-32
              blocks     20.51 GB    23.8-37.2 (33.2), then 26.2-40.4 (35.5)      10-34

Half again as fast as a 16 GB machine, where the experts in memory double and the reads halve. As a 24 GB one 6
and 13% in the two runs, which is less than the same form differs from one run to the other. Nothing as a
32 GB one: a tenth fewer reads and a millisecond more a pass. It is on wherever experts are read from the SSD:
one rule, and nowhere measurably slower. It fits the budgets measured before (with the draft head 9 tok/s and
257 reads a token at 2 GB, 12.5 and 175 at 4): a 16 GB machine went from under the first to near the second.

The server started as each machine (`bench/compare_servers.py --only "strata-mlx default" --engine-args
"--machine-memory 16"`, the file dropped from the page cache first, two starts, 256 tokens an answer):

    machine    experts in memory    answers        1,180-token prompt    up in
     16 GB          3.3 GB          11-12 tok/s      107-121 tok/s        19-24 s
     24 GB         11.9 GB          18-27            125-133              21-24 s
     32 GB         20.5 GB          23-38            109-145              21-26 s

As a 32 GB machine one answer of sixteen came at 9.7 tok/s, its prompt at half the usual speed too; two more
starts did not repeat it (their eight answers are in the range above). The 28-47 of the first version's table
is from another day; in turns today the two forms do not differ at that size. The process is as large as
before: what the blocks spare, the experts take.

Against llama.cpp's saved answers with 3.3 GB of experts in memory, top token equal a token at a time / four a
pass: after 1,537 tokens 64 / 64, after 3,161 tokens 64 / 64, and the same with the arrays; in float32, a token
at a time, 64 against its Metal answer and 63 against its CPU one, and the same with the arrays.

**A figure above was wrong**, found on the way. The 273-287 tok/s for the 1,537-token prompt on these machines
was measured with the model's file in the page cache, and a read past the cache is still answered from it when
the pages are there. The same script, 3.3 GB of experts in memory: 255-277 tok/s with the file in the page
cache, 149-162 with it dropped. A chunk of a thousand tokens uses nearly every expert of a layer, so every layer
is read whole: 30 GB from the disk for one chunk, 8 to 11 s. With the page cache dropped, 1,537 tokens as a 16,
24 and 32 GB machine: 146-171 tok/s with the dense tensors as arrays, 160-182 as blocks; the first version had
136-159. So the tiles made a prompt on these machines a tenth faster, not twice as fast, and reading one is
bound by the disk.

What was measured before building it:

- **The disk.** Expert-sized reads at random places that are not in the page cache: 1.9 GB/s from one thread,
  6.5 GB/s from eight or more, which is about 4,700 Q2_0 experts a second. This is a 1 TB internal SSD; the
  256-512 GB ones in 16 GB MacBooks are slower, and that has not been measured.
- **How spread out the routing is.** Over seven conversations (prose, code, reasoning, an edit, Chinese, a
  summary, SQL) a token uses 480 of the 24,576 experts, and they are not the same few: with the experts most used
  in the other six conversations resident and the least recently used one replaced at each miss,

      resident experts     share    memory    misses per answer token (range over conversations)
           1,500             6%     2.1 GB    226  (201-260)
           3,000            12%     4.1 GB    147  (120-174)
           6,000            24%     8.3 GB     66  (43-95)
          12,000            49%    16.6 GB     21  (15-40)

  A fixed set, never replaced, misses about twice as often.
- **What that leaves a 16 GB machine** (as estimated then; the table above is what was measured since). Some
  4 GB for experts was the guess; the dense weights turned out to be 5.6 GB as arrays, not the file's 3.6, so with
  the draft head it is nearer 2 GB, the first row. The guess that the draft head would not help was wrong too:
  it does, by a third to a half, because the wait at each layer is paid once for a window.

Settled: the target file is Q2_0, the one on disk (no space for the Coder or any other file), behind a memory
budget that one build serves 16, 24 and 32 GB machines with; it is measured here by capping the budget, reading
experts past the page cache. 24 and 32 GB machines are the easier case: the 6,000 and 12,000 rows.

## On an M5 Ultra with 96 GB: the open issues, and the two files not run before

Run by a second person on their own Mac: an M5 Ultra with 96 GB, macOS 27.0.1, MLX 0.32.3, llama.cpp 0.6.0 from
Homebrew (the comparisons above had 0.4.1). Every number in this section is from that one machine; the M4 Max has
not been run again with any of this.

### A short prompt after a rest: not the clock (e37, e38; issue 1)

The next step of issue 1 first, without a model (`bench/experiments/e37_idle_gpu.py`): after two seconds or more
of nothing an addition of two numbers takes 4-15 ms where it took 0.2, and 400 small products one after another
37-58 ms where they took 16. The CPU slows too: a loop of Python 6.4 ms one time after another, 14-23 after a
rest of half a second or more. So the GPU does come back slowly; but that is not what a prompt pays.

With the model (`bench/experiments/e38_idle_parts.py`, Q2_0), a piece of 64 tokens split into the CPU building
its graph and the GPU computing it, in ms:

    one after another                                          4 +  92
    after 2-8 s                                            15-24 + 223-379
    after 2-8 s, the GPU given work every 0.05-0.5 s        12-20 + 221-336   (8 MB of products, or 1-2 GB
                                                                                summed; 2 to 40% of the time)
    after 0.5-1 s, the CPU kept busy                            5 + 101-115
    after 2-8 s, the CPU kept busy                              8 + 223-306
    after 2-8 s, MLX's arrays wired                         16-21 + 150-163   but after 4-8 s the first thing
                                                                               on the GPU waits 250-320 ms
    after 2-8 s, wired, one addition every 1-2.5 s           5-20 + 103-161

Work during the rest did not help, however much: not the clock. Two things instead. The arrays MLX does not keep
resident are let go of by the GPU after about two seconds of nothing, and every piece after that computes at
less than half speed; resident (`mx.set_wired_limit`, which mlx-lm sets and this engine did not), they are not.
And after about four seconds of nothing the GPU sleeps, and the first thing given to it waits 250-320 ms; one
addition a second keeps it awake, at a cost too small to see. What is left, 60-80 ms after a rest of half a
second or more, is the CPU's: kept busy it goes, which a server should not do.

The server now does both where the model is held whole (`serve.py`): its arrays wired, and for `--keep-awake`
seconds after a request (600 by default, 0 for never) an addition a second. The four prompts of the comparison
and its 1,180-token one, as the server reads them with 8 s between requests (`bench/compare_servers.py --only
"strata-mlx default"`, Q2_0, two rounds):

                              27 tokens   31 tokens   49 tokens   340 tokens   1,180 tokens   answers
    before                    522-548     560-623     534-547     587-619      688-1,187 ms   97-166 tok/s
    wired, no additions       436-456     630-636     592-643     533-655      966-969 ms     134-187 tok/s
    wired and additions       124-145     136-146     152-156     276-282      507 ms         127-180 tok/s

Wired alone makes answers faster and prompts no faster; both, a short prompt is read four times as fast. Where
experts are read from the SSD the arrays are not wired, so that macOS can still move memory on a small Mac.

### All engines in one session (issue 2)

`bench/compare_servers.py`, each engine in turn and again in the opposite order, on this machine; tok/s:

                         answers (four prompts of 256 tokens)    reading 1,180 tokens
                         strata-mlx   --spec 1   llama.cpp        strata-mlx    --spec 1      llama.cpp
    Q2_0                 132-189      75-102     60-61            2,346-2,360   2,449-2,675   1,291-1,292
    IQ3_S                103-152      71-90      57-58            2,364         2,384-2,483   1,288-1,304
    IQ3_XXS              122-161      71-86      58-59            2,413-2,514   2,554-2,574   1,317-1,326
    IQ2_XS               118-166      72-102     60-61            2,500-2,501   2,551-2,561   1,346-1,348

The 27-token prompt: 136-137 ms here (102-103 with `--spec 1`), 248-971 by llama.cpp. LM Studio is not on this
machine.

### The floats kept for prompts, counted (issue 3)

They are kept only where the whole model and they fit (`dense.kept_floats`, `ScaledLinear.keep`, decided in
`load.py`); elsewhere they are made for each chunk and let go, as before they were kept. A machine that reads
experts from the SSD never keeps them: the experts have more use for the memory. `--check` says the same. The
IQ3_S file as a 64 GB machine (`bench/paged.py --machine-memory 64`): after the first prompt the arrays were 0.9
GB more than the experts and everything else, where issue 3 saw 3.6; the process at its peak 50.8 GB, which
leaves macOS 17.9 of a 64 GB machine's 68.7.

### A prompt as a small machine: the next layer read while one computes (issue 4)

Timed per layer, as a 16 GB machine with the 1,468-token prompt below: a layer's 0.7 GB read in 57-69 ms (this
SSD gives 14 GB/s in pieces of 1 MiB, `bench/ssd_reads.py`), then copied into MLX's arrays on the thread that
runs the model, 34-64 ms more, while the disk waited.

What is built (`paged.py`): for a chunk of 8,192 (token, expert) pairs or more, the next layer's experts that are
not in memory are read on another thread while this layer computes, straight into arrays made for them
beforehand (MLX lets their memory be written where it lies; nothing of MLX is called on that thread); the GPU then
joins them with the experts in memory, which are no longer read again. A layer read ahead is used whole even
where the chunk turns out to use fewer of its experts. The store keeps one layer's experts free for it (0.71 GB
with Q2_0, 1.36 with IQ3_S; `--check` counts it): the process at its peak was 0.48 GB larger than before, as a 16
GB machine.

The 1,468-token prompt (`bench/prompt_1468.txt`, the edit example of `spec_decode.py` five times) and the
340-token edit prompt, Q2_0, the file dropped from the page cache, tok/s:

                          1,468 tokens                    340 tokens          answers
                          before       now                before    now
    as a 16 GB machine    217-247      390-403            99-102    96-102 (83 once)  the same
    as a 24 GB machine    214-216      382-386            117-119   117-119    the same
    as a 32 GB machine    128-183      410-433            22-135    152-153    the same

Tried on the way and not kept: reading ahead for every chunk of more than 640 pairs (a chunk of 65 tokens whose
experts were all in memory then read 502 more a layer; and the 340-token prompt was a sixth slower as a 24 GB
machine); guessing per layer from the experts' profile whether it would go whole (the profile said every layer
would at 1,468 tokens, and five did not); copying the experts in memory into the read layer on the CPU (a second
a prompt as a 32 GB machine); reading through runs of up to 4 or 16 experts in memory to make fewer, longer reads
(twice the bytes, and slower).

### Float32 on an M5 (issue 5), and the two files not run before

`bench/experiments/e29_answer_quality.py` with the Q2_0 file: float32 with the draft head and the list wrote the
model's own answer at every token of the four, 256 of 256 each, and the same answer again; float16 at 252-256 of
256, where two tokens were nearly level, as on the M4 Max. Float32 93-128 tok/s, float16 134-169. (e29 and
`spec_decode.py --dtype float32` now turn MLX's reduced float32 off, as the server does: on this machine a
float32 product of 1024-cubes is off by 7.6e-4 of its largest value by default, 1.2e-6 so.)

IQ3_XXS and IQ2_XS load and run as they are. Against llama.cpp from the same file (`bench/compare_llama.py
--windows 1 4`, the 1,468-token prompt and 64 tokens after it), the top token the same at every step: IQ3_XXS 64
of 64 in one pass, a token at a time and four a pass, 63 with the engine a token a pass; IQ2_XS 64 of 64 every
way; IQ3_S 63 of 64 every way, all at the same step. Their speeds are in the table above.
