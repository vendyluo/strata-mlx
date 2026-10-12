# Open issues

Problems that are known and not solved. What was tried and measured on the way to the current state is in
`ROADMAP.md`; this file holds only what is still wrong.

Issues 1, 2 and 3 were solved, and issue 5 run, on an M5 Ultra with 96 GB: what was found and what changed is
in `ROADMAP.md`, "On an M5 Ultra with 96 GB". None of it has been run on the M4 Max again.

## 4. Where experts are read from the SSD, a prompt's chunk still stops at every layer

[Issue 4 on GitHub](https://github.com/yibie/strata-mlx/issues/4)

**What was done.** For a chunk of 8,192 (token, expert) pairs or more the next layer's experts are read while a
layer computes, straight into the arrays the kernels take, and those in memory are not read again (`ROADMAP.md`,
"On an M5 Ultra with 96 GB"). The 1,468-token prompt, Q2_0, on an M5 Ultra: 390-403 tok/s as a 16 GB machine
(217-247 before), 382-386 as 24 GB (214-216), 410-433 as 32 GB (128-183). A 340-token prompt and answers as before.

**What is left.**

- The first layer of each chunk is read when it asks: nothing computes before it.
- A layer whose chunk uses fewer than 384 of its experts and was not read ahead reads them one by one
  (`ExpertStore.get`): with the 1,468-token prompt, five layers of 48, 0.74 s of 5.7 before the reads ahead.
- A chunk of fewer than 8,192 pairs reads each layer when it asks, as before: the 340-token prompt as a 16 GB
  machine reads at about 100 tok/s.
- Before the reads ahead, 2.6 s of the 1,468-token prompt's 5.7 as a 16 GB machine on the M5 Ultra were neither
  reading nor waiting for the disk, where the whole model in memory reads it in 0.6: the experts' kernels on
  arrays that are never unpacked, and the stop for the router at every layer. Not looked into.
- On the M4 Max, whose disk gave 6.5 GB/s and whose GPU computes a chunk about twice as slowly, none of this has
  been measured.

## 6. On a real 32 GB Mac the default leaves macOS too little memory, and it swaps

[Issue 6 on GitHub](https://github.com/yibie/strata-mlx/issues/6)

**What is seen.** On a MacBook Air M5 with 32 GB (macOS 27.0.1, MLX 0.32.3, commit `1d75ad4`, Q2_0 with the
draft head, about 2 GB of other programs open) the engine by default kept 20.5 GB of experts in memory, and the
machine swapped. With fewer experts in memory it was about twice as fast. Measured by @guoggg
([comment on issue 3](https://github.com/yibie/strata-mlx/issues/3#issuecomment-6110466402)) with
`bench/paged.py`, four prompts by two rounds of 200 tokens, and `--read` of a 1,180-token prompt:

| experts in memory | draft head tok/s (mean) | one token a pass (mean) | read 1,180 tokens, tok/s | arrays at peak | swap peak | swapped out during run |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 GB (`--budget 14`) | 20.9-24.8 (23.6) | 13.6 | 152, 155 | 22.4 GB | not recorded | not recorded |
| 16 GB | 23.4-26.8 (25.3) | 13.8 | 95, 119 | 24.5 GB | 2.8 GB | 1.7 GB |
| 18 GB | 12.0-24.4 (15.4) | 9.2 | 89, 110 | 26.5 GB | 5.1 GB | 39 GB |
| 20.5 GB (the default then) | 5.8-19.2 (11.9) | 7.1 | 49, 136 | 29.1 GB | 6.9 GB | 106 GB |

Each row is one run; the prompt read is the reporter's own text, not the one used here.

The 16 GB row again with every other program closed
([comment](https://github.com/yibie/strata-mlx/issues/6#issuecomment-6115147674)): answers at 24.3-28.8
(27.1), one token a pass 14.9, the prompt at 108 and 157, arrays at the peak 24.5 GB, 0.39 GB swapped out during
the run. Before it `vm_stat` gave 23.65 GB free and macOS with its background items 6.1 GB (wired, active and
compressed), 3.6 GB more inactive.

**What it says.** The rule in `strata_mlx/paged.py` counts the process rightly: in all four rows the arrays
peak at the experts' budget plus 8.4 to 8.6 GB. What was too little is what it left to macOS and other
programs, `SYSTEM`, 4.5 GB: with the default the machine swapped 106 GB out during the run, and with about 8 GB
left it hardly swapped. Of those 8 GB about 6 are macOS itself and 2 were the other programs.

**Why it was not seen here.** The speeds given for 16, 24 and 32 GB machines are from the 128 GB M4 Max with
the experts' budget limited. Nothing else is short of memory there, so it never swaps. The README's 23-38 tok/s
"as 32 GB" is such a figure; the real machine's default gave 5.8-19.2.

**What was done.** `SYSTEM` is 8.5 GB, which gave a 32 GB machine's experts 16.2 GB: the row that was
fastest. Since issue 4 one layer's experts more are kept free for the layer read ahead (0.71 GB with Q2_0), so it
gives them 15.5. By the same rule a 16 GB Mac is left the least, 0.5 GB of Q2_0's experts (it kept 9% of them; `--check`
calls that too small), a 48 GB Mac keeps
96% of Q2_0's experts where it held the file whole, and a 64 GB Mac 93% of IQ3_S's (98% and 95% before one layer
was kept free for the reads ahead).

**Not known.** Whether a 16, 24, 36 or 48 GB Mac needs the same amount, or less, or the same share: this is one
machine, one run a row. A 16 GB Mac may well do with less than 8.5 GB and is cut to the least here on no
evidence of its own; `--expert-memory` sets the experts' share by hand. The speeds given for 16, 24 and 32 GB machines were
measured with 3.3, 11.9 and 20.5 GB of experts and not again since.

**The next step.** Any run on a real 16, 24 or 48 GB Mac, and the 32 GB machine with the new default.
