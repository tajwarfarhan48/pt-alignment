# gpu-msa

GPU-accelerated multiple sequence alignment for the ByteBoost GGG workshop,
targeting TAMU ACES (`gpu` / `gpu_debug` partitions — H100 and A30 nodes).

Approach: progressive alignment. All-pairs pairwise alignment on the GPU →
guide tree → progressive profile merge → global MSA. Background reading:
Gupta, Stuart & Owens, *"A Study of Persistent Threads Style GPU Programming
for GPGPU Workloads"* (`ByteBoost GGG/`), which this project's kernel design
directly follows for the wavefront synchronization strategy.

**Picking this up on different hardware (e.g. porting to Neocortex/Cerebras
CS-3) or after a long gap? Read `HANDOFF.md` first** — it covers what
transfers (the algorithm) vs. what doesn't (the CUDA code) and what's left
to do.

## What's implemented (Day 1)

A single generalized anti-diagonal wavefront DP kernel, parameterized by
alignment mode:

- **global** (Needleman-Wunsch)
- **local** (Smith-Waterman)
- **semiglobal** ("glocal", free end-gaps — this is the mode Day 2's
  profile-profile merge step will use)

All three modes share one recurrence (`compute_cell` in `src/gpu_align.cu`);
they only differ in border initialization and where traceback starts. This
mirrors the observation that NW/SW/semiglobal are the same DP with different
boundary conditions, not three separate algorithms.

Two GPU implementations of that kernel are built side by side, on purpose —
this reproduces the PT-vs-nonPT comparison methodology from the paper's
Global Synchronization use case (§3.4), applied to real alignment DP instead
of a synthetic microbenchmark:

- **nonPT** (`align_gpu_nonpt`): one kernel launch per anti-diagonal. Simple,
  correct, pays kernel-launch overhead at every diagonal boundary.
- **PT** (`align_gpu_pt`): a single "maximal launch" (only as many blocks as
  can be simultaneously resident — queried via
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor`) of persistent thread
  groups. Threads loop over all diagonals inside the kernel; `grid.sync()`
  (cooperative groups) replaces the kernel relaunch as the inter-diagonal
  barrier.

A CPU reference (`src/cpu_reference.cpp`) uses the identical recurrence and
the identical traceback code (`src/traceback.cpp`, shared by all three
paths) so GPU output is checked cell-for-cell equivalent, not just
score-equivalent.

## Dataset

`test/silva_sample.fasta` — 10 real 16S/18S rRNA sequences (901–3617 bp),
taxonomically diverse (archaea, bacteria, eukaryotic chloroplasts), pulled
from ACES's shared SILVA 138 database at `/scratch/data/bio/silva/138` via
`module load MMseqs2-GPU/18-8cc5c-Linux64` + `mmseqs convert2fasta`, then
subsampled to span the real length distribution (min/median/max of a 2000-record
sample: 42–1464–3617 bp — the length-42 outlier was a parsing artifact and
was excluded). U was converted to T for consistency with the DNA scoring model.

The length spread is intentional: it's real data exercising exactly the kind
of variable-length, irregular workload the Load-Balancing use case (§3.2 of
the PT paper) is about, rather than uniform synthetic sequences.

ACES also has `/scratch/data/bio/pfam` (Pfam-A seed alignments, MMseqs2 DB
format) if a curated real MSA reference (for accuracy comparison, not just
timing) is wanted later — not pulled in yet since it needs the same
`mmseqs convert2fasta` extraction step and Day 1 is timing/correctness-focused
on synthetic + SILVA data.

Try it:
```
./bin/msa_align --fasta test/silva_sample.fasta --idxA 0 --idxB 9 --mode global --show
```

## Build

```
module load CUDA/12.6.0
make
```

Builds a fat binary covering both ACES GPU architectures
(`sm_80` = A30, `sm_90` = H100), so it runs unmodified on either.

## Run

```
./bin/msa_align                                   # built-in smoke test, all 3 modes, verifies CPU==nonPT==PT
./bin/msa_align --seqA ACGT... --seqB ACGT... --mode local --show
./bin/msa_align --benchmark                        # PT vs nonPT timing sweep, prints CSV
```

On ACES, submit rather than running on the login node (no GPU there):

```
sbatch jobs/debug_run.slurm        # ~1 min smoke test on gpu_debug (A30 by default)
sbatch jobs/benchmark_run.slurm    # full PT vs nonPT sweep on an H100
```

## Day 1 results

Correctness: verified on both A30 (`gpu_debug`, job 2019141) and H100 (`gpu`,
job 2021937) — CPU, nonPT, and PT agree cell-for-cell on all three modes.

Performance (`--benchmark`, global mode, random DNA, `nonpt_ms/pt_ms`):

| length | A30 speedup | H100 speedup |
|-------:|------------:|-------------:|
|    100 |       1.25x |         1.29x |
|    500 |       1.20x |         1.33x |
|   1000 |       1.10x |         1.35x |
|   2000 |       0.99x |         1.22x |
|   5000 |       0.93x |         0.87x |
|  10000 |       0.91x |         0.75x |

PT wins for short-to-medium sequences, then loses — worse on H100 than A30 at
the largest size. Root cause: the PT kernel launches once with a **fixed**
grid sized by `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (maximal
launch), so on a wide diagonal near the middle of a large DP matrix, threads
must loop internally over multiple cells. nonPT instead re-sizes its grid
*every single launch* to match that diagonal's exact width. H100 has far more
SMs than A30, so its fixed PT grid is proportionally an even smaller share of
a wide diagonal's real parallelism — hence the larger H100 slowdown at
10,000bp. This is a direct, concrete instance of the PT paper's own
conclusion (§4.2): PT is not an unconditional win, and over-subscription
(nonPT's traditional weakness) can act as automatic load balancing that a
fixed-size PT launch doesn't get for free.

## Day 2 plan

1. ✅ **All-pairs batch**: run the pairwise kernel over every sequence pair in
   the input set — independent, variable-length DP problems, i.e. the
   Load-Balancing/Irregular-Parallelism use case (§3.2) rather than the
   Global-Sync one Day 1 focused on.
2. ✅ **Guide tree**: UPGMA over the pairwise scores from step 1 (`src/guide_tree.cpp`,
   host-side). Run with:
   ```
   ./bin/msa_align --fasta test/silva_sample.fasta --guidetree
   ```
   Prints the score matrix and a Newick tree. Note: the distance used for
   clustering is `-score`, a simplified proxy for divergence, not a
   calibrated evolutionary distance — fine for ordering merges, not for
   publishing branch lengths.
3. ✅ **Progressive merge** (`src/progressive.cpp`): walk the guide tree
   bottom-up, aligning profile-vs-profile at each internal node, producing the
   final MSA. Run the whole pipeline end-to-end with:
   ```
   ./bin/msa_align --fasta test/silva_sample.fasta --msa
   ```
   Scope decision: this step generalizes the pairwise DP recurrence to
   **sum-of-pairs column scoring** (every cross-pair between two profile
   columns, gap cost scaled by `depthA*depthB`) and runs on the **host**, not
   as a new GPU kernel. Reasoning: profile depth here is just the sequence
   count (small at workshop scale), so this step is not the bottleneck --
   the O(n^2) all-pairs pairwise stage (step 1, GPU-parallel) dominates the
   runtime. This mirrors how real progressive aligners are usually structured:
   parallelize the pairwise distance computation, keep the inherently
   sequential tree-guided merge simple. An early "align single representative
   sequences per cluster, then propagate the gap pattern" shortcut was
   considered and rejected: it has a real ordering ambiguity whenever a
   subtree already contains internal gaps from an earlier merge, which a
   proper profile-profile DP resolves automatically by construction.

### Stretch goals (only if time remains)

- Affine gaps (Gotoh) — separate gap-open/gap-extend costs instead of the
  current flat linear penalty; same DP shape, 3 matrices instead of 1.
- Linear-space traceback (Hirschberg/Myers-Miller) — current implementation
  copies the full `H`/`TB` matrices back to the host, which is O(m·n)
  memory; fine at workshop scale, not for long sequences.

## Layout

```
include/align.h        shared types (ScoreParams, AlignMode, AlignResult) + prototypes
src/cpu_reference.cpp  CPU reference DP
src/traceback.cpp      traceback shared by CPU + both GPU paths
src/gpu_align.cu       kernels (compute_cell, nonPT step kernel, PT persistent kernel) + host wrappers
src/main.cu            CLI: verification runs + benchmark sweep
jobs/                  SLURM scripts for ACES gpu_debug / gpu partitions
```
