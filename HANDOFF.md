# Handoff notes

Written for a fresh agent/person picking this up cold — on ACES to continue
the CUDA work, or on **Neocortex (Cerebras CS-3)** to reimplement the same
project on fundamentally different hardware. Read this before `README.md`'s
technical detail; this file is about *what to do next and why*, not *how the
current code works* (that's in `README.md` and the inline comments).

## What this project is

GPU-accelerated multiple sequence alignment (MSA) via progressive alignment:
all-pairs pairwise alignment → UPGMA guide tree → progressive profile merge.
Built for a 2-day GPU-programming workshop (ByteBoost GGG), on TAMU ACES
(NVIDIA A30 / H100). Background reading that shaped the design: Gupta,
Stuart & Owens, *"A Study of Persistent Threads Style GPU Programming for
GPGPU Workloads"* (in `ByteBoost GGG/` at the repo root's sibling directory,
not committed here — ask whoever ran the workshop for the PDF if it's not
already alongside this repo).

## Status as of this handoff

Everything below is implemented, compiles cleanly with CUDA 12.6 on ACES,
and has been run (not just compiled) on real A30 and H100 nodes:

- Generalized wavefront DP kernel (global/local/semiglobal in one recurrence)
- Two GPU implementations for comparison: nonPT (kernel-launch-per-diagonal)
  and PT (single persistent-kernel launch, `grid.sync()` between diagonals)
- CPU reference + shared traceback, all three paths verified to agree
  cell-for-cell
- Real dataset: 10 SILVA rRNA sequences (901-3617bp), `test/silva_sample.fasta`
- All-pairs GPU scoring, UPGMA guide tree, sum-of-pairs progressive merge —
  i.e. a working end-to-end MSA pipeline (`--msa` flag)

**Not yet done** (see README's "stretch goals"): affine gaps (Gotoh),
linear-space traceback (Hirschberg/Myers-Miller) for sequences long enough
that O(m·n) matrices don't fit in memory, and true concurrent batching of
the all-pairs stage (currently sequential per-pair GPU calls, not streamed).

**Key empirical finding** (details + numbers in README): PT beats nonPT for
short-to-medium sequences (up to ~1.3x on both A30 and H100), then *loses*
past ~2000-5000bp (down to 0.75x on H100 at 10,000bp). Root cause: PT's
"maximal launch" grid size is fixed once via occupancy query and never
adapts per-diagonal, while nonPT re-sizes its grid every single launch to
match that diagonal's exact width — over-subscription turns out to be a
form of automatic load balancing that a fixed PT launch doesn't get for
free. This is a real, hardware-measured instance of the source paper's own
conclusion that PT is not an unconditional win.

## Porting to Neocortex / Cerebras CS-3 — read this before writing any code there

**The CUDA code in this repo will not run on CS-3, and should not be
ported line-by-line.** CS-3 is not a GPU — there's no SIMT model, no
warps/SMs, no `grid.sync()`, no CUDA toolchain at all. It's a wafer-scale
dataflow chip (Cerebras WSE-3): a 2D fabric of ~900,000 independent
processing elements (PEs), each with its own small local memory, connected
by a fast on-chip mesh network, programmed in **CSL** (Cerebras SDK's own
language) using a fundamentally different execution model — you write
per-PE kernels and explicit inter-PE data movement over the fabric, not a
grid of threads pulling from global memory.

**What *does* transfer:**
- The **algorithm design**: progressive alignment (all-pairs → guide tree →
  profile merge) is hardware-agnostic; keep that pipeline structure.
- The **anti-diagonal wavefront dependency structure** of the DP recurrence
  (`compute_cell`'s diag/up/left dependencies) — this is actually a
  *better* conceptual fit for CS-3 than for a GPU. Wavefront/systolic
  algorithms are a well-known strong use case for wafer-scale dataflow
  architectures: map each row (or a block of rows) of the DP matrix to a
  physical PE, and let anti-diagonal values flow between neighboring PEs
  over the fabric as they're produced — no equivalent of `grid.sync()`
  needed at all, because the fabric's point-to-point communication *is*
  the synchronization, only between PEs that actually have a data
  dependency, not a global barrier across the whole chip. This should be
  the starting design point on CS-3, not a translation of the CUDA kernels.
- The **test dataset** (`test/silva_sample.fasta`) and the **CPU reference +
  traceback logic** (`src/cpu_reference.cpp`, `src/traceback.cpp`) — plain
  C++, no CUDA, useful for verifying a CS-3 implementation's correctness the
  same way it verified the GPU one.
- The **PT-vs-nonPT lesson**: on CS-3 there's no equivalent distinction
  (no kernel-relaunch-overhead vs persistent-thread tradeoff — the
  programming model doesn't have kernels in the CUDA sense), but the
  underlying question — "does a fixed resource allocation adapt well to a
  workload whose shape changes over time (the diamond-shaped diagonal
  widths)?" — will likely resurface in a CS-3-specific form (e.g. how many
  PEs to dedicate to a given alignment, and whether that's decided once or
  adapts as the wavefront widens/narrows). Worth actively looking for.

**What does not transfer:** `src/gpu_align.cu` in its entirety (CUDA
kernels, `cooperative_groups`, `cudaLaunchCooperativeKernel`, the Makefile's
`nvcc`/`-gencode` flags, the SLURM GPU job scripts). Expect to write CSL
kernels from scratch and a different job submission setup (Neocortex uses
its own scheduler/SDK workflow, not plain `sbatch --gres=gpu:...`).

**Suggested first step on CS-3**: don't start from the pairwise kernel.
Start smaller — get a single-PE-row-per-sequence-position wavefront running
correctly for one small pairwise alignment (e.g. reuse the 9x9 worked
example from this project's design discussions), verify against
`align_cpu()`, *then* scale up to the full pipeline. Same incremental order
this project followed (Day 1: one kernel design + verification; Day 2:
batch + tree + merge), just with CS-3-appropriate building blocks.

## Repo layout

See `README.md` for the full file-by-file breakdown and build/run
instructions. Quick orientation: `include/align.h` has the shared types;
`src/gpu_align.cu` is 100% CUDA-specific (the part that needs a full
rewrite for CS-3); `src/cpu_reference.cpp`, `src/traceback.cpp`,
`src/guide_tree.cpp`, `src/progressive.cpp` are plain C++ with no GPU
dependency and should port to any platform essentially unchanged.
