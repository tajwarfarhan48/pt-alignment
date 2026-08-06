# Hardware provenance for the cross-backend benchmark results

Raised because the two independent CPU sweeps (`results/sweep_cpu.csv` vs.
`results/sweep_cpu_aces.csv`) disagree by ~13-15% across every sweep point
(e.g. 154ms vs 177ms at n=10, 686s vs 803s at n=640) -- plausibly explained
by different CPU hardware, not noise, so hardware details need to be part
of the record, not just the timing numbers. Same principle applies to
GPU/CS-3: a shared, contended node produces different numbers than an
idle one, and that's a confound worth knowing about, not hiding.

## GPU (ACES, `results/gpu/sweep_gpu.csv`, job 2025807)

- **Node**: `ac045`
- **GPU**: NVIDIA H100 PCIe, compute capability 9.0, 81559 MiB
- **Sharing**: `ac045` has 8 H100s total; our job requested/held exactly 1
  (`--gres=gpu:h100:1`), which SLURM allocates exclusively -- no other job
  could have used *our* GPU. However, the node itself was **not exclusive
  to us**: at least 3 other users (`u.dr62133`, `u.yk3479+`, `u.ew1261+`)
  had jobs running on `ac045`'s other GPUs during our run window
  (2026-08-06 10:34-11:03), confirmed via
  `sacct -a --nodelist=ac045 --starttime=... --endtime=...`. Those jobs
  were presumably using the other 7 H100s, not competing for compute, but
  potentially sharing host memory bandwidth / PCIe root complex / CPU
  cores for host-side work with our job. Not quantified how much this
  actually affected our numbers -- flagging as a known confound, not a
  correction.

## CPU, ACES run (`results/sweep_cpu_aces.csv`, job 2025826)

- **Node**: `ac093`
- **CPU**: Intel(R) Xeon(R) Platinum 8468, 96 cores, base 3.1GHz / max 3.8GHz
- **Sharing**: `sweep_cpu.cpp` is single-threaded (1 core requested,
  `--cpus-per-task=1`); the node's other 95 cores were running other users'
  jobs (`u.sl3510+`, `u.lr2868+`, `u.ns2138+` confirmed via the same sacct
  query) during our window. Memory-bandwidth/NUMA contention from those
  neighbors is possible but not measured.

## CPU, canonical run (`results/sweep_cpu.csv`, from Neocortex/Cerebras cloud host)

**Not yet documented.** This is the CSV currently treated as canonical in
`results/README.md`'s schema table, and it's ~13-15% faster than the ACES
run across every sweep point -- exactly the kind of gap that needs a CPU
model/clock speed/sharing-status explanation to be a fair comparison point,
not just a number. Whoever is driving the Neocortex-side session should
capture: CPU model (e.g. `lscpu` or equivalent), core count, clock speed,
and whether that host was shared with other workloads during the run.

## Cerebras CS-3 (`cs3/milestone2_allpairs/sweep_cerebras.csv`)

**Partially documented, and what exists isn't quite the right thing.** The
committed `wsjob-*-cluster-details.json` files describe the **appliance
coordinator task** (`taskType: CRD`, node `inf008-es-sr04`, 72GB memory,
24 CPU millicores) -- that's the host process managing the job, not the
WSE-3 wafer itself. It doesn't tell us:
- Whether the wafer (or the specific PE region used) was exclusively ours
  during the run, or time-shared with another queued job
- Any wafer-level utilization/occupancy figure
- Which physical CS-3 system / fabric coordinates were used, if that
  varies between the `wsjob-*` runs (there are 5 different job IDs
  committed, suggesting multiple submission attempts -- worth clarifying
  whether they're the same hardware allocation or different ones)

Given the CS-3 sweep's own results/README.md note that ~99% of measured
time is I/O-bound (network round-trip per pair), sharing status of the
wafer itself may matter less than network path variability -- but that's
a guess, not a measured fact, and should be replaced with a real answer
from whoever has Neocortex queue visibility.

## Action items

1. Neocortex-side session: capture CPU host specs for `results/sweep_cpu.csv`
   and wafer exclusivity/utilization for the CS-3 sweep, add to this file.
2. Consider whether the ~13-15% CPU discrepancy is worth reconciling (e.g.
   re-running both on directly comparable hardware) or is fine to report
   as-is with this hardware note attached -- a call for whoever's
   presenting these results, not something to silently resolve either way.
