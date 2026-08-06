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

- **Host**: `inf004-us-sr17` (Cerebras cloud usernode)
- **CPU**: AMD EPYC 9354P, 32 cores / 64 threads (1 socket, 2 threads/core),
  3250 MHz
- **Sharing**: **not dedicated** -- this is a shared multi-tenant usernode.
  At the time of the run, at least 7 other users were logged in concurrently
  (`haochenl-cloud`, `prathyusht-cloud`, `prabhuv-cloud`, `horaciog-cloud`,
  `vishali-cloud`, `oyinkansolaa-cloud`, `peijingl-cloud`, via `who`/`w`),
  and `ps aux` showed active competing workloads from other users, including
  another Cerebras inference job (`hosseing-cloud` running
  `soda_inference.py` against a live CSX session). `sweep_cpu.cpp` is
  single-threaded like the ACES CPU leg, so it used 1 of 64 threads; the
  other 63 were available to (and in active use by) everyone else on the
  box. Same class of confound as the ACES CPU run, not a cleaner
  measurement -- the two CPU CSVs are both "shared host, 1 thread used,"
  not "dedicated vs. shared."
- This plausibly explains at least part of the ~13-15% gap vs. the ACES
  CPU run in the other direction than you might guess: EPYC 9354P's
  3.25GHz base is *higher* than the Xeon Platinum 8468's 3.1GHz base,
  which is consistent with this host being faster despite also being
  shared -- clock speed and microarchitecture differences are a more
  likely explanation than sharing status, since both runs were on shared
  hosts with comparable contention profiles (single-threaded job, several
  other tenants active).

## Cerebras CS-3 (`cs3/milestone2_allpairs/sweep_cerebras.csv`)

- **Physical system**: `wse127-cs-sy01` -- confirmed via `csctl get job
  <id> -oyaml`'s `status.systems` field to be the *only* physical CS-3 on
  this entire cluster (every `csctl get system`/`csctl get cluster` check
  throughout this session, across several hours, ever showed exactly one
  system row). Of the 5 committed `wsjob-*-cluster-details.json` files,
  only 2 (`wsjob-hwarpyc9stjqejwcvbk9w4`, `wsjob-vazpky3hptlgvjrrnjul5o`)
  are worker (`taskType: WRK`) tasks that actually touch the wafer
  (`wseIds: [0]`); the other 3 are compiler coordinator tasks
  (`taskType: CRD`, node `inf008-es-sr04`, no wafer contact at all) from
  separate compile-only attempts. Both wafer-touching jobs ran on the same
  `wse127-cs-sy01` -- there is no second system this cluster could have
  scheduled onto.
- **Exclusivity: yes, confirmed empirically, not assumed.** Every `csctl
  get jobs` check made throughout this session (dozens, over several
  hours, including while explicitly watching the queue for this sweep)
  showed at most **one** `execute`-type job in `RUNNING` phase at any
  time, with every other job -- including several from other users --
  sitting `QUEUED` until the running one finished or was cancelled. The
  scheduler serializes wafer access at the execute-job level; there is no
  time-multiplexing across concurrent jobs on this cluster. So whenever
  `sweep_cerebras.csv`'s numbers were being measured, the job had
  exclusive use of the wafer -- this is the one part of the whole
  cross-backend comparison that did **not** have to trade off isolation
  for time budget, because the scheduler enforces it by construction.
- **Allocation quantity: the entire wafer, always -- no sub-wafer
  granularity exists on this cluster.** Nothing observed this session
  (job queue behavior, `wseInUse` flipping true/false as a single unit,
  one `execute` job at a time) suggests the scheduler can split the chip
  so two jobs hold different regions concurrently. Allocation is
  all-or-nothing at the whole-system level -- so "how much was allocated
  to us" has a simple, confirmed answer: 100% of `wse127-cs-sy01`, for
  the duration of our `execute` job.
- **Wafer-level utilization/occupancy API: does not exist, checked
  thoroughly, not guessed.** Two things checked, not one:
  1. `csctl get job <id> -oyaml --debug=2` (max verbosity) exposes job
     lifecycle timestamps, the allocated system name, and one networking
     counter (`cerebras/act-spine-load`, which read `0 times from 0 total
     AX->WSE connection(s)` for our jobs and isn't a utilization metric)
     -- no PE-occupancy or FLOP-utilization figure anywhere.
  2. The SDK ships real instrumentation/trace tooling
     (`cerebras.sdk.sdk_debug_instr_trace.InstrTrace`,
     `cerebras.sdk.sdk_debug_wavelet_trace.WaveletTrace`) that looks like
     it could answer this -- but both constructors require a `simfab_log`
     argument (`InstrTrace(elf_dir, simfab_log)`,
     `WaveletTrace(simfab_log)`): the **simulator's** log file (same
     `simfab_traces/` directory that grew to 57GB during an earlier
     simulator run in this session). Neither tool takes anything
     equivalent for a real-hardware run -- the SDK's own PE-activity
     tracing is simulator-only in this version, not just absent from
     `csctl`. `sdk_debug_pe_symbol_dump.PESymbolDump` also exists but only
     inspects the compiled ELF's static symbol table (addresses/sizes),
     not runtime activity.
  If real-hardware PE-utilization telemetry exists in some other tool or
  a newer SDK version, it wasn't reachable from this session's install.
- **What we can say instead (computed, not measured)**: our kernel uses a
  1x1000 vertical line of PEs (see `cs3/milestone2_allpairs/layout.csl`)
  against a full addressable fabric of 762x1172 = 893,064 PEs (the
  `--fabric-dims` passed at compile time, matching the real WSE-3's
  addressable grid) -- so our own PE-grid occupancy was analytically
  ~0.11% of the fabric we were allocated, not measured but exactly
  derivable from the compile parameters. That's a statement about how
  much of the *allocation* our design used, not about contention from
  other jobs (there was none, per the exclusivity point above) -- and
  it's a compile-time fact, not a runtime measurement, because no tool
  available this session can produce the latter for real hardware.

## Action items

1. ~~Neocortex-side session: capture CPU host specs...~~ Done, see above.
2. Consider whether the ~13-15% CPU discrepancy is worth reconciling (e.g.
   re-running both on directly comparable hardware) or is fine to report
   as-is with this hardware note attached -- a call for whoever's
   presenting these results, not something to silently resolve either way.
   Given both hosts turned out to be shared with comparable contention
   profiles, the gap is more likely CPU model/clock than isolation --
   reconciling would mean matching hardware, not chasing exclusivity.
