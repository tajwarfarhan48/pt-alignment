#!/bin/bash
#SBATCH --job-name=msa-sweep-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/sweep_cpu_%j.out
#SBATCH --mem=4G

# CPU leg of the cross-backend (CPU / GPU / Cerebras CS-3) sweep benchmark,
# ACES side. See results/README.md for methodology. sweep_cpu.cpp is
# single-threaded (plain sequential double for-loop over pairs, no OpenMP),
# so this measures single-core performance -- core count on the node is
# irrelevant to the work itself.
#
# NOT --exclusive: tried that, but it needs an entirely free node and the
# scheduler ETA came back too slow given the time budget. The node may
# still be shared with other users (as ac093 was on the earlier run,
# confirmed via sacct -- see results/HARDWARE.md), but live per-core
# utilization is captured throughout so sharing shows up in the data.
#
# Writes to sweep_cpu_aces.csv, NOT sweep_cpu.csv -- that path is the
# canonical result merged from the Neocortex-side run; overwriting it here
# would silently clobber their data.

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

module load GCC/13.2.0
g++ -O3 -std=c++17 -Iinclude \
    results/sweep_cpu.cpp src/cpu_reference.cpp src/traceback.cpp src/fasta.cpp \
    -o bin/sweep_cpu

echo "=== node/exclusivity ==="
echo "node: $SLURM_JOB_NODELIST"
scontrol show job "$SLURM_JOB_ID" | grep -i "Shared\|OverSubscribe" || true

echo "=== CPU info ==="
lscpu | grep -E "Model name|CPU\(s\)|MHz"

# Live per-core utilization for the whole run (mpstat, sysstat package,
# confirmed present on ACES -- no install needed).
mpstat -P ALL 5 > results/sweep_cpu_aces_utilization.log &
MONITOR_PID=$!
trap 'kill $MONITOR_PID 2>/dev/null || true' EXIT

echo "=== sweep: n=10,20,40,80,160,320,640 on test/sweep_dataset.fasta ==="
./bin/sweep_cpu test/sweep_dataset.fasta 10,20,40,80,160,320,640 results/sweep_cpu_aces.csv

kill $MONITOR_PID 2>/dev/null || true
trap - EXIT

echo "=== results/sweep_cpu_aces.csv ==="
cat results/sweep_cpu_aces.csv
echo "=== results/sweep_cpu_aces_utilization.log (head) ==="
head -10 results/sweep_cpu_aces_utilization.log
