#!/bin/bash
#SBATCH --job-name=msa-sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/sweep_%j.out
#SBATCH --mem=32G

# GPU leg of the cross-backend (CPU / GPU / Cerebras CS-3) sweep benchmark.
# See results/README.md for full methodology. Large job: n=640 alone is
# 204,480 pairs; the full sweep is 272,415 pairwise alignments total, hence
# the generous 3hr budget on gpu (not gpu_debug, which caps at 2hr and is
# a much smaller/more contested pool of nodes).
#
# NOT --exclusive: tried that, but it requires an entirely free node rather
# than just one free GPU, and the scheduler ETA came back ~08:34 the next
# day -- too slow given the time budget. Trading isolation for turnaround:
# the node may still be shared with other users (as the earlier run on
# ac045 was, confirmed via sacct -- see results/HARDWARE.md), but we at
# least capture live utilization throughout the run instead of flying
# blind, so sharing is visible in the data rather than just inferred after
# the fact.

set -euo pipefail
mkdir -p logs results/gpu
cd "$SLURM_SUBMIT_DIR"

module load CUDA/12.6.0
make

echo "=== node/exclusivity ==="
echo "node: $SLURM_JOB_NODELIST"
scontrol show job "$SLURM_JOB_ID" | grep -i "Shared\|OverSubscribe" || true

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# Live GPU utilization/power/memory sampled every 5s for the whole run,
# so we have more than just static specs -- actual load during the sweep.
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,temperature.gpu,memory.used,memory.total \
    --format=csv -l 5 > results/gpu/gpu_utilization.csv &
MONITOR_PID=$!
trap 'kill $MONITOR_PID 2>/dev/null || true' EXIT

echo "=== sweep: n=10,20,40,80,160,320,640 on test/sweep_dataset.fasta ==="
./bin/msa_align --fasta test/sweep_dataset.fasta --sweep 10,20,40,80,160,320,640 \
    --sweep-out results/gpu/sweep_gpu.csv

kill $MONITOR_PID 2>/dev/null || true
trap - EXIT

echo "=== results/gpu/sweep_gpu.csv ==="
cat results/gpu/sweep_gpu.csv
echo "=== results/gpu/gpu_utilization.csv (first/last 3 lines) ==="
head -4 results/gpu/gpu_utilization.csv
tail -3 results/gpu/gpu_utilization.csv
