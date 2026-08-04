#!/bin/bash
#SBATCH --job-name=msa-bench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --mem=32G

# Full PT-vs-nonPT benchmark sweep on an H100 (see the `gpu` partition, up to
# 2-day time limit -- we request 1hr, plenty for the DP sizes here).
#
# Submit with:  sbatch jobs/benchmark_run.slurm

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

module load CUDA/12.6.0
make

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

echo "=== verification (all modes) ==="
./bin/msa_align

echo "=== benchmark sweep: length,nonpt_ms,pt_ms,speedup ==="
./bin/msa_align --benchmark > results_$(date +%Y%m%d_%H%M%S).csv
cat results_*.csv
