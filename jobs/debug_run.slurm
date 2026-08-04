#!/bin/bash
#SBATCH --job-name=msa-debug
#SBATCH --partition=gpu_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a30:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/debug_%j.out
#SBATCH --mem=8G

# Quick correctness/smoke test on gpu_debug (A30 or H100 nodes, 2hr partition
# limit but we only need a few minutes). Use this while iterating on the kernel.
#
# Submit with:  sbatch jobs/debug_run.slurm
# Swap to H100 instead:  sbatch --gres=gpu:h100:1 jobs/debug_run.slurm

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

module load CUDA/12.6.0
make

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv

echo "=== built-in smoke test (all 3 modes, CPU vs nonPT vs PT verification) ==="
./bin/msa_align

echo "=== small benchmark sweep ==="
./bin/msa_align --benchmark
