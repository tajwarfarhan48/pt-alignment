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

set -euo pipefail
mkdir -p logs results/gpu
cd "$SLURM_SUBMIT_DIR"

module load CUDA/12.6.0
make

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

echo "=== sweep: n=10,20,40,80,160,320,640 on test/sweep_dataset.fasta ==="
./bin/msa_align --fasta test/sweep_dataset.fasta --sweep 10,20,40,80,160,320,640 \
    --sweep-out results/gpu/sweep_gpu.csv

echo "=== results/gpu/sweep_gpu.csv ==="
cat results/gpu/sweep_gpu.csv
