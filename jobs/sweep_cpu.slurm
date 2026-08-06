#!/bin/bash
#SBATCH --job-name=msa-sweep-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/sweep_cpu_%j.out
#SBATCH --mem=4G

# CPU leg of the cross-backend (CPU / GPU / Cerebras CS-3) sweep benchmark.
# See results/README.md for methodology. sweep_cpu.cpp is single-threaded
# (plain sequential double for-loop over pairs, no OpenMP), so this measures
# single-core performance -- core count on the node is irrelevant here.
# Generous 4hr budget: same 272,415-pairwise-alignment workload as the GPU
# sweep, but done serially on CPU with no parallelism across cells/diagonals,
# so it's expected to be markedly slower than the GPU leg per pair.

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

module load GCC/13.2.0
g++ -O3 -std=c++17 -Iinclude \
    results/sweep_cpu.cpp src/cpu_reference.cpp src/traceback.cpp src/fasta.cpp \
    -o bin/sweep_cpu

echo "=== CPU info ==="
lscpu | grep -E "Model name|CPU\(s\)|MHz"

echo "=== sweep: n=10,20,40,80,160,320,640 on test/sweep_dataset.fasta ==="
./bin/sweep_cpu test/sweep_dataset.fasta 10,20,40,80,160,320,640 results/sweep_cpu.csv

echo "=== results/sweep_cpu.csv ==="
cat results/sweep_cpu.csv
