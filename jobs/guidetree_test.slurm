#!/bin/bash
#SBATCH --job-name=msa-guidetree
#SBATCH --partition=gpu_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a30:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/guidetree_%j.out
#SBATCH --mem=8G

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
module load CUDA/12.6.0
make
./bin/msa_align --fasta test/silva_sample.fasta --guidetree
