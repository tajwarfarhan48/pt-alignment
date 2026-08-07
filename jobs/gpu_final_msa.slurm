#!/bin/bash
#SBATCH --job-name=msa-gpu-final
#SBATCH --partition=gpu_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a30:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/gpu_final_%j.out
#SBATCH --mem=8G

# GPU leg of the 10-sequence correctness comparison (results/README.md,
# section 1) -- the same deliverable already done for cpu/ and in progress
# for cerebras/. 10 sequences -> C(10,2) = 45 pairs.

set -euo pipefail
mkdir -p logs results/gpu
cd "$SLURM_SUBMIT_DIR"
module load CUDA/12.6.0
make

./bin/msa_align --fasta test/plasmid_sample.fasta --msa --out results/gpu/final_msa.fasta \
    | tee results/gpu/full_output.txt

# score_matrix.txt: strip the header (column indices) row and each row's
# leading "N:" label, leaving just whitespace-separated ints -- same format
# as results/cpu/score_matrix.txt.
awk '/^score matrix:/{flag=1; next} /^all-pairs \(GPU, PT\):/{flag=0} flag' results/gpu/full_output.txt \
    | tail -n +2 \
    | sed -E 's/^[[:space:]]*[0-9]+:[[:space:]]*//; s/[[:space:]]+/ /g' \
    > results/gpu/score_matrix.txt

# guide_tree.newick: just the tree line, no "guide tree: " prefix.
grep "^guide tree: " results/gpu/full_output.txt | sed 's/^guide tree: //' > results/gpu/guide_tree.newick

echo "=== results/gpu/score_matrix.txt ==="
cat results/gpu/score_matrix.txt
echo "=== results/gpu/guide_tree.newick ==="
cat results/gpu/guide_tree.newick
echo "=== results/gpu/final_msa.fasta (headers only) ==="
grep "^>" results/gpu/final_msa.fasta
