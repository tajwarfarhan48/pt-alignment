#!/bin/bash
#SBATCH --job-name=msa-gpu-final-push
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/finalize_gpu_final_%j.out
#SBATCH --mem=1G
#SBATCH --dependency=afterany:2026104

# Runs once the GPU 10-sequence correctness comparison job (2026104)
# finishes. Sanity-checks the three output files, then commits+pushes if
# they look right. Login node has internet access for the push; compute
# nodes (this job's cpu partition allocation) do not -- so this job only
# does the check here and defers the actual push to a login-node step by
# printing clear instructions if push fails, same lesson learned from the
# earlier sweep finalize job's silent-push-failure bug.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"

echo "=== job state ==="
sacct -j 2026104 --format=JobID,JobName,State,ExitCode,Elapsed

ok=1
for f in results/gpu/score_matrix.txt results/gpu/guide_tree.newick results/gpu/final_msa.fasta; do
    if [ ! -s "$f" ]; then
        echo "FAIL: $f missing or empty"
        ok=0
    else
        echo "OK: $f ($(wc -l < "$f") lines)"
    fi
done

if [ "$ok" -ne 1 ]; then
    echo "=== NOT committing -- see failures above, check logs/gpu_final_2026104.out ==="
    exit 1
fi

git add results/gpu/score_matrix.txt results/gpu/guide_tree.newick results/gpu/final_msa.fasta results/gpu/full_output.txt
git -c user.name="Souvik" -c user.email="souvik.piyal@gmail.com" commit -m "$(cat <<'EOF'
Add GPU leg of 10-sequence correctness comparison (45 pairs)

Completes the gpu/ deliverable from results/README.md's correctness
section, matching the cpu/ format (score_matrix.txt, guide_tree.newick,
final_msa.fasta, full_output.txt).
EOF
)"

if git push origin souvik 2>&1; then
    echo "=== pushed successfully ==="
else
    echo "=== PUSH FAILED (likely no internet from compute node) ==="
    echo "=== commit is saved locally -- push from the login node with: ==="
    echo "    cd $SLURM_SUBMIT_DIR && git push origin souvik"
    exit 1
fi
