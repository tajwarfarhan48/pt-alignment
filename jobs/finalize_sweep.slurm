#!/bin/bash
#SBATCH --job-name=msa-sweep-finalize
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/finalize_%j.out
#SBATCH --mem=1G
#SBATCH --dependency=afterany:2025807:2025820

# Runs automatically once both sweep jobs (2025807 GPU, 2025820 CPU) reach a
# terminal state (success OR failure -- afterany, not afterok, so we still
# get a report if one of them fails). Verifies both CSVs, commits + pushes
# if they look sane, otherwise just reports what's wrong without guessing at
# a fix. This is deliberately a SLURM job, not a background process tied to
# any interactive session -- it runs on the cluster's own scheduler.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"

echo "=== job states ==="
sacct -j 2025807,2025820 --format=JobID,JobName,State,ExitCode,Elapsed

check_csv() {
    local path="$1"
    if [ ! -s "$path" ]; then
        echo "FAIL: $path missing or empty"
        return 1
    fi
    local rows
    rows=$(($(wc -l < "$path") - 1))
    if [ "$rows" -ne 7 ]; then
        echo "FAIL: $path has $rows data rows, expected 7"
        return 1
    fi
    echo "OK: $path has 7 data rows"
    cat "$path"
    return 0
}

echo "=== results/gpu/sweep_gpu.csv ==="
gpu_ok=1
check_csv results/gpu/sweep_gpu.csv || gpu_ok=0

echo "=== results/sweep_cpu.csv ==="
cpu_ok=1
check_csv results/sweep_cpu.csv || cpu_ok=0

if [ "$gpu_ok" -eq 1 ] && [ "$cpu_ok" -eq 1 ]; then
    echo "=== both CSVs look sane, committing ==="
    git add results/gpu/sweep_gpu.csv results/sweep_cpu.csv
    git -c user.name="Souvik" -c user.email="souvik.piyal@gmail.com" commit -m "$(cat <<'EOF'
Add GPU and CPU sweep benchmark results

n=10,20,40,80,160,320,640 all-pairs alignment on test/sweep_dataset.fasta,
same methodology on both backends (see results/README.md). GPU run on
H100 (job 2025807), CPU run single-threaded (job 2025820).
EOF
)"
    git push origin souvik
    echo "=== pushed ==="
else
    echo "=== NOT pushing -- one or both CSVs failed verification, see above ==="
    echo "GPU job log: logs/sweep_2025807.out"
    echo "CPU job log: logs/sweep_cpu_2025826.out"
fi
