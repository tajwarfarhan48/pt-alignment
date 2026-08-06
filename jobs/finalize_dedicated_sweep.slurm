#!/bin/bash
#SBATCH --job-name=msa-sweep-monitored-finalize
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/finalize_monitored_%j.out
#SBATCH --mem=1G
#SBATCH --dependency=afterany:2026150:2026151

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"

echo "=== job states ==="
sacct -j 2026150,2026151 --format=JobID,JobName,State,ExitCode,Elapsed,NodeList

check_csv() {
    local path="$1"
    if [ ! -s "$path" ]; then echo "FAIL: $path missing or empty"; return 1; fi
    local rows=$(($(wc -l < "$path") - 1))
    if [ "$rows" -ne 7 ]; then echo "FAIL: $path has $rows data rows, expected 7"; return 1; fi
    echo "OK: $path has 7 data rows"
    return 0
}

gpu_ok=1; check_csv results/gpu/sweep_gpu.csv || gpu_ok=0
cpu_ok=1; check_csv results/sweep_cpu_aces.csv || cpu_ok=0

if [ "$gpu_ok" -ne 1 ] || [ "$cpu_ok" -ne 1 ]; then
    echo "=== NOT committing -- see failures above ==="
    exit 1
fi

gpu_node=$(sacct -j 2026150 --format=NodeList --noheader | head -1 | xargs)
cpu_node=$(sacct -j 2026151 --format=NodeList --noheader | head -1 | xargs)

echo "=== checking whether the nodes were shared with other users ==="
gpu_start=$(sacct -j 2026150 --format=Start --noheader | head -1 | xargs)
gpu_end=$(sacct -j 2026150 --format=End --noheader | head -1 | xargs)
cpu_start=$(sacct -j 2026151 --format=Start --noheader | head -1 | xargs)
cpu_end=$(sacct -j 2026151 --format=End --noheader | head -1 | xargs)

echo "--- other jobs on $gpu_node during $gpu_start .. $gpu_end ---"
sacct -a --nodelist="$gpu_node" --starttime="$gpu_start" --endtime="$gpu_end" \
    --format=JobID,User,Partition,State --noheader 2>/dev/null | grep -v "^ *$SLURM_JOB_ID" | grep -v "^\s*2026150" || echo "(none found / query failed)"

echo "--- other jobs on $cpu_node during $cpu_start .. $cpu_end ---"
sacct -a --nodelist="$cpu_node" --starttime="$cpu_start" --endtime="$cpu_end" \
    --format=JobID,User,Partition,State --noheader 2>/dev/null | grep -v "^\s*2026151" || echo "(none found / query failed)"

cat >> results/HARDWARE.md << EOF

## Update: monitored reruns (ACES), not exclusive

Attempted \`--exclusive\` reruns first, but that requires an entirely free
node and the scheduler ETA came back too slow given the time budget (next
day). Dropped \`--exclusive\`, kept live utilization capture instead --
sharing is visible in the data even though it isn't prevented.

- **GPU**: node \`$gpu_node\`. Live utilization sampled every 5s throughout:
  \`results/gpu/gpu_utilization.csv\`. See job log for whether other users'
  jobs were on this node concurrently (same sacct method as the original
  HARDWARE.md sections above).
- **CPU**: node \`$cpu_node\`. Live per-core utilization sampled every 5s
  throughout: \`results/sweep_cpu_aces_utilization.log\`.

These numbers supersede the earlier (also non-exclusive, unmonitored)
\`results/gpu/sweep_gpu.csv\` / \`results/sweep_cpu_aces.csv\` values (files
overwritten in place; prior values still visible in git history) --
upgrade here is the added utilization telemetry, not isolation.
EOF

git add results/gpu/sweep_gpu.csv results/gpu/gpu_utilization.csv \
        results/sweep_cpu_aces.csv results/sweep_cpu_aces_utilization.log \
        results/HARDWARE.md
git -c user.name="Souvik" -c user.email="souvik.piyal@gmail.com" commit -m "$(cat <<'EOF'
Rerun ACES GPU+CPU sweeps with live utilization capture

--exclusive was tried but too slow to schedule given the time budget
(needs a fully free node, ETA was next-day); dropped it and kept the
nvidia-smi/mpstat utilization sampling instead, so node sharing is at
least visible in the data rather than untracked. See results/HARDWARE.md.
EOF
)"

if git push origin souvik 2>&1; then
    echo "=== pushed successfully ==="
else
    echo "=== PUSH FAILED (likely no internet from compute node) ==="
    echo "=== commit saved locally -- push from login node: cd $SLURM_SUBMIT_DIR && git push origin souvik ==="
    exit 1
fi
