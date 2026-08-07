"""Shared verification logic for milestone 2 (generalized, runtime-uploaded
kernel): reconstruct the full H/TB matrix from each PE's exported row, run
traceback in Python (mirroring traceback.cpp's MODE_GLOBAL path), and diff
against the C++ align_cpu() reference (./reference/dump_pair).
"""

import subprocess
import sys

import numpy as np

BASE_CODE = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
GAP = 2

TB_STOP, TB_DIAG, TB_UP, TB_LEFT = 0, 1, 2, 3


def encode(seq):
    return np.array([BASE_CODE[c] for c in seq], dtype=np.int32)


def load_reference(fasta_path, idx_a, idx_b):
    """Run the C++ align_cpu() reference dump for a real pair and parse it."""
    out = subprocess.run(
        ["./reference/dump_pair", fasta_path, str(idx_a), str(idx_b)],
        capture_output=True, text=True, check=True
    ).stdout.splitlines()

    idx = {line.split(" ", 1)[0]: i for i, line in enumerate(out)}
    m = int(out[idx["m"]].split()[1])
    n = int(out[idx["n"]].split()[1])
    score = int(out[idx["score"]].split()[1])
    aligned_a = out[idx["alignedA"]].split()[1]
    aligned_b = out[idx["alignedB"]].split()[1]

    h_start = idx["H"] + 1
    tb_start = idx["TB"] + 1
    H = np.array([[int(x) for x in out[h_start + i].split()] for i in range(m + 1)])
    TB = np.array([[int(x) for x in out[tb_start + i].split()] for i in range(m + 1)])
    return m, n, score, aligned_a, aligned_b, H, TB


def traceback_global(A, B, M, N, H, TB):
    """Mirror traceback.cpp's MODE_GLOBAL path exactly."""
    i, j = M, N
    aligned_a, aligned_b = [], []
    while i > 0 and j > 0 and TB[i][j] != TB_STOP:
        d = TB[i][j]
        if d == TB_DIAG:
            aligned_a.append(A[i - 1]); aligned_b.append(B[j - 1]); i -= 1; j -= 1
        elif d == TB_UP:
            aligned_a.append(A[i - 1]); aligned_b.append('-'); i -= 1
        else:
            aligned_a.append('-'); aligned_b.append(B[j - 1]); j -= 1
    while i > 0:
        aligned_a.append(A[i - 1]); aligned_b.append('-'); i -= 1
    while j > 0:
        aligned_a.append('-'); aligned_b.append(B[j - 1]); j -= 1
    return "".join(reversed(aligned_a)), "".join(reversed(aligned_b))


def reconstruct_matrix(M, N, device_rows_h, device_rows_tb):
    """Row 0 is the border row: never computed/exported by a PE, synthesized
    identically to cpu_reference.cpp / align_cpu()'s init loop."""
    H = np.zeros((M + 1, N + 1), dtype=np.int64)
    TB = np.zeros((M + 1, N + 1), dtype=np.int64)
    for j in range(N + 1):
        H[0][j] = -GAP * j
        TB[0][j] = TB_STOP if j == 0 else TB_LEFT
    H[1:, :] = device_rows_h
    TB[1:, :] = device_rows_tb
    return H, TB


def verify_and_report(A, B, H, TB, fasta_path, idx_a, idx_b):
    """Diff (H, TB, score, aligned strings) against align_cpu() for a real
    pair. Exits 1 on mismatch, prints SUCCESS and returns otherwise."""
    M, N = len(A), len(B)
    score = int(H[M][N])
    aligned_a, aligned_b = traceback_global(A, B, M, N, H, TB)

    ref_m, ref_n, ref_score, ref_a, ref_b, ref_H, ref_TB = load_reference(fasta_path, idx_a, idx_b)
    assert (ref_m, ref_n) == (M, N), f"shape mismatch: device ({M},{N}) vs ref ({ref_m},{ref_n})"

    ok = True
    if not np.array_equal(H, ref_H):
        ok = False
        print("MISMATCH: H matrix differs from align_cpu() reference", file=sys.stderr)
    if not np.array_equal(TB, ref_TB):
        ok = False
        print("MISMATCH: TB matrix differs from align_cpu() reference", file=sys.stderr)
    if score != ref_score:
        ok = False
        print(f"MISMATCH: score {score} != reference {ref_score}", file=sys.stderr)
    if aligned_a != ref_a or aligned_b != ref_b:
        ok = False
        print("MISMATCH: aligned strings differ", file=sys.stderr)

    if not ok:
        sys.exit(1)

    print(f"SUCCESS: pair ({idx_a},{idx_b}) score={score} "
          "(matches align_cpu() cell-for-cell, full 1001x1001 H/TB matrix + traceback)")
