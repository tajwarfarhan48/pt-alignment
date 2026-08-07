"""Shared verification logic for milestone 1, used by both the simulator
driver (run.py) and the real-hardware driver (run_hw.py): reconstruct the
full H/TB matrix from each PE's exported row, run traceback in Python
(mirroring traceback.cpp's MODE_GLOBAL path), and diff against the C++
align_cpu() reference (./reference/dump_reference).
"""

import subprocess
import sys

import numpy as np

M = 8  # len(A), one PE per row
N = 8  # len(B)
GAP = 2

A = "ACGTACGT"
B = "ACGTTCGT"

TB_STOP, TB_DIAG, TB_UP, TB_LEFT = 0, 1, 2, 3


def load_reference():
    """Run the C++ align_cpu() reference dump and parse its output."""
    out = subprocess.run(
        ["./reference/dump_reference"], capture_output=True, text=True, check=True
    ).stdout.splitlines()

    idx = {line.split(" ", 1)[0]: i for i, line in enumerate(out)}
    score = int(out[idx["score"]].split()[1])
    aligned_a = out[idx["alignedA"]].split()[1]
    aligned_b = out[idx["alignedB"]].split()[1]

    h_start = idx["H"] + 1
    tb_start = idx["TB"] + 1
    H = np.array([[int(x) for x in out[h_start + i].split()] for i in range(M + 1)])
    TB = np.array([[int(x) for x in out[tb_start + i].split()] for i in range(M + 1)])
    return score, aligned_a, aligned_b, H, TB


def traceback_global(H, TB):
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


def reconstruct_matrix(device_rows_h, device_rows_tb):
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


def verify_and_report(H, TB):
    """Diff (H, TB, score, aligned strings) against align_cpu(). Exits 1 on
    mismatch, prints SUCCESS and returns otherwise."""
    score = int(H[M][N])
    aligned_a, aligned_b = traceback_global(H, TB)

    ref_score, ref_a, ref_b, ref_H, ref_TB = load_reference()

    ok = True
    if not np.array_equal(H, ref_H):
        ok = False
        print("MISMATCH: H matrix differs from align_cpu() reference", file=sys.stderr)
        print("device H:\n", H, file=sys.stderr)
        print("ref H:\n", ref_H, file=sys.stderr)
    if not np.array_equal(TB, ref_TB):
        ok = False
        print("MISMATCH: TB matrix differs from align_cpu() reference", file=sys.stderr)
        print("device TB:\n", TB, file=sys.stderr)
        print("ref TB:\n", ref_TB, file=sys.stderr)
    if score != ref_score:
        ok = False
        print(f"MISMATCH: score {score} != reference {ref_score}", file=sys.stderr)
    if aligned_a != ref_a or aligned_b != ref_b:
        ok = False
        print(f"MISMATCH: aligned strings differ:\n  CS-3: {aligned_a} / {aligned_b}\n"
              f"  ref:  {ref_a} / {ref_b}", file=sys.stderr)

    if not ok:
        sys.exit(1)

    print(f"SUCCESS: score={score} alignedA={aligned_a} alignedB={aligned_b} "
          "(matches align_cpu() cell-for-cell)")
