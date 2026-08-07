#!/usr/bin/env python3
"""Batched host driver: uploads WIDTH pairs' sequences in one memcpy_h2d
each, launches once, reads back WIDTH scores in one memcpy_d2h. Works
against either the simulator (cs_python, no --cmaddr) or real hardware
(appliance mode, separate script) -- this one targets whichever runner is
passed in, so the same packing/unpacking logic gets exercised at both tiny
scale (fast simulator verification) and full scale (real hardware).

Memcpy addressing: a_char is one i32 per PE over a (width, m) rectangle;
B is n i32s per PE over the same rectangle. ROW_MAJOR order is assumed to
mean the host array is shaped (m, width, ...) flattened C-order (row = a
fixed height-position y, spanning all width/columns at that y) -- verified
empirically below against align_cpu(), not just assumed.
"""

import time

import numpy as np

BASE_CODE = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def encode(seq):
    return np.array([BASE_CODE[c] for c in seq], dtype=np.int32)


def pack_batch(pairs, m, n):
    """pairs: list of (A, B) sequence tuples, each len m / len n respectively.
    Returns (a_flat, b_flat) ready for memcpy_h2d."""
    width = len(pairs)
    a_mat = np.zeros((width, m), dtype=np.int32)   # [x][y]
    b_mat = np.zeros((width, n), dtype=np.int32)   # [x][:]
    for x, (A, B) in enumerate(pairs):
        a_mat[x] = encode(A)
        b_mat[x] = encode(B)

    # a_char: one element per PE, (h=m, w=width) row-major
    a_flat = a_mat.T.flatten()  # [y*width + x] = A_x[y]

    # B: n elements per PE, (h=m, w=width, elt_per_pe=n) row-major, same
    # B_x repeated down every row y of column x
    b_full = np.tile(b_mat, (m, 1, 1))  # shape (m, width, n)
    b_flat = b_full.flatten()

    return a_flat, b_flat


def run_batch(runner, symbols, pairs, m, n, memcpy_order, memcpy_dtype, timing=None):
    """timing, if passed a dict, gets populated with pack_s/h2d_a_s/h2d_b_s/
    launch_s/d2h_s/total_s for this call -- lets callers distinguish fixed
    per-call overhead from data-volume-proportional transfer time."""
    width = len(pairs)
    a_symbol, b_symbol, h_symbol, tb_symbol = symbols

    t0 = time.perf_counter()
    a_flat, b_flat = pack_batch(pairs, m, n)
    t1 = time.perf_counter()

    runner.memcpy_h2d(a_symbol, a_flat, 0, 0, width, m, 1, streaming=False,
                       order=memcpy_order, data_type=memcpy_dtype, nonblock=False)
    t2 = time.perf_counter()
    runner.memcpy_h2d(b_symbol, b_flat, 0, 0, width, m, n, streaming=False,
                       order=memcpy_order, data_type=memcpy_dtype, nonblock=False)
    t3 = time.perf_counter()

    runner.launch('compute', nonblock=False)
    t4 = time.perf_counter()

    # Only need the last row (y=m-1) of every column for the score.
    h_flat = np.zeros([width * 1 * (n + 1)], dtype=np.int32)
    runner.memcpy_d2h(h_flat, h_symbol, 0, m - 1, width, 1, n + 1, streaming=False,
                       order=memcpy_order, data_type=memcpy_dtype, nonblock=False)
    t5 = time.perf_counter()

    if timing is not None:
        timing.update(pack_s=t1 - t0, h2d_a_s=t2 - t1, h2d_b_s=t3 - t2,
                       launch_s=t4 - t3, d2h_s=t5 - t4, total_s=t5 - t0,
                       b_bytes=b_flat.nbytes)

    scores = h_flat.reshape(width, n + 1)[:, n]
    return scores.tolist()
