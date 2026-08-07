"""Same as milestone3's run_batch, except B is written once per column
(row y=0 only) instead of once per PE -- the fabric relay in pe_program.csl
propagates it to the rest of the column. This is the whole point of
milestone4: cut the host-side B transfer by a factor of M.
"""

import time

import numpy as np

BASE_CODE = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def encode(seq):
    return np.array([BASE_CODE[c] for c in seq], dtype=np.int32)


def pack_batch(pairs, m, n):
    width = len(pairs)
    a_mat = np.zeros((width, m), dtype=np.int32)
    b_mat = np.zeros((width, n), dtype=np.int32)  # one row per column now, not m rows
    for x, (A, B) in enumerate(pairs):
        a_mat[x] = encode(A)
        b_mat[x] = encode(B)

    a_flat = a_mat.T.flatten()  # (h=m, w=width) row-major, 1 elt/PE
    b_flat = b_mat.flatten()    # (h=1, w=width) row-major, n elts/PE -- 1/m the size

    return a_flat, b_flat


def run_batch(runner, symbols, pairs, m, n, memcpy_order, memcpy_dtype, timing=None):
    width = len(pairs)
    a_symbol, b_symbol, h_symbol, tb_symbol = symbols

    t0 = time.perf_counter()
    a_flat, b_flat = pack_batch(pairs, m, n)
    t1 = time.perf_counter()

    runner.memcpy_h2d(a_symbol, a_flat, 0, 0, width, m, 1, streaming=False,
                       order=memcpy_order, data_type=memcpy_dtype, nonblock=False)
    t2 = time.perf_counter()
    # B: only row y=0, h=1 instead of h=m -- the key change vs milestone3.
    runner.memcpy_h2d(b_symbol, b_flat, 0, 0, width, 1, n, streaming=False,
                       order=memcpy_order, data_type=memcpy_dtype, nonblock=False)
    t3 = time.perf_counter()

    runner.launch('compute', nonblock=False)
    t4 = time.perf_counter()

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
