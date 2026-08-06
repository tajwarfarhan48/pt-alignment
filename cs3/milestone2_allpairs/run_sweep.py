#!/usr/bin/env python3
"""CS-3 sweep, same CSV schema as the CPU/GPU sweeps (see results/README.md):
n,pairs,wall_ms,kernel_ms,io_ms,io_pct,mem_used_mb.

Reuses ONE compile across every sweep point: all sequences in the dataset
are the same 1000bp, so the kernel shape (m=1000,n=1000 PEs) never changes
between sweep points -- only how many pairs get run through it. That's the
key difference from a naive per-N-recompile approach, and why this doesn't
multiply real-hardware compile time (the expensive, queue-contended step) by
the number of sweep points.

kernel_ms = time inside runner.launch('compute') (the actual on-wafer
compute). io_ms = time inside the memcpy_h2d/memcpy_d2h calls (host<->device
transfer). mem_used_mb is NOT runtime-queried (the SdkRuntime client here
has no memory-query API found) -- it's computed analytically from the
kernel's static per-PE array sizes (H_row/TB_row/up_row/B/a_char), which are
fixed regardless of how many sequences are in the all-pairs batch, since
every pair reuses the same PE memory sequentially.
"""

import itertools
import sys
import time

import numpy as np

from cerebras.sdk.client import SdkRuntime  # pylint: disable=no-name-in-module

import verify
from fasta_util import read_fasta
from run_hw import compile_kernel, M, N

FASTA_PATH = "./sweep_dataset.fasta"

# Fixed per-PE memory footprint (bytes): H_row + TB_row + up_row (N+1 x i32/u32
# each) + B (N x i32) + a_char (1 x i32), times M PEs. Constant across all
# sweep points -- see module docstring.
BYTES_PER_PE = (N + 1) * 4 * 3 + N * 4 + 4
MEM_USED_MB = (BYTES_PER_PE * M) / (1024.0 * 1024.0)


def run_pair_timed(runner, a_symbol, b_symbol, h_symbol, tb_symbol, A, B):
    a_codes = verify.encode(A)
    b_codes = np.tile(verify.encode(B), (M, 1)).flatten()

    t0 = time.perf_counter()
    runner.memcpy_h2d(a_symbol, a_codes, 0, 0, 1, M, 1, streaming=False,
                       order=verify_order(), data_type=verify_dtype(), nonblock=False)
    runner.memcpy_h2d(b_symbol, b_codes, 0, 0, 1, M, N, streaming=False,
                       order=verify_order(), data_type=verify_dtype(), nonblock=False)
    t1 = time.perf_counter()

    runner.launch('compute', nonblock=False)
    t2 = time.perf_counter()

    h_flat = np.zeros([1 * M * (N + 1)], dtype=np.int32)
    runner.memcpy_d2h(h_flat, h_symbol, 0, 0, 1, M, N + 1, streaming=False,
                       order=verify_order(), data_type=verify_dtype(), nonblock=False)
    tb_flat = np.zeros([1 * M * (N + 1)], dtype=np.uint32)
    runner.memcpy_d2h(tb_flat, tb_symbol, 0, 0, 1, M, N + 1, streaming=False,
                       order=verify_order(), data_type=verify_dtype(), nonblock=False)
    t3 = time.perf_counter()

    io_s = (t1 - t0) + (t3 - t2)
    kernel_s = t2 - t1
    score = int(h_flat.reshape(M, N + 1)[M - 1][N])
    return score, io_s, kernel_s


def verify_order():
    from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyOrder  # pylint: disable=no-name-in-module
    return MemcpyOrder.ROW_MAJOR


def verify_dtype():
    from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType  # pylint: disable=no-name-in-module
    return MemcpyDataType.MEMCPY_32BIT


def main():
    ns = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [10, 20, 40, 80, 160, 320, 640]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "sweep_cerebras.csv"

    artifact_path = compile_kernel()
    records = read_fasta(FASTA_PATH)
    for _, seq in records:
        assert len(seq) == M

    with open(out_csv, "w") as csv, SdkRuntime(artifact_path, simulator=False, disable_version_check=True) as runner:
        csv.write("n,pairs,wall_ms,kernel_ms,io_ms,io_pct,mem_used_mb\n")
        a_symbol = runner.get_id('a_char')
        b_symbol = runner.get_id('B')
        h_symbol = runner.get_id('H_row')
        tb_symbol = runner.get_id('TB_row')

        for n in ns:
            if n > len(records):
                print(f"skip n={n} (only {len(records)} available)", file=sys.stderr)
                continue
            pairs = list(itertools.combinations(range(n), 2))
            total_io_s = 0.0
            total_kernel_s = 0.0
            t_wall0 = time.perf_counter()
            for i, j in pairs:
                A, B = records[i][1], records[j][1]
                _score, io_s, kernel_s = run_pair_timed(runner, a_symbol, b_symbol, h_symbol, tb_symbol, A, B)
                total_io_s += io_s
                total_kernel_s += kernel_s
            wall_ms = (time.perf_counter() - t_wall0) * 1000.0
            io_ms = total_io_s * 1000.0
            kernel_ms = total_kernel_s * 1000.0
            io_pct = 100.0 * io_ms / wall_ms if wall_ms > 0 else 0.0

            print(f"n={n} pairs={len(pairs)} wall_ms={wall_ms:.1f} kernel_ms={kernel_ms:.1f} "
                  f"io_pct={io_pct:.1f}% mem_used_mb={MEM_USED_MB:.1f}")
            csv.write(f"{n},{len(pairs)},{wall_ms:.3f},{kernel_ms:.3f},{io_ms:.3f},{io_pct:.3f},{MEM_USED_MB:.3f}\n")
            csv.flush()

    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
