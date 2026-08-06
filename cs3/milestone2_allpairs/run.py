#!/usr/bin/env cs_python
"""Simulator driver for milestone 2's generalized kernel: uploads a real pair
(runtime memcpy_h2d, not compile-time constants) from the plasmid sample
fasta, runs the vertical-PE-line wavefront, and verifies against align_cpu()
(see verify.py) for one real pair before any real-hardware time is spent.
"""

import argparse

import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    SdkRuntime, MemcpyDataType, MemcpyOrder,
)

import verify
from fasta_util import read_fasta

M = 1000
N = 1000
FASTA_PATH = "./plasmid_sample.fasta"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help="the test compile output dir", default="out")
    parser.add_argument('--cmaddr', help="IP:port for CS system")
    parser.add_argument('--idxA', type=int, default=0)
    parser.add_argument('--idxB', type=int, default=1)
    args = parser.parse_args()

    records = read_fasta(FASTA_PATH)
    A = records[args.idxA][1]
    B = records[args.idxB][1]
    assert len(A) == M and len(B) == N, f"expected {M}x{N}, got {len(A)}x{len(B)}"

    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    a_symbol = runner.get_id('a_char')
    b_symbol = runner.get_id('B')
    h_symbol = runner.get_id('H_row')
    tb_symbol = runner.get_id('TB_row')

    runner.load()
    runner.run()

    a_codes = verify.encode(A)  # one distinct value per PE, shape (M,)
    b_codes = np.tile(verify.encode(B), (M, 1)).flatten()  # same B broadcast to every PE

    runner.memcpy_h2d(a_symbol, a_codes, 0, 0, 1, M, 1, streaming=False,
                       order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
                       nonblock=False)
    runner.memcpy_h2d(b_symbol, b_codes, 0, 0, 1, M, N, streaming=False,
                       order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
                       nonblock=False)

    runner.launch('compute', nonblock=False)

    h_flat = np.zeros([1 * M * (N + 1)], dtype=np.int32)
    runner.memcpy_d2h(h_flat, h_symbol, 0, 0, 1, M, N + 1, streaming=False,
                       order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
                       nonblock=False)

    tb_flat = np.zeros([1 * M * (N + 1)], dtype=np.uint32)
    runner.memcpy_d2h(tb_flat, tb_symbol, 0, 0, 1, M, N + 1, streaming=False,
                       order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
                       nonblock=False)

    runner.stop()

    H, TB = verify.reconstruct_matrix(M, N, h_flat.reshape(M, N + 1), tb_flat.reshape(M, N + 1))
    verify.verify_and_report(A, B, H, TB, FASTA_PATH, args.idxA, args.idxB)


if __name__ == "__main__":
    main()
