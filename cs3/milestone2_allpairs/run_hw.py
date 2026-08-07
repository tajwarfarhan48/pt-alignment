#!/usr/bin/env python3
"""Real-CS-3 driver for milestone 2's generalized kernel (same appliance-mode
API as milestone1_pairwise/run_hw.py). Compiles once against the real chip's
full fabric-dims (762x1172), then verifies a single real pair against
align_cpu() before the all-pairs batch script (run_allpairs.py) reuses this
same compiled artifact across all 45 pairs.
"""

import numpy as np

from cerebras.sdk.client import SdkCompiler, SdkRuntime  # pylint: disable=no-name-in-module
from cerebras.appliance.pb.sdk.sdk_common_pb2 import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder,
)

import verify
from fasta_util import read_fasta

M = 1000
N = 1000
FASTA_PATH = "./plasmid_sample.fasta"


def compile_kernel():
    with SdkCompiler(disable_version_check=True) as compiler:
        artifact_path = compiler.compile(
            ".", "layout.csl",
            "--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 "
            "--params=m:1000,n:1000 --memcpy --channels=1 -o out",
            ".",
        )
    print(f"compiled artifact: {artifact_path}")
    return artifact_path


def run_pair(runner, a_symbol, b_symbol, h_symbol, tb_symbol, A, B):
    a_codes = verify.encode(A)
    b_codes = np.tile(verify.encode(B), (M, 1)).flatten()

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

    return verify.reconstruct_matrix(M, N, h_flat.reshape(M, N + 1), tb_flat.reshape(M, N + 1))


def main():
    artifact_path = compile_kernel()
    records = read_fasta(FASTA_PATH)
    idx_a, idx_b = 0, 1
    A, B = records[idx_a][1], records[idx_b][1]
    assert len(A) == M and len(B) == N

    with SdkRuntime(artifact_path, simulator=False, disable_version_check=True) as runner:
        a_symbol = runner.get_id('a_char')
        b_symbol = runner.get_id('B')
        h_symbol = runner.get_id('H_row')
        tb_symbol = runner.get_id('TB_row')

        H, TB = run_pair(runner, a_symbol, b_symbol, h_symbol, tb_symbol, A, B)

    verify.verify_and_report(A, B, H, TB, FASTA_PATH, idx_a, idx_b)


if __name__ == "__main__":
    main()
