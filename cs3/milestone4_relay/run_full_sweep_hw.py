#!/usr/bin/env python3
"""Full B-relay sweep across the SAME n values as the sequential design's
sweep_cerebras.csv (10,20,40,80,160,320,640), all-pairs among the first n
sequences of sweep_dataset.fasta, one compile/session for the whole sweep.
Writes sweep_batched.csv in the same schema as the other sweeps so it can
sit as a real, complete, directly-comparable series next to the sequential
CS-3 line (which stops at 160) instead of just two spot points.
"""

import itertools
import time

import numpy as np

from cerebras.sdk.client import SdkCompiler, SdkRuntime  # pylint: disable=no-name-in-module
from cerebras.appliance.pb.sdk.sdk_common_pb2 import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder,
)

from run_batch_relay import run_batch
from fasta_util import read_fasta

M = 1000
N = 1000
WIDTH = 750
FASTA_PATH = "./sweep_dataset.fasta"
NS = [10, 20, 40, 80, 160, 320, 640]

# Fixed per-PE memory footprint (bytes): H_row + TB_row + up_row (N+1 x i32/u32
# each) + B (N x i32, only at row 0 per column) + a_char (1 x i32), times
# total active PEs (M rows x WIDTH columns) for this batch width.
BYTES_PER_PE = (N + 1) * 4 * 3 + N * 4 + 4
MEM_USED_MB = (BYTES_PER_PE * M * WIDTH) / (1024.0 * 1024.0)


def compile_kernel(width):
    with SdkCompiler(disable_version_check=True) as compiler:
        artifact_path = compiler.compile(
            ".", "layout.csl",
            f"--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 "
            f"--params=m:{M},n:{N},width:{width} --memcpy --channels=1 -o out",
            ".",
        )
    print(f"compiled artifact (width={width}): {artifact_path}")
    return artifact_path


def main():
    records = read_fasta(FASTA_PATH)
    artifact_path = compile_kernel(WIDTH)

    with open("sweep_batched.csv", "w") as csv, \
         SdkRuntime(artifact_path, simulator=False, disable_version_check=True) as runner:
        csv.write("n,pairs,wall_ms,kernel_ms,io_ms,io_pct,mem_used_mb\n")
        symbols = (runner.get_id('a_char'), runner.get_id('B'),
                   runner.get_id('H_row'), runner.get_id('TB_row'))

        for n in NS:
            all_pairs_idx = list(itertools.combinations(range(n), 2))
            n_launches = (len(all_pairs_idx) + WIDTH - 1) // WIDTH

            t0 = time.time()
            for batch_start in range(0, len(all_pairs_idx), WIDTH):
                batch_idx = all_pairs_idx[batch_start:batch_start + WIDTH]
                pairs = [(records[i][1], records[j][1]) for i, j in batch_idx]
                pad = WIDTH - len(pairs)
                if pad:
                    pairs = pairs + [pairs[0]] * pad
                run_batch(runner, symbols, pairs, M, N, MemcpyOrder.ROW_MAJOR, MemcpyDataType.MEMCPY_32BIT)
            wall_ms = (time.time() - t0) * 1000.0

            print(f"n={n} pairs={len(all_pairs_idx)} launches={n_launches} wall_ms={wall_ms:.1f}")
            csv.write(f"{n},{len(all_pairs_idx)},{wall_ms:.3f},0,0,0,{MEM_USED_MB:.3f}\n")
            csv.flush()

    print("wrote sweep_batched.csv")


if __name__ == "__main__":
    main()
