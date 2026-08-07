#!/usr/bin/env python3
"""CS-3 correctness deliverable: all-pairs on plasmid_sample.fasta (the 10
diverse-species dataset, same one used for the CPU/GPU final-MSA
comparison), via the B-relay batched kernel. 45 pairs fits in a single
width>=45 batch -- one launch, sub-second compute. Writes scores.txt in the
n-lines-of-n-ints format cs3_msa_cli --scores expects, so the guide-tree/
merge step can run unchanged on top of it, exactly like the CPU path.
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
WIDTH = 45  # >= the 45 pairs needed, single batch
FASTA_PATH = "./plasmid_sample.fasta"


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
    n = len(records)
    assert n == 10, f"expected 10 sequences, got {n}"
    for _, seq in records:
        assert len(seq) == M

    artifact_path = compile_kernel(WIDTH)

    with SdkRuntime(artifact_path, simulator=False, disable_version_check=True) as runner:
        symbols = (runner.get_id('a_char'), runner.get_id('B'),
                   runner.get_id('H_row'), runner.get_id('TB_row'))

        all_pairs_idx = list(itertools.combinations(range(n), 2))
        pairs = [(records[i][1], records[j][1]) for i, j in all_pairs_idx]
        pad = WIDTH - len(pairs)
        if pad:
            pairs = pairs + [pairs[0]] * pad

        t0 = time.time()
        batch_scores = run_batch(runner, symbols, pairs, M, N,
                                  MemcpyOrder.ROW_MAJOR, MemcpyDataType.MEMCPY_32BIT)
        wall_s = time.time() - t0

        scores = {}
        for (i, j), s in zip(all_pairs_idx, batch_scores):
            scores[(i, j)] = s
        print(f"{len(all_pairs_idx)} pairs in 1 launch, {wall_s:.3f}s wall time")

    with open("scores.txt", "w") as f:
        for i in range(n):
            f.write(" ".join(str(scores.get((min(i, j), max(i, j)), 0)) for j in range(n)) + "\n")
    print("wrote scores.txt")


if __name__ == "__main__":
    main()
