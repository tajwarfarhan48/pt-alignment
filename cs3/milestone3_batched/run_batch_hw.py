#!/usr/bin/env python3
"""Real-CS-3 driver for the batched kernel. Compiles once for a given WIDTH,
then can process any all-pairs workload in ceil(num_pairs/WIDTH) launches
instead of num_pairs launches. First call (--verify) checks a small batch
against align_cpu() on real hardware before trusting the full-width run.
"""

import argparse
import itertools
import sys
import time

import numpy as np

from cerebras.sdk.client import SdkCompiler, SdkRuntime  # pylint: disable=no-name-in-module
from cerebras.appliance.pb.sdk.sdk_common_pb2 import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder,
)

from run_batch import run_batch, encode
from fasta_util import read_fasta

M = 1000
N = 1000
FASTA_PATH = "./sweep_dataset.fasta"


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


def verify_small_batch(runner, symbols, records, width):
    """Check `width` real pairs against align_cpu() before trusting the batch."""
    import subprocess
    pairs = [(records[i][1], records[i + 1][1]) for i in range(0, 2 * width, 2)]
    timing = {}
    got = run_batch(runner, symbols, pairs, M, N, MemcpyOrder.ROW_MAJOR, MemcpyDataType.MEMCPY_32BIT,
                     timing=timing)
    print(f"timing (width={width}): {timing}")

    expected = []
    for i in range(0, 2 * width, 2):
        out = subprocess.run(["./reference/dump_pair", FASTA_PATH, str(i), str(i + 1)],
                              capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            if line.startswith("score "):
                expected.append(int(line.split()[1]))

    print(f"got:      {got}")
    print(f"expected: {expected}")
    if got != expected:
        print("MISMATCH -- not proceeding to full batch", file=sys.stderr)
        sys.exit(1)
    print(f"SUCCESS: {width}-wide real-hardware batch matches align_cpu() exactly")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=4)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--n-seqs', type=int, default=None,
                         help="run all-pairs on the first N sequences using this batch width")
    args = parser.parse_args()

    records = read_fasta(FASTA_PATH)
    artifact_path = compile_kernel(args.width)

    with SdkRuntime(artifact_path, simulator=False, disable_version_check=True) as runner:
        symbols = (runner.get_id('a_char'), runner.get_id('B'),
                   runner.get_id('H_row'), runner.get_id('TB_row'))

        verify_small_batch(runner, symbols, records, min(args.width, 4))

        if args.verify_only:
            return

        n = args.n_seqs
        all_pairs_idx = list(itertools.combinations(range(n), 2))
        print(f"all-pairs batched: {n} sequences, {len(all_pairs_idx)} pairs, width={args.width}")

        scores = {}
        t0 = time.time()
        for batch_start in range(0, len(all_pairs_idx), args.width):
            batch_idx = all_pairs_idx[batch_start:batch_start + args.width]
            pairs = [(records[i][1], records[j][1]) for i, j in batch_idx]
            # pad the last (possibly short) batch by repeating the first pair
            pad = args.width - len(pairs)
            if pad:
                pairs = pairs + [pairs[0]] * pad
            batch_scores = run_batch(runner, symbols, pairs, M, N,
                                      MemcpyOrder.ROW_MAJOR, MemcpyDataType.MEMCPY_32BIT)
            for (i, j), s in zip(batch_idx, batch_scores):
                scores[(i, j)] = s
            print(f"  batch {batch_start}-{batch_start+len(batch_idx)}/{len(all_pairs_idx)} done")
        wall_s = time.time() - t0

        n_launches = (len(all_pairs_idx) + args.width - 1) // args.width
        print(f"\n{len(all_pairs_idx)} pairs in {n_launches} launches, {wall_s:.2f}s wall time")

        with open(f"scores_n{n}_w{args.width}.txt", "w") as f:
            for i in range(n):
                f.write(" ".join(str(scores.get((min(i, j), max(i, j)), 0)) for j in range(n)) + "\n")
        print(f"wrote scores_n{n}_w{args.width}.txt")


if __name__ == "__main__":
    main()
