#!/usr/bin/env python3
"""Real-CS-3 all-pairs batch: compiles the generalized kernel once, opens a
single SdkRuntime session, and reuses it across every pair in the dataset
(all sequences are 1000bp, so no recompilation is needed between pairs --
only the two memcpy_h2d payloads change). Writes the resulting score matrix
to scores.txt (n lines of n whitespace-separated ints), the format
cs3_msa_cli's --scores flag expects.

Run only after run_hw.py has verified a single pair cell-for-cell against
align_cpu() -- this script does not re-verify, it trusts that check.
"""

import itertools
import time

import numpy as np

from cerebras.sdk.client import SdkRuntime  # pylint: disable=no-name-in-module

import verify
from fasta_util import read_fasta
from run_hw import compile_kernel, run_pair, M, N, FASTA_PATH


def main():
    artifact_path = compile_kernel()
    records = read_fasta(FASTA_PATH)
    n_seqs = len(records)
    for _, seq in records:
        assert len(seq) == M, f"expected all sequences {M}bp, got {len(seq)}"

    score = np.zeros((n_seqs, n_seqs), dtype=np.int64)
    pairs = list(itertools.combinations(range(n_seqs), 2))
    print(f"all-pairs: {n_seqs} sequences, {len(pairs)} pairs")

    t0 = time.time()
    with SdkRuntime(artifact_path, simulator=False, disable_version_check=True) as runner:
        a_symbol = runner.get_id('a_char')
        b_symbol = runner.get_id('B')
        h_symbol = runner.get_id('H_row')
        tb_symbol = runner.get_id('TB_row')

        for k, (i, j) in enumerate(pairs):
            A, B = records[i][1], records[j][1]
            H, _TB = run_pair(runner, a_symbol, b_symbol, h_symbol, tb_symbol, A, B)
            s = int(H[M][N])
            score[i][j] = score[j][i] = s
            print(f"  [{k + 1}/{len(pairs)}] ({i},{j}) score={s}")
    wall_s = time.time() - t0

    with open("scores.txt", "w") as f:
        for i in range(n_seqs):
            f.write(" ".join(str(score[i][j]) for j in range(n_seqs)) + "\n")

    print(f"\nwrote scores.txt ({n_seqs}x{n_seqs})")
    print(f"all-pairs wall time: {wall_s:.1f}s ({len(pairs)} sequential CS-3 alignments, one session)")


if __name__ == "__main__":
    main()
