#!/usr/bin/env cs_python
"""Tiny-scale (m=n=8, width=3) simulator verification of the B-relay
scheme before trusting it at full scale on real hardware."""

import argparse

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    SdkRuntime, MemcpyDataType, MemcpyOrder,
)

from run_batch_relay import run_batch

M = 8
N = 8
PAIRS = [
    ("ACGTACGT", "ACGTTCGT"),
    ("AAAAAAAA", "TTTTTTTT"),
    ("GATTACAG", "GATCACAG"),
]
EXPECTED = [13, -8, 13]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="out")
    parser.add_argument('--cmaddr')
    args = parser.parse_args()

    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    symbols = (runner.get_id('a_char'), runner.get_id('B'),
               runner.get_id('H_row'), runner.get_id('TB_row'))

    runner.load()
    runner.run()

    timing = {}
    scores = run_batch(runner, symbols, PAIRS, M, N,
                        MemcpyOrder.ROW_MAJOR, MemcpyDataType.MEMCPY_32BIT, timing=timing)

    runner.stop()

    print(f"got:      {scores}")
    print(f"expected: {EXPECTED}")
    print(f"timing:   {timing}")
    if scores == EXPECTED:
        print("SUCCESS: B-relay scheme verified against align_cpu()")
    else:
        print("MISMATCH")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
