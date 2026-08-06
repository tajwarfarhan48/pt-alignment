#!/usr/bin/env cs_python
"""Host driver for milestone 1: run the PE-per-row global-alignment wavefront
kernel against the SDK simulator, then verify (see verify.py) against the
existing C++ align_cpu() reference. For the real CS-3, see run_hw.py.
"""

import argparse

import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    SdkRuntime, MemcpyDataType, MemcpyOrder,
)

import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help="the test compile output dir", default="out")
    parser.add_argument('--cmaddr', help="IP:port for CS system")
    args = parser.parse_args()

    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    h_symbol = runner.get_id('H_row')
    tb_symbol = runner.get_id('TB_row')

    runner.load()
    runner.run()
    runner.launch('compute', nonblock=False)

    h_flat = np.zeros([verify.M * 1 * (verify.N + 1)], dtype=np.int32)
    runner.memcpy_d2h(h_flat, h_symbol, 0, 0, verify.M, 1, verify.N + 1, streaming=False,
                       order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
                       nonblock=False)

    tb_flat = np.zeros([verify.M * 1 * (verify.N + 1)], dtype=np.uint32)
    runner.memcpy_d2h(tb_flat, tb_symbol, 0, 0, verify.M, 1, verify.N + 1, streaming=False,
                       order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
                       nonblock=False)

    runner.stop()

    H, TB = verify.reconstruct_matrix(h_flat.reshape(verify.M, verify.N + 1),
                                       tb_flat.reshape(verify.M, verify.N + 1))
    verify.verify_and_report(H, TB)


if __name__ == "__main__":
    main()
