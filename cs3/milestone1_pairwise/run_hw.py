#!/usr/bin/env python3
"""Host driver for milestone 1 on the real CS-3 (appliance mode), not the
simulator. Compiles via SdkCompiler and runs via SdkRuntime(simulator=False)
from cerebras.sdk.client -- the appliance-mode API, a separate pip package
(cerebras_sdk + cerebras_appliance) from the SDK container's bundled
cerebras.sdk.runtime used by run.py. No mgmt_namespace is passed: this
cluster has a single cluster-wide orchestration namespace, so job submission
commands that don't specify one just work.

Requires the cerebras_sdk / cerebras_appliance packages (see run_hw.sh for
how to set up the venv this needs) -- run via that venv's python, not
cs_python.
"""

import numpy as np

from cerebras.sdk.client import SdkCompiler, SdkRuntime  # pylint: disable=no-name-in-module
from cerebras.appliance.pb.sdk.sdk_common_pb2 import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder,
)

import verify


def main():
    with SdkCompiler(disable_version_check=True) as compiler:
        artifact_path = compiler.compile(
            ".", "layout.csl",
            "--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 "
            "--params=m:8,n:8 --memcpy --channels=1 -o out",
            ".",
        )
    print(f"compiled artifact: {artifact_path}")

    with SdkRuntime(artifact_path, simulator=False, disable_version_check=True) as runner:
        h_symbol = runner.get_id('H_row')
        tb_symbol = runner.get_id('TB_row')

        runner.launch('compute', nonblock=False)

        h_flat = np.zeros([verify.M * 1 * (verify.N + 1)], dtype=np.int32)
        runner.memcpy_d2h(h_flat, h_symbol, 0, 0, verify.M, 1, verify.N + 1, streaming=False,
                           order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
                           nonblock=False)

        tb_flat = np.zeros([verify.M * 1 * (verify.N + 1)], dtype=np.uint32)
        runner.memcpy_d2h(tb_flat, tb_symbol, 0, 0, verify.M, 1, verify.N + 1, streaming=False,
                           order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
                           nonblock=False)

    H, TB = verify.reconstruct_matrix(h_flat.reshape(verify.M, verify.N + 1),
                                       tb_flat.reshape(verify.M, verify.N + 1))
    verify.verify_and_report(H, TB)


if __name__ == "__main__":
    main()
