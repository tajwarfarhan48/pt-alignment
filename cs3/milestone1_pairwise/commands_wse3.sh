#!/usr/bin/env bash
# Build + run milestone 1 against the SDK simulator (no --cmaddr => simulated
# fabric, not the physical CS-3). SDK tools live in /cra-1684/SDK_Image on
# this cluster; add that dir to PATH or call cslc/cs_python by full path.

set -e

# CPU reference (align_cpu(), no CUDA/CSL) -- the correctness oracle.
g++ -std=c++17 -O2 reference/dump_reference.cpp \
  ../../src/cpu_reference.cpp ../../src/traceback.cpp \
  -I ../../include -o reference/dump_reference

cslc --arch=wse3 ./layout.csl --fabric-dims=15,3 \
  --fabric-offsets=4,1 --params=m:8,n:8 -o out --memcpy --channels 1

cs_python run.py --name out
