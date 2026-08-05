# Requires: module load CUDA/12.6.0   (or any CUDA >= 11.x module on ACES)
NVCC := nvcc

# H100 (sm_90) and A30 (sm_80) are the two GPU types on ACES's gpu/gpu_debug
# partitions (see `sinfo -p gpu,gpu_debug -o "%N %G"`). Build a fat binary
# covering both so the same executable runs on either without recompiling.
ARCHFLAGS := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90

# -rdc=true + -lcudadevrt: required for cooperative_groups grid.sync() (the PT
# global-synchronization barrier) to link correctly.
CXXFLAGS := -O3 -std=c++17 -Iinclude -rdc=true
LDFLAGS  := -lcudadevrt

SRC := src/cpu_reference.cpp src/traceback.cpp src/fasta.cpp src/guide_tree.cpp src/progressive.cpp src/gpu_align.cu src/main.cu
TARGET := bin/msa_align

.PHONY: all clean check-nvcc

all: check-nvcc $(TARGET)

check-nvcc:
	@command -v $(NVCC) >/dev/null 2>&1 || { \
		echo "nvcc not found. On ACES run: module load CUDA/12.6.0"; exit 1; }

$(TARGET): $(SRC) include/align.h
	@mkdir -p bin
	$(NVCC) $(CXXFLAGS) $(ARCHFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -rf bin
