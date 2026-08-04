#include "align.h"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

static constexpr int BLOCK_SIZE = 256;

// Single-cell recurrence, identical to the CPU reference (src/cpu_reference.cpp)
// so the two are directly comparable. H/TB are flattened row-major, stride ld = n+1.
__device__ __forceinline__ void compute_cell(const char *A, const char *B, int *H, uint8_t *TB,
                                              int ld, int i, int j,
                                              int match, int mismatch, int gap, int mode) {
    int idx  = i * ld + j;
    int diag = H[(i - 1) * ld + (j - 1)] + (A[i - 1] == B[j - 1] ? match : mismatch);
    int up   = H[(i - 1) * ld + j] - gap;
    int left = H[i * ld + (j - 1)] - gap;

    int best = diag; uint8_t dir = TB_DIAG;
    if (up   > best) { best = up;   dir = TB_UP; }
    if (left > best) { best = left; dir = TB_LEFT; }
    if (mode == MODE_LOCAL && best < 0) { best = 0; dir = TB_STOP; }

    H[idx]  = best;
    TB[idx] = dir;
}

__global__ void init_borders_kernel(int m, int n, int *H, uint8_t *TB, int ld, int gap, int mode) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i <= m; i += stride) {
        H[i * ld + 0]  = (mode == MODE_GLOBAL) ? -gap * i : 0;
        TB[i * ld + 0] = (i == 0) ? TB_STOP : TB_UP;
    }
    for (int j = tid; j <= n; j += stride) {
        H[0 * ld + j]  = (mode == MODE_GLOBAL) ? -gap * j : 0;
        TB[0 * ld + j] = (j == 0) ? TB_STOP : TB_LEFT;
    }
}

// nonPT: computes exactly one anti-diagonal. The host loops this once per
// diagonal (m+n launches total) -- this is the "traditional" style from the
// Persistent Threads paper, paying kernel-launch overhead at every barrier.
__global__ void wavefront_step_kernel(const char *A, const char *B, int m, int n,
                                       int *H, uint8_t *TB, int ld, int d,
                                       int match, int mismatch, int gap, int mode) {
    int i_min = max(1, d - n), i_max = min(m, d - 1);
    int i = i_min + blockIdx.x * blockDim.x + threadIdx.x;
    if (i > i_max) return;
    int j = d - i;
    compute_cell(A, B, H, TB, ld, i, j, match, mismatch, gap, mode);
}

// PT: a single "maximal launch" of persistent thread groups. Each diagonal is
// processed by looping inside the kernel; grid.sync() (a cooperative-groups
// device-wide barrier) replaces the kernel relaunch as the synchronization
// point between diagonals -- this is the Global Synchronization use case
// (Section 3.4 of Gupta/Stuart/Owens) applied to wavefront DP.
__global__ void wavefront_persistent_kernel(const char *A, const char *B, int m, int n,
                                             int *H, uint8_t *TB, int ld, int num_diag,
                                             int match, int mismatch, int gap, int mode) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int d = 1; d <= num_diag; d++) {
        int i_min = max(1, d - n), i_max = min(m, d - 1);
        int count = i_max - i_min + 1;
        for (int k = tid; k < count; k += stride) {
            int i = i_min + k;
            int j = d - i;
            compute_cell(A, B, H, TB, ld, i, j, match, mismatch, gap, mode);
        }
        grid.sync();
    }
}

// Shared setup: allocate device buffers, copy sequences over, run the border-init
// kernel. Returns device pointers via out-params; caller frees them.
static void gpu_setup(const std::string &A, const std::string &B, const ScoreParams &sp, AlignMode mode,
                       char **dA, char **dB, int **dH, uint8_t **dTB, int &m, int &n, int &ld) {
    m = (int)A.size(); n = (int)B.size(); ld = n + 1;
    size_t hSize  = (size_t)(m + 1) * ld * sizeof(int);
    size_t tbSize = (size_t)(m + 1) * ld * sizeof(uint8_t);

    CUDA_CHECK(cudaMalloc(dA, m));
    CUDA_CHECK(cudaMalloc(dB, n));
    CUDA_CHECK(cudaMalloc(dH, hSize));
    CUDA_CHECK(cudaMalloc(dTB, tbSize));
    CUDA_CHECK(cudaMemcpy(*dA, A.data(), m, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*dB, B.data(), n, cudaMemcpyHostToDevice));

    int initBlocks = ( (m + n + 2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_borders_kernel<<<initBlocks, BLOCK_SIZE>>>(m, n, *dH, *dTB, ld, sp.gap, mode);
    CUDA_CHECK(cudaGetLastError());
}

static AlignResult gpu_finish(const std::string &A, const std::string &B, AlignMode mode,
                               int *dH, uint8_t *dTB, char *dA, char *dB, int m, int n, int ld) {
    std::vector<int>     H((m + 1) * ld);
    std::vector<uint8_t> TB((m + 1) * ld);
    CUDA_CHECK(cudaMemcpy(H.data(),  dH,  H.size()  * sizeof(int),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(TB.data(), dTB, TB.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dH); cudaFree(dTB);
    return traceback(A, B, H.data(), TB.data(), ld, mode);
}

AlignResult align_gpu_nonpt(const std::string &A, const std::string &B,
                             const ScoreParams &sp, AlignMode mode) {
    char *dA, *dB; int *dH; uint8_t *dTB; int m, n, ld;
    gpu_setup(A, B, sp, mode, &dA, &dB, &dH, &dTB, m, n, ld);

    int num_diag = m + n;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int d = 1; d <= num_diag; d++) {
        int i_min = std::max(1, d - n), i_max = std::min(m, d - 1);
        int count = i_max - i_min + 1;
        if (count <= 0) continue;
        int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        wavefront_step_kernel<<<blocks, BLOCK_SIZE>>>(dA, dB, m, n, dH, dTB, ld, d,
                                                        sp.match, sp.mismatch, sp.gap, mode);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start); cudaEventDestroy(stop);

    AlignResult res = gpu_finish(A, B, mode, dH, dTB, dA, dB, m, n, ld);
    res.kernel_ms = ms;
    res.num_diagonals = num_diag;
    res.num_launches = num_diag;
    return res;
}

AlignResult align_gpu_pt(const std::string &A, const std::string &B,
                          const ScoreParams &sp, AlignMode mode) {
    char *dA, *dB; int *dH; uint8_t *dTB; int m, n, ld;
    gpu_setup(A, B, sp, mode, &dA, &dB, &dH, &dTB, m, n, ld);
    int num_diag = m + n;

    int device; CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (!prop.cooperativeLaunch) {
        fprintf(stderr, "Device does not support cooperative launch (grid.sync()); "
                         "PT path unavailable on this GPU.\n");
        exit(1);
    }

    // Maximal launch (Section 2.2 of the PT paper): use exactly as many blocks
    // as can be simultaneously resident, no more -- that's what makes grid.sync()
    // legal (every block must be able to run concurrently) and is the defining
    // property of the PT style vs. the traditional over-subscribed nonPT launch.
    int maxActiveBlocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerSM, wavefront_persistent_kernel, BLOCK_SIZE, 0));
    int numBlocks = maxActiveBlocksPerSM * prop.multiProcessorCount;

    // Local copies matching the kernel's exact parameter types -- the cooperative
    // launch API reads raw bytes from these addresses per the kernel signature,
    // so e.g. passing an AlignMode's address where an int is expected would be
    // relying on enum/int layout compatibility; safer to just copy explicitly.
    int match = sp.match, mismatch = sp.mismatch, gap = sp.gap, modeInt = (int)mode;
    void *kernelArgs[] = { (void*)&dA, (void*)&dB, (void*)&m, (void*)&n,
                            (void*)&dH, (void*)&dTB, (void*)&ld, (void*)&num_diag,
                            (void*)&match, (void*)&mismatch, (void*)&gap, (void*)&modeInt };

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)wavefront_persistent_kernel,
                                            dim3(numBlocks), dim3(BLOCK_SIZE), kernelArgs));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start); cudaEventDestroy(stop);

    AlignResult res = gpu_finish(A, B, mode, dH, dTB, dA, dB, m, n, ld);
    res.kernel_ms = ms;
    res.num_diagonals = num_diag;
    res.num_launches = 1;
    return res;
}
