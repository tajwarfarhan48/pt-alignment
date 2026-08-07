#pragma once
#include <string>
#include <cstdint>

enum AlignMode {
    MODE_GLOBAL     = 0,  // Needleman-Wunsch
    MODE_LOCAL      = 1,  // Smith-Waterman
    MODE_SEMIGLOBAL = 2,  // free end-gaps ("glocal"), used for profile merging
};

struct ScoreParams {
    int match    = 2;
    int mismatch = -1;
    int gap      = 2;   // penalty magnitude, subtracted from score
};

struct AlignResult {
    int score = 0;
    std::string alignedA;
    std::string alignedB;
    double kernel_ms   = 0.0;  // pure device time (excludes H2D/D2H/traceback)
    int num_diagonals  = 0;
    int num_launches   = 0;    // 1 for PT, num_diagonals for nonPT
};

// Traceback pointer encoding, stored per cell.
enum TBDir : uint8_t { TB_STOP = 0, TB_DIAG = 1, TB_UP = 2, TB_LEFT = 3 };

// Shared traceback: walks a fully-computed (H, TB) matrix, flattened row-major
// with row stride ld = n+1. Used by both the CPU reference and the GPU paths so
// tie-breaking can't silently drift between them and break verification.
AlignResult traceback(const std::string &A, const std::string &B,
                       const int *H, const uint8_t *TB, int ld, AlignMode mode);

// CPU reference implementation (O(m*n) time and space) used to verify GPU output.
AlignResult align_cpu(const std::string &A, const std::string &B,
                       const ScoreParams &sp, AlignMode mode);

// nonPT GPU implementation: one kernel launch per anti-diagonal.
AlignResult align_gpu_nonpt(const std::string &A, const std::string &B,
                             const ScoreParams &sp, AlignMode mode);

// PT GPU implementation: single persistent-kernel launch, cooperative-groups
// grid.sync() as the inter-diagonal barrier instead of a kernel relaunch.
AlignResult align_gpu_pt(const std::string &A, const std::string &B,
                          const ScoreParams &sp, AlignMode mode);
