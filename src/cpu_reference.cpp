#include "align.h"
#include <vector>

// Straightforward O(m*n) DP, used only to check the GPU kernels for correctness.
// Not performance-relevant, so no attempt is made to keep it fast or low-memory.
// Uses the exact same flat layout and tie-breaking as the GPU kernels (see
// gpu_align.cu: compute_cell) so results are comparable cell-for-cell.
AlignResult align_cpu(const std::string &A, const std::string &B,
                       const ScoreParams &sp, AlignMode mode) {
    int m = (int)A.size(), n = (int)B.size();
    int ld = n + 1;
    std::vector<int>     H((m + 1) * ld, 0);
    std::vector<uint8_t> TB((m + 1) * ld, TB_STOP);

    for (int i = 0; i <= m; i++) {
        H[i * ld + 0]  = (mode == MODE_GLOBAL) ? -sp.gap * i : 0;
        TB[i * ld + 0] = (i == 0) ? TB_STOP : TB_UP;
    }
    for (int j = 0; j <= n; j++) {
        H[0 * ld + j]  = (mode == MODE_GLOBAL) ? -sp.gap * j : 0;
        TB[0 * ld + j] = (j == 0) ? TB_STOP : TB_LEFT;
    }

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int diag = H[(i - 1) * ld + (j - 1)] + (A[i - 1] == B[j - 1] ? sp.match : sp.mismatch);
            int up   = H[(i - 1) * ld + j] - sp.gap;
            int left = H[i * ld + (j - 1)] - sp.gap;

            int best = diag; uint8_t dir = TB_DIAG;
            if (up   > best) { best = up;   dir = TB_UP; }
            if (left > best) { best = left; dir = TB_LEFT; }
            if (mode == MODE_LOCAL && best < 0) { best = 0; dir = TB_STOP; }

            H[i * ld + j]  = best;
            TB[i * ld + j] = dir;
        }
    }

    return traceback(A, B, H.data(), TB.data(), ld, mode);
}
