#include "align.h"
#include <algorithm>

AlignResult traceback(const std::string &A, const std::string &B,
                       const int *H, const uint8_t *TB, int ld, AlignMode mode) {
    int m = (int)A.size(), n = (int)B.size();
    auto at = [&](int i, int j) { return H[i * ld + j]; };

    int si = m, sj = n, best = at(m, n);
    if (mode == MODE_LOCAL) {
        best = 0;
        for (int i = 0; i <= m; i++)
            for (int j = 0; j <= n; j++)
                if (at(i, j) > best) { best = at(i, j); si = i; sj = j; }
    } else if (mode == MODE_SEMIGLOBAL) {
        best = at(m, 0); si = m; sj = 0;
        for (int j = 0; j <= n; j++) if (at(m, j) > best) { best = at(m, j); si = m; sj = j; }
        for (int i = 0; i <= m; i++) if (at(i, n) > best) { best = at(i, n); si = i; sj = n; }
    }

    AlignResult res;
    res.score = best;

    std::string alignedA, alignedB;
    int i = si, j = sj;
    if (mode == MODE_SEMIGLOBAL) {
        // Whichever sequence didn't reach its end at the optimal start point gets
        // its unaligned overhang emitted as a free (unscored) trailing gap.
        for (int k = m - 1; k >= i; k--) { alignedA += A[k]; alignedB += '-'; }
        for (int k = n - 1; k >= j; k--) { alignedA += '-'; alignedB += B[k]; }
    }
    while (i > 0 && j > 0 && TB[i * ld + j] != TB_STOP) {
        uint8_t dir = TB[i * ld + j];
        if (dir == TB_DIAG)      { alignedA += A[i - 1]; alignedB += B[j - 1]; i--; j--; }
        else if (dir == TB_UP)   { alignedA += A[i - 1]; alignedB += '-';      i--; }
        else /* TB_LEFT */       { alignedA += '-';       alignedB += B[j - 1]; j--; }
    }
    if (mode == MODE_GLOBAL) {
        while (i > 0) { alignedA += A[i - 1]; alignedB += '-'; i--; }
        while (j > 0) { alignedA += '-';     alignedB += B[j - 1]; j--; }
    } else if (mode == MODE_SEMIGLOBAL) {
        for (int k = i - 1; k >= 0; k--) { alignedA += A[k]; alignedB += '-'; }
        for (int k = j - 1; k >= 0; k--) { alignedA += '-'; alignedB += B[k]; }
    }

    std::reverse(alignedA.begin(), alignedA.end());
    std::reverse(alignedB.begin(), alignedB.end());
    res.alignedA = alignedA;
    res.alignedB = alignedB;
    return res;
}
