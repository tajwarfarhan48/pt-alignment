// Standalone reference harness (no CUDA, no CSL) for milestone-1 correctness
// checking on CS-3. Runs align_cpu() on a fixed small sequence pair and prints
// the score, aligned strings, and the full H/TB matrices so a CSL kernel's
// output can be diffed cell-for-cell, matching how this project's CUDA paths
// were verified against align_cpu() on ACES.
#include "../../../include/align.h"
#include <cstdio>
#include <vector>

int main() {
    const std::string A = "ACGTACGT";
    const std::string B = "ACGTTCGT";
    ScoreParams sp; // match=2, mismatch=-1, gap=2

    int m = (int)A.size(), n = (int)B.size();
    int ld = n + 1;
    std::vector<int>     H((m + 1) * ld, 0);
    std::vector<uint8_t> TB((m + 1) * ld, TB_STOP);

    for (int i = 0; i <= m; i++) {
        H[i * ld + 0]  = -sp.gap * i;
        TB[i * ld + 0] = (i == 0) ? TB_STOP : TB_UP;
    }
    for (int j = 0; j <= n; j++) {
        H[0 * ld + j]  = -sp.gap * j;
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
            H[i * ld + j]  = best;
            TB[i * ld + j] = dir;
        }
    }

    AlignResult ref = align_cpu(A, B, sp, MODE_GLOBAL);

    printf("A %s\n", A.c_str());
    printf("B %s\n", B.c_str());
    printf("m %d\n", m);
    printf("n %d\n", n);
    printf("score %d\n", ref.score);
    printf("alignedA %s\n", ref.alignedA.c_str());
    printf("alignedB %s\n", ref.alignedB.c_str());
    printf("H\n");
    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) printf("%d ", H[i * ld + j]);
        printf("\n");
    }
    printf("TB\n");
    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) printf("%d ", TB[i * ld + j]);
        printf("\n");
    }
    return 0;
}
