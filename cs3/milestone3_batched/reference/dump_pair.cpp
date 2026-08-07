// Generalized version of milestone1's dump_reference: takes a fasta file and
// two sequence indices instead of a hardcoded pair, so it can verify any real
// pair from the dataset the generalized (runtime-uploaded) CS-3 kernel is
// meant to handle. Same align_cpu()-based recurrence and output format.
#include "../../../include/align.h"
#include "../../../include/fasta.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s fasta.fa idxA idxB\n", argv[0]);
        return 1;
    }
    std::vector<FastaRecord> records = read_fasta(argv[1]);
    int idxA = atoi(argv[2]), idxB = atoi(argv[3]);
    const std::string &A = records[idxA].seq;
    const std::string &B = records[idxB].seq;
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
