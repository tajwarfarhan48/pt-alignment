#include "progressive.h"
#include <vector>
#include <algorithm>

static long long column_score(const std::vector<char> &colA, const std::vector<char> &colB, const ScoreParams &sp) {
    long long total = 0;
    for (char a : colA) {
        for (char b : colB) {
            if (a == '-' && b == '-') continue;  // both already gaps from earlier merges: no new cost
            total += (a == b) ? sp.match : sp.mismatch;
        }
    }
    return total;
}

static std::vector<char> get_column(const Profile &p, int col) {
    std::vector<char> c(p.rows.size());
    for (size_t r = 0; r < p.rows.size(); r++) c[r] = p.rows[r][col];
    return c;
}

// Sum-of-pairs profile-profile alignment: a full-column gap step costs
// depthA*depthB*gap (every row in the other profile effectively gets a fresh
// gap against every row here), which is the direct generalization of the
// single-sequence case (depth 1x1) back to the original gap penalty.
static Profile align_profiles(const Profile &A, const Profile &B, const ScoreParams &sp) {
    int m = (int)A.rows[0].size();
    int n = (int)B.rows[0].size();
    int depthA = (int)A.rows.size(), depthB = (int)B.rows.size();
    long long gapCost = (long long)depthA * depthB * sp.gap;

    std::vector<std::vector<long long>> H(m + 1, std::vector<long long>(n + 1));
    std::vector<std::vector<uint8_t>> TB(m + 1, std::vector<uint8_t>(n + 1, TB_STOP));
    for (int i = 0; i <= m; i++) { H[i][0] = -gapCost * i; TB[i][0] = (i == 0) ? TB_STOP : TB_UP; }
    for (int j = 0; j <= n; j++) { H[0][j] = -gapCost * j; TB[0][j] = (j == 0) ? TB_STOP : TB_LEFT; }

    for (int i = 1; i <= m; i++) {
        std::vector<char> colA = get_column(A, i - 1);
        for (int j = 1; j <= n; j++) {
            std::vector<char> colB = get_column(B, j - 1);
            long long diag = H[i - 1][j - 1] + column_score(colA, colB, sp);
            long long up   = H[i - 1][j] - gapCost;
            long long left = H[i][j - 1] - gapCost;
            long long best = diag; uint8_t dir = TB_DIAG;
            if (up   > best) { best = up;   dir = TB_UP; }
            if (left > best) { best = left; dir = TB_LEFT; }
            H[i][j]  = best;
            TB[i][j] = dir;
        }
    }

    std::vector<std::string> outA(depthA), outB(depthB);
    int i = m, j = n;
    while (i > 0 || j > 0) {
        uint8_t dir;
        if (i > 0 && j > 0) dir = TB[i][j];
        else if (i == 0)    dir = TB_LEFT;
        else                dir = TB_UP;

        if (dir == TB_DIAG) {
            for (int r = 0; r < depthA; r++) outA[r] += A.rows[r][i - 1];
            for (int r = 0; r < depthB; r++) outB[r] += B.rows[r][j - 1];
            i--; j--;
        } else if (dir == TB_UP) {
            for (int r = 0; r < depthA; r++) outA[r] += A.rows[r][i - 1];
            for (int r = 0; r < depthB; r++) outB[r] += '-';
            i--;
        } else {
            for (int r = 0; r < depthA; r++) outA[r] += '-';
            for (int r = 0; r < depthB; r++) outB[r] += B.rows[r][j - 1];
            j--;
        }
    }
    for (auto &s : outA) std::reverse(s.begin(), s.end());
    for (auto &s : outB) std::reverse(s.begin(), s.end());

    Profile merged;
    merged.leafIds = A.leafIds;
    merged.leafIds.insert(merged.leafIds.end(), B.leafIds.begin(), B.leafIds.end());
    merged.rows = outA;
    merged.rows.insert(merged.rows.end(), outB.begin(), outB.end());
    return merged;
}

static Profile build(const std::vector<GuideTreeNode> &tree, int idx, const std::vector<std::string> &rawSeqs, const ScoreParams &sp) {
    const GuideTreeNode &node = tree[idx];
    if (node.leaf >= 0) {
        Profile p;
        p.leafIds = {node.leaf};
        p.rows = {rawSeqs[node.leaf]};
        return p;
    }
    Profile left  = build(tree, node.left,  rawSeqs, sp);
    Profile right = build(tree, node.right, rawSeqs, sp);
    return align_profiles(left, right, sp);
}

Profile progressive_align(const std::vector<GuideTreeNode> &tree, const std::vector<std::string> &rawSeqs, const ScoreParams &sp) {
    return build(tree, (int)tree.size() - 1, rawSeqs, sp);
}
