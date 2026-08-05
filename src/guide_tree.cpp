#include "guide_tree.h"

// Standard UPGMA: at each step merge the two closest active clusters, and set
// the new cluster's distance to every remaining cluster as the size-weighted
// average of its children's distances. O(n^3), fine at the sequence counts
// this workshop is aligning (tens to low hundreds), not meant to scale further.
std::vector<GuideTreeNode> build_upgma(const std::vector<std::vector<double>> &dist, int n) {
    std::vector<GuideTreeNode> tree;
    tree.reserve(2 * n - 1);
    for (int i = 0; i < n; i++) {
        GuideTreeNode leafNode;
        leafNode.leaf = i;
        tree.push_back(leafNode);
    }

    std::vector<int> active(n);
    for (int i = 0; i < n; i++) active[i] = i;
    std::vector<int> csize(n, 1);
    std::vector<std::vector<double>> cur = dist;  // indexed by position within `active`

    while (active.size() > 1) {
        int m = (int)active.size();
        int bi = 0, bj = 1;
        double best = cur[0][1];
        for (int p = 0; p < m; p++)
            for (int q = p + 1; q < m; q++)
                if (cur[p][q] < best) { best = cur[p][q]; bi = p; bj = q; }

        GuideTreeNode node;
        node.left = active[bi];
        node.right = active[bj];
        node.height = best / 2.0;
        int newIdx = (int)tree.size();
        tree.push_back(node);

        int sizeI = csize[bi], sizeJ = csize[bj];
        int newSize = sizeI + sizeJ;

        std::vector<int> keepPos;
        for (int p = 0; p < m; p++) if (p != bi && p != bj) keepPos.push_back(p);

        std::vector<int> newActive, newSizeVec;
        std::vector<std::vector<double>> newCur(keepPos.size() + 1, std::vector<double>(keepPos.size() + 1, 0.0));
        for (size_t a = 0; a < keepPos.size(); a++) {
            newActive.push_back(active[keepPos[a]]);
            newSizeVec.push_back(csize[keepPos[a]]);
            for (size_t b = 0; b < keepPos.size(); b++)
                newCur[a][b] = cur[keepPos[a]][keepPos[b]];
        }
        int newPos = (int)keepPos.size();
        newActive.push_back(newIdx);
        newSizeVec.push_back(newSize);
        for (size_t a = 0; a < keepPos.size(); a++) {
            int p = keepPos[a];
            double d = (sizeI * cur[bi][p] + sizeJ * cur[bj][p]) / (double)newSize;
            newCur[a][newPos] = d;
            newCur[newPos][a] = d;
        }

        active = newActive;
        csize = newSizeVec;
        cur = newCur;
    }

    return tree;
}

static void print_rec(const std::vector<GuideTreeNode> &tree, int idx, const std::vector<std::string> &labels, std::string &out) {
    const GuideTreeNode &node = tree[idx];
    if (node.leaf >= 0) { out += labels[node.leaf]; return; }
    out += "(";
    print_rec(tree, node.left, labels, out);
    out += ",";
    print_rec(tree, node.right, labels, out);
    out += ")";
}

std::string print_tree(const std::vector<GuideTreeNode> &tree, const std::vector<std::string> &labels) {
    std::string out;
    print_rec(tree, (int)tree.size() - 1, labels, out);
    out += ";";
    return out;
}
