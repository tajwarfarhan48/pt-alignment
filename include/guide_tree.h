#pragma once
#include <vector>
#include <string>

// Binary guide tree node. Leaves have leaf >= 0 and left == right == -1.
// Internal nodes have left/right pointing into the same node array.
struct GuideTreeNode {
    int left = -1, right = -1;
    int leaf = -1;
    double height = 0.0;  // UPGMA merge height (half the cluster-cluster distance)
};

// UPGMA clustering: n leaves in, 2n-1 nodes out (n leaves + n-1 internal),
// root is always the last element. dist must be an n x n symmetric matrix
// (only the upper triangle is read).
std::vector<GuideTreeNode> build_upgma(const std::vector<std::vector<double>> &dist, int n);

// Newick-ish pretty print using the given leaf labels, for inspection.
std::string print_tree(const std::vector<GuideTreeNode> &tree, const std::vector<std::string> &labels);
