#pragma once
#include "guide_tree.h"
#include "align.h"
#include <vector>
#include <string>

struct Profile {
    std::vector<int> leafIds;       // original sequence indices, in row order
    std::vector<std::string> rows;  // rows.size() == leafIds.size(), all rows same length
};

// Progressive profile-profile merge along a UPGMA guide tree: at each internal
// node, align the two child profiles using sum-of-pairs column scoring (same
// diag/up/left DP recurrence as the pairwise kernel, generalized from "score
// one char pair" to "score every cross pair between two columns"). Global
// alignment throughout -- a merge step must place every residue somewhere.
//
// Runs on the host: profile depth here is only the sequence count (tens at
// workshop scale), so this is not the bottleneck. The GPU-parallel part of
// the pipeline is the O(n^2) all-pairs pairwise stage that produced the tree.
Profile progressive_align(const std::vector<GuideTreeNode> &tree,
                           const std::vector<std::string> &rawSeqs,
                           const ScoreParams &sp);
