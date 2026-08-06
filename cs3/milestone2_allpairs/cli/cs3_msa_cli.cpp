// Standalone CLI reusing this project's portable (non-GPU) pipeline stages --
// guide_tree.cpp and progressive.cpp, unchanged -- to turn an all-pairs score
// matrix into a guide tree and final MSA, exactly like main.cu's run_msa().
// The only thing that differs between hardware backends is *how the score
// matrix gets produced*: this tool accepts it two ways so the same tree/merge
// code can be checked against a CPU-computed reference and driven by real
// CS-3 output.
//
//   --fasta FILE --cpu-reference         score matrix computed here via align_cpu()
//   --fasta FILE --scores SCOREFILE      score matrix read from SCOREFILE (n lines
//                                        of n whitespace-separated ints; produced by
//                                        run_allpairs.py from real CS-3 runs)
//   --out OUT.fasta                      also write the final MSA as aligned FASTA
//                                        (original record IDs, full-length gapped
//                                        sequences) -- optional, console output
//                                        always happens regardless
//
// Prints the score matrix, Newick guide tree, and final MSA -- same format as
// main.cu's run_msa().

#include "align.h"
#include "fasta.h"
#include "guide_tree.h"
#include "progressive.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void print_score_matrix(const std::vector<std::vector<int>> &score) {
    int n = (int)score.size();
    std::cout << "score matrix:\n     ";
    for (int j = 0; j < n; j++) std::cout << std::setw(7) << j;
    std::cout << "\n";
    for (int i = 0; i < n; i++) {
        std::cout << std::setw(4) << i << ":";
        for (int j = 0; j < n; j++) std::cout << std::setw(7) << (i == j ? 0 : score[i][j]);
        std::cout << "\n";
    }
}

static std::vector<std::vector<int>> score_via_cpu(const std::vector<FastaRecord> &records, const ScoreParams &sp) {
    int n = (int)records.size();
    std::vector<std::vector<int>> score(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            AlignResult res = align_cpu(records[i].seq, records[j].seq, sp, MODE_GLOBAL);
            score[i][j] = score[j][i] = res.score;
        }
    return score;
}

static std::vector<std::vector<int>> score_from_file(const std::string &path, int n) {
    std::vector<std::vector<int>> score(n, std::vector<int>(n, 0));
    std::ifstream f(path);
    if (!f) { std::cerr << "cannot open score file: " << path << "\n"; exit(1); }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (!(f >> score[i][j])) { std::cerr << "score file too short\n"; exit(1); }
    return score;
}

int main(int argc, char **argv) {
    std::string fastaPath, scoresPath, outPath;
    bool cpuReference = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--fasta") fastaPath = next();
        else if (a == "--scores") scoresPath = next();
        else if (a == "--out") outPath = next();
        else if (a == "--cpu-reference") cpuReference = true;
    }
    if (fastaPath.empty() || (!cpuReference && scoresPath.empty())) {
        std::cerr << "usage: " << argv[0] << " --fasta FILE (--cpu-reference | --scores FILE)\n";
        return 1;
    }

    ScoreParams sp; // defaults: match=2, mismatch=-1, gap=2 (same as align_cpu()'s default)
    std::vector<FastaRecord> records = read_fasta(fastaPath);
    int n = (int)records.size();

    std::vector<std::vector<int>> score =
        cpuReference ? score_via_cpu(records, sp) : score_from_file(scoresPath, n);

    print_score_matrix(score);

    std::vector<std::vector<double>> dist(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (i != j) dist[i][j] = -(double)score[i][j];

    std::vector<GuideTreeNode> tree = build_upgma(dist, n);
    std::vector<std::string> labels;
    for (auto &r : records) labels.push_back(r.id.substr(0, 20));
    std::cout << "\nguide tree (Newick):\n" << print_tree(tree, labels) << "\n\n";

    std::vector<std::string> raw;
    for (auto &r : records) raw.push_back(r.seq);
    Profile msa = progressive_align(tree, raw, sp);

    std::cout << "final MSA (" << msa.rows.size() << " sequences x " << msa.rows[0].size() << " columns):\n";
    size_t maxLabelLen = 0;
    for (auto &l : labels) maxLabelLen = std::max(maxLabelLen, l.size());
    for (size_t r = 0; r < msa.rows.size(); r++) {
        std::string label = labels[msa.leafIds[r]];
        std::cout << std::left << std::setw((int)maxLabelLen + 2) << label << msa.rows[r] << "\n";
    }

    if (!outPath.empty()) {
        std::ofstream out(outPath);
        if (!out) { std::cerr << "cannot write: " << outPath << "\n"; return 1; }
        for (size_t r = 0; r < msa.rows.size(); r++) {
            out << ">" << records[msa.leafIds[r]].id << "\n" << msa.rows[r] << "\n";
        }
        std::cout << "\nwrote aligned FASTA: " << outPath << "\n";
    }
    return 0;
}
