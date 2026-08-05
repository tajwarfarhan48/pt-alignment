#include "align.h"
#include "fasta.h"
#include "guide_tree.h"
#include "progressive.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <random>
#include <cstring>
#include <chrono>

static const char *mode_name(AlignMode m) {
    switch (m) {
        case MODE_GLOBAL:     return "global (NW)";
        case MODE_LOCAL:      return "local (SW)";
        case MODE_SEMIGLOBAL: return "semiglobal";
    }
    return "?";
}

static std::string random_dna(int len, std::mt19937 &rng) {
    static const char bases[] = "ACGT";
    std::uniform_int_distribution<int> d(0, 3);
    std::string s(len, 'A');
    for (int i = 0; i < len; i++) s[i] = bases[d(rng)];
    return s;
}

static void print_alignment(const std::string &alignedA, const std::string &alignedB, size_t width = 80) {
    for (size_t off = 0; off < alignedA.size(); off += width) {
        std::cout << "  A: " << alignedA.substr(off, width) << "\n";
        std::cout << "  B: " << alignedB.substr(off, width) << "\n\n";
    }
}

// Runs CPU + both GPU variants on one sequence pair, verifies agreement, prints a report.
static bool run_one(const std::string &A, const std::string &B, const ScoreParams &sp, AlignMode mode, bool showAlignment) {
    std::cout << "=== mode: " << mode_name(mode) << "  (|A|=" << A.size() << ", |B|=" << B.size() << ") ===\n";

    AlignResult cpu    = align_cpu(A, B, sp, mode);
    AlignResult nonpt  = align_gpu_nonpt(A, B, sp, mode);
    AlignResult pt     = align_gpu_pt(A, B, sp, mode);

    bool ok = true;
    if (cpu.score != nonpt.score) { std::cout << "  MISMATCH cpu vs nonPT score: " << cpu.score << " vs " << nonpt.score << "\n"; ok = false; }
    if (cpu.score != pt.score)    { std::cout << "  MISMATCH cpu vs PT score: "    << cpu.score << " vs " << pt.score    << "\n"; ok = false; }
    if (nonpt.alignedA != pt.alignedA || nonpt.alignedB != pt.alignedB) {
        std::cout << "  MISMATCH nonPT vs PT alignment strings differ\n"; ok = false;
    }

    std::cout << "  score: " << cpu.score << (ok ? "  [verified: CPU == nonPT == PT]" : "  [VERIFICATION FAILED]") << "\n";
    std::cout << "  diagonals: " << nonpt.num_diagonals
               << "   nonPT: " << nonpt.num_launches << " launches, " << nonpt.kernel_ms << " ms"
               << "   PT: " << pt.num_launches << " launch, " << pt.kernel_ms << " ms"
               << "   speedup: " << (pt.kernel_ms > 0 ? nonpt.kernel_ms / pt.kernel_ms : 0.0) << "x\n";

    if (showAlignment) print_alignment(cpu.alignedA, cpu.alignedB);
    std::cout << "\n";
    return ok;
}

static void run_benchmark(const std::vector<int> &lengths, const ScoreParams &sp) {
    std::mt19937 rng(42);
    std::cout << "length,nonpt_ms,pt_ms,speedup\n";
    for (int len : lengths) {
        std::string A = random_dna(len, rng);
        std::string B = random_dna(len, rng);
        AlignResult nonpt = align_gpu_nonpt(A, B, sp, MODE_GLOBAL);
        AlignResult pt    = align_gpu_pt(A, B, sp, MODE_GLOBAL);
        double speedup = pt.kernel_ms > 0 ? nonpt.kernel_ms / pt.kernel_ms : 0.0;
        std::cout << len << "," << nonpt.kernel_ms << "," << pt.kernel_ms << "," << speedup << "\n";
    }
}

// Day 2, step 1: all-pairs GPU alignment -- the Load-Balancing/Irregular-Parallelism
// use case (every pair is an independent, unequal-length DP problem), unlike Day
// 1's Global-Sync focus. Uses the PT path since that's the one we're trying to
// justify; nonPT numbers from --benchmark already show where it wins instead.
struct AllPairsResult {
    std::vector<std::vector<int>> score;
    std::vector<std::vector<double>> dist;
    double wall_ms;
};

static AllPairsResult compute_all_pairs(const std::vector<FastaRecord> &records, const ScoreParams &sp) {
    int n = (int)records.size();
    AllPairsResult r;
    r.score.assign(n, std::vector<int>(n, 0));
    r.dist.assign(n, std::vector<double>(n, 0.0));

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            AlignResult res = align_gpu_pt(records[i].seq, records[j].seq, sp, MODE_GLOBAL);
            r.score[i][j] = r.score[j][i] = res.score;
            // Simplified distance proxy for clustering purposes (not a calibrated
            // evolutionary distance): more negative score -> more divergent.
            r.dist[i][j] = r.dist[j][i] = -(double)res.score;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    r.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return r;
}

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

static void run_guidetree(const std::vector<FastaRecord> &records, const ScoreParams &sp) {
    int n = (int)records.size();
    std::cout << "all-pairs alignment: " << n << " sequences, " << (n * (n - 1) / 2) << " pairs\n";

    AllPairsResult ap = compute_all_pairs(records, sp);
    std::cout << "\n"; print_score_matrix(ap.score);

    std::vector<GuideTreeNode> tree = build_upgma(ap.dist, n);
    std::vector<std::string> labels;
    for (auto &r : records) labels.push_back(r.id.substr(0, 20));

    std::cout << "\nguide tree (Newick):\n" << print_tree(tree, labels) << "\n";
    std::cout << "\nall-pairs wall time: " << ap.wall_ms << " ms (" << (n * (n - 1) / 2) << " sequential PT alignments)\n";
}

// Day 2, step 3: full pipeline -- all-pairs GPU scoring -> UPGMA guide tree ->
// progressive profile merge -> final MSA.
static void run_msa(const std::vector<FastaRecord> &records, const ScoreParams &sp) {
    int n = (int)records.size();
    std::cout << "=== MSA pipeline: " << n << " sequences ===\n\n";

    AllPairsResult ap = compute_all_pairs(records, sp);
    print_score_matrix(ap.score);
    std::cout << "all-pairs (GPU, PT): " << ap.wall_ms << " ms\n\n";

    std::vector<GuideTreeNode> tree = build_upgma(ap.dist, n);
    std::vector<std::string> labels;
    for (auto &r : records) labels.push_back(r.id.substr(0, 20));
    std::cout << "guide tree: " << print_tree(tree, labels) << "\n\n";

    std::vector<std::string> raw;
    for (auto &r : records) raw.push_back(r.seq);

    auto t0 = std::chrono::steady_clock::now();
    Profile msa = progressive_align(tree, raw, sp);
    auto t1 = std::chrono::steady_clock::now();
    double merge_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "final MSA (" << msa.rows.size() << " sequences x " << msa.rows[0].size() << " columns):\n";
    size_t maxLabelLen = 0;
    for (auto &l : labels) maxLabelLen = std::max(maxLabelLen, l.size());
    for (size_t r = 0; r < msa.rows.size(); r++) {
        std::string label = labels[msa.leafIds[r]];
        std::cout << std::left << std::setw((int)maxLabelLen + 2) << label << msa.rows[r] << "\n";
    }
    std::cout << "\nprogressive merge (CPU, " << (n - 1) << " profile-profile alignments): " << merge_ms << " ms\n";
}

int main(int argc, char **argv) {
    std::string seqA, seqB;
    std::string fastaPath;
    int idxA = 0, idxB = 1;
    std::string modeStr = "all";
    ScoreParams sp;
    bool doBenchmark = false;
    bool doGuideTree = false;
    bool doMsa = false;
    std::vector<int> benchLengths = {100, 500, 1000, 2000, 5000, 10000};
    bool showAlignment = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--seqA") seqA = next();
        else if (a == "--seqB") seqB = next();
        else if (a == "--fasta") fastaPath = next();
        else if (a == "--idxA") idxA = std::stoi(next());
        else if (a == "--idxB") idxB = std::stoi(next());
        else if (a == "--mode") modeStr = next();
        else if (a == "--match") sp.match = std::stoi(next());
        else if (a == "--mismatch") sp.mismatch = std::stoi(next());
        else if (a == "--gap") sp.gap = std::stoi(next());
        else if (a == "--benchmark") doBenchmark = true;
        else if (a == "--guidetree") doGuideTree = true;
        else if (a == "--msa") doMsa = true;
        else if (a == "--show") showAlignment = true;
        else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " [--seqA S --seqB S | --fasta FILE [--idxA N --idxB M | --guidetree | --msa]] "
                         "[--mode global|local|semiglobal|all] "
                         "[--match N --mismatch N --gap N] [--show] [--benchmark]\n";
            return 0;
        }
    }

    if (doGuideTree) {
        if (fastaPath.empty()) { std::cerr << "--guidetree requires --fasta FILE\n"; return 1; }
        run_guidetree(read_fasta(fastaPath), sp);
        return 0;
    }
    if (doMsa) {
        if (fastaPath.empty()) { std::cerr << "--msa requires --fasta FILE\n"; return 1; }
        run_msa(read_fasta(fastaPath), sp);
        return 0;
    }

    if (!fastaPath.empty()) {
        std::vector<FastaRecord> records = read_fasta(fastaPath);
        if (idxA < 0 || idxA >= (int)records.size() || idxB < 0 || idxB >= (int)records.size()) {
            std::cerr << "idxA/idxB out of range for " << fastaPath << " (" << records.size() << " records)\n";
            return 1;
        }
        seqA = records[idxA].seq;
        seqB = records[idxB].seq;
        std::cout << "loaded from " << fastaPath << ":\n"
                   << "  A[" << idxA << "] " << records[idxA].id << " (" << seqA.size() << " bp)\n"
                   << "  B[" << idxB << "] " << records[idxB].id << " (" << seqB.size() << " bp)\n\n";
    }

    if (doBenchmark) {
        run_benchmark(benchLengths, sp);
        return 0;
    }

    if (seqA.empty() || seqB.empty()) {
        // Built-in smoke-test pair when nothing is supplied on the command line.
        seqA = "GATTACAGATTACACCGTGCATCGATCGATCGATCGGGCATTACA";
        seqB = "GATTACAGGATTACAGCGTGCATCGATCGATCGGCATTACA";
        showAlignment = true;
        std::cout << "(no --seqA/--seqB given, using built-in test sequences)\n\n";
    }

    std::vector<AlignMode> modes;
    if (modeStr == "all") modes = {MODE_GLOBAL, MODE_LOCAL, MODE_SEMIGLOBAL};
    else if (modeStr == "global") modes = {MODE_GLOBAL};
    else if (modeStr == "local") modes = {MODE_LOCAL};
    else if (modeStr == "semiglobal") modes = {MODE_SEMIGLOBAL};
    else { std::cerr << "unknown --mode " << modeStr << "\n"; return 1; }

    bool allOk = true;
    for (AlignMode m : modes) allOk &= run_one(seqA, seqB, sp, m, showAlignment);
    return allOk ? 0 : 1;
}
