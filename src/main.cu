#include "align.h"
#include "fasta.h"
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cstring>

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

int main(int argc, char **argv) {
    std::string seqA, seqB;
    std::string fastaPath;
    int idxA = 0, idxB = 1;
    std::string modeStr = "all";
    ScoreParams sp;
    bool doBenchmark = false;
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
        else if (a == "--show") showAlignment = true;
        else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " [--seqA S --seqB S | --fasta FILE [--idxA N --idxB M]] "
                         "[--mode global|local|semiglobal|all] "
                         "[--match N --mismatch N --gap N] [--show] [--benchmark]\n";
            return 0;
        }
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
