// CPU sweep tool, same CSV schema as the GPU --sweep mode in src/main.cu, for
// direct cross-backend comparison (see results/README.md). CPU has no
// device/host transfer step, so kernel_ms == wall_ms and io_ms/io_pct are
// always 0 by definition -- not a missing measurement, a real structural
// difference between backends worth showing on the I/O-share graph.
#include "align.h"
#include "fasta.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/resource.h>
#include <vector>
#include <string>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s fasta.fa n1,n2,... [out.csv]\n", argv[0]); return 1; }
    auto records = read_fasta(argv[1]);
    std::string outCsv = argc > 3 ? argv[3] : "sweep.csv";

    std::vector<int> ns;
    char *tok = strtok(argv[2], ",");
    while (tok) { ns.push_back(atoi(tok)); tok = strtok(nullptr, ","); }

    FILE *csv = fopen(outCsv.c_str(), "w");
    fprintf(csv, "n,pairs,wall_ms,kernel_ms,io_ms,io_pct,mem_used_mb\n");

    ScoreParams sp;
    for (int n : ns) {
        if (n > (int)records.size()) { fprintf(stderr, "skip n=%d (only %zu available)\n", n, records.size()); continue; }

        auto t0 = std::chrono::steady_clock::now();
        long long pairs = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                AlignResult r = align_cpu(records[i].seq, records[j].seq, sp, MODE_GLOBAL);
                pairs++;
            }
        auto t1 = std::chrono::steady_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        struct rusage ru; getrusage(RUSAGE_SELF, &ru);
        double mem_mb = ru.ru_maxrss / 1024.0; // ru_maxrss is KB on Linux

        printf("n=%d pairs=%lld wall_ms=%.1f mem_used_mb=%.1f\n", n, pairs, wall_ms, mem_mb);
        fprintf(csv, "%d,%lld,%.3f,%.3f,0,0,%.3f\n", n, pairs, wall_ms, wall_ms, mem_mb);
    }
    fclose(csv);
    printf("wrote %s\n", outCsv.c_str());
    return 0;
}
