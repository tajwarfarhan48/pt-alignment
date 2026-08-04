#include "fasta.h"
#include <fstream>
#include <stdexcept>
#include <cctype>

std::vector<FastaRecord> read_fasta(const std::string &path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open FASTA file: " + path);

    std::vector<FastaRecord> records;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line[0] == '>') {
            records.push_back({line.substr(1), ""});
        } else if (!records.empty()) {
            for (char c : line) if (!std::isspace((unsigned char)c)) records.back().seq += (char)std::toupper(c);
        }
    }
    return records;
}
