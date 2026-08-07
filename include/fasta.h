#pragma once
#include <string>
#include <vector>

struct FastaRecord {
    std::string id;
    std::string seq;
};

// Minimal FASTA reader: no line-wrap assumptions, uppercases sequence, strips whitespace.
std::vector<FastaRecord> read_fasta(const std::string &path);
