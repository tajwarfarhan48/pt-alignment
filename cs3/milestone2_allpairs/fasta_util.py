"""Minimal FASTA reader, mirroring src/fasta.cpp's behavior (no line-wrap
assumptions, uppercases sequence, strips whitespace) -- kept independent of
the C++ reader since the Python driver needs sequences in-process for
memcpy_h2d, not just for shelling out to reference tools.
"""


def read_fasta(path):
    records = []
    header = None
    seq_parts = []

    def flush():
        if header is not None:
            records.append((header, "".join(seq_parts).upper()))

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    flush()
    return records
