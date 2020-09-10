#!/usr/bin/env python3
import re
import argparse
import math

hydrophobicity_ph2 = {"L":100,
                     "I":100,
                     "F":92,
                     "W":84,
                     "V":79,
                     "M":74,
                     "C":52,
                     "Y":49,
                     "A":47,
                     "T":13,
                     "E":8,
                     "G":0,
                     "S":-7,
                     "Q":-18,
                     "D":-18,
                     "R":-26,
                     "K":-37,
                     "N":-41,
                     "H":-42,
                     "P":-46}

hydrophobicity_ph7 = {"L":100,
                     "I":99,
                     "F":97,
                     "W":97,
                     "V":76,
                     "M":74,
                     "C":63,
                     "Y":49,
                     "A":41,
                     "T":13,
                     "E":8,
                     "G":0,
                     "S":-5,
                     "Q":-10,
                     "D":-55,
                     "R":-14,
                     "K":-23,
                     "N":-28,
                     "H":-31,
                     "P":-46}

pI = {"A":6.,
                     "R":10.76,
                     "N":5.41,
                     "D":2.77,
                     "C":5.07,
                     "E":3.22,
                     "Q":5.65,
                     "G":5.97,
                     "H":7.59,
                     "I":6.02,
                     "L":5.98,
                     "K":9.74,
                     "M":5.74,
                     "F":5.48,
                     "P":6.30,
                     "S":5.68,
                     "T":5.60,
                     "W":5.89,
                     "Y":5.66,
                     "V":5.96}

molecular_weight  = {"A":89.1,
                     "R":174.2,
                     "N":132.12,
                     "D":133.11,
                     "C":121.16,
                     "E":147.13,
                     "Q":146.15,
                     "G":75.07,
                     "H":155.16,
                     "I":131.18,
                     "L":131.18,
                     "K":146.19,
                     "M":149.21,
                     "F":165.19,
                     "P":115.13,
                     "S":139.11,
                     "T":119.12,
                     "W":204.23,
                     "Y":181.19,
                     "V":117.15}

alphabet = 'ADEFGHIKLNPQRSTVWY'  # C and M excluded

kmer_list = [aa for aa in alphabet]
for aa in alphabet:
    for bb in alphabet:
        kmer_list.append(aa+bb)
        for cc in alphabet:
            kmer_list.append(aa+bb+cc)

kmer_to_idx = {aa: i for i, aa in enumerate(kmer_list)}


def make_feature_matrix(input_fa, output_csv):
    header_list = ['length', 'hydrophobicity_ph2', 'hydrophobicity_ph7', 'pI', 'mw'] + kmer_list

    def calc_feature_matrix(name, seq):
        feature_list = [
            name,
            len(seq),
            math.fsum(hydrophobicity_ph2[aa] for aa in seq),
            math.fsum(hydrophobicity_ph7[aa] for aa in seq),
            math.fsum(pI[aa] for aa in seq),
            math.fsum(molecular_weight[aa] for aa in seq)
        ]
        # feature_list += [seq.count(kmer) for kmer in kmer_list]
        feature_list += [len(re.findall(kmer, seq)) for kmer in kmer_list]
        return feature_list

    with open(input_fa, 'r') as INPUT, open(output_csv, 'w') as OUTPUT:
        OUTPUT.write(",".join(header_list) + '\n')
        seq_i = -1
        for i, line in enumerate(INPUT):
            if i > -1:
                line = line.rstrip()
                if line[0] == '>':
                    name = line[1:]

                else:
                    seq_i += 1
                    if seq_i % 10000 == 0:
                        print(seq_i, flush=True)
                    fl = calc_feature_matrix(name, line)
                    OUTPUT.write(",".join([str(val) for val in fl]) + '\n')


def make_birch_hash_matrix(input_fa, output_csv):
    header_list = ['name', 'length', 'hydrophobicity_ph2', 'hydrophobicity_ph7', 'pI', 'mw', 'kmers_idx']

    def calc_feature_matrix(name, seq):
        if i % 10000 == 0:
            print(i, flush=True)

        feature_list = [
            name,
            len(seq),
            math.fsum([hydrophobicity_ph2[aa] for aa in seq]),
            math.fsum([hydrophobicity_ph7[aa] for aa in seq]),
            math.fsum([pI[aa] for aa in seq]),
            math.fsum([molecular_weight[aa] for aa in seq]),
        ]
        kmer_counts = {}

        kmer_len = 1
        num_chunks = (len(seq) - kmer_len) + 1
        for idx in range(0, num_chunks):
            kmer = seq[idx:idx + kmer_len]
            if kmer in kmer_counts:
                kmer_counts[kmer] += 1
            else:
                kmer_counts[kmer] = 1

        kmer_len = 2
        num_chunks = (len(seq) - kmer_len) + 1
        for idx in range(0, num_chunks):
            kmer = seq[idx:idx + kmer_len]
            if kmer in kmer_counts:
                kmer_counts[kmer] += 1
            else:
                kmer_counts[kmer] = 1

        kmer_len = 3
        num_chunks = (len(seq) - kmer_len) + 1
        for idx in range(0, num_chunks):
            kmer = seq[idx:idx + kmer_len]
            if kmer in kmer_counts:
                kmer_counts[kmer] += 1
            else:
                kmer_counts[kmer] = 1

        del kmer_counts['']
        out_features = feature_list + [f'{key}:{val}' for key, val in kmer_counts.items()]
        return out_features

    with open(input_fa, 'r') as INPUT, open(output_csv, 'w') as OUTPUT:
        OUTPUT.write(",".join(header_list) + '\n')
        seq_i = -1
        for i, line in enumerate(INPUT):
            if i > -1:
                line = line.rstrip()
                if line[0] == '>':
                    name = line[1:]

                else:
                    seq_i += 1
                    if seq_i % 10000 == 0:
                        print(seq_i, flush=True)
                    fl = calc_feature_matrix(name, line)
                    OUTPUT.write(",".join([str(val) for val in fl]) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Make a feature matrix for BIRCH clustering from a fasta file.")
    parser.add_argument("--input", type=str, required=True, help="Input sequence fasta file.")  # '../nanobody_id80_temp-1.0_param-nanobody_18Apr18_1129PM.ckpt-250000unique_nanobodies.fa'
    parser.add_argument("--output", type=str, required=True, help="Output feature matrix csv.")  # 'nanobody_id80_temp-1.0_param-nanobody_18Apr18_1129PM.ckpt-250000unique_nanobodies_feat_matrix.csv'
    parser.add_argument("--output-type", type=str, default='hash_matrix', choices=('matrix', 'hash_matrix'), help="Output type. {matrix|hash_matrix}")
    args = parser.parse_args()

    if args.output_type == 'hash_matrix':
        make_birch_hash_matrix(args.input, args.output)
    elif args.output_type == 'matrix':
        make_feature_matrix(args.input, args.output)


