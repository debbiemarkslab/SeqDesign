#!/usr/bin/env python3
import glob
import re
import argparse
from seqdesign.text_histogram import histogram, Histogram


def main():
    parser = argparse.ArgumentParser(description="Preprocess generated nanobody sequences.")
    parser.add_argument("--file-prefix-in", type=str, required=True, metavar='P',
                        help="Prefix for input files (supports glob).")
    parser.add_argument("--file-prefix-out", type=str, required=True, metavar='P',
                        help="Prefix for output file.")
    parser.add_argument("--prev-sequences", type=str, default='', metavar='P',
                        help="Location of previous nanobody sequences with labeled CDRs. "
                             "Leave blank to disable filtering of previous CDRs")
    parser.add_argument("--cdr3-start", type=int, default=97,
                        help="Start position for nanobody CDR3")
    parser.add_argument("--cdr3-end", type=int, default=11,
                        help="Length of final nanobody beta strand")
    ARGS = parser.parse_args()

    prev_cdr3s = {}
    after_name = True
    after_seq = False
    seq = ""
    label = ""

    def get_cdr3_seq(seq, label):
        label_split = label.split("C")

        cdr_count = 0
        counter = 0
        cdr_seq = ""

        for chunk in label_split:

            if chunk != "":
                cdr_count += 1

            if chunk == "":
                counter += 1
            else:
                for k in range(len(chunk)):
                    counter += 1

                    if cdr_count == 3:
                        cdr_seq += seq[counter+1]

        return cdr_seq

    if ARGS.prev_sequences:
        with open(ARGS.prev_sequences, "r") as prev_nanobody_data_file:
            for i,line in enumerate(prev_nanobody_data_file):
                line = line.rstrip()
                if line[0] == ">":
                    cdr_seq = get_cdr3_seq(seq, label)
                    prev_cdr3s[cdr_seq] = ""
                    after_name = True
                    after_seq = False

                elif after_name and not after_seq:
                    seq = line
                    after_seq = True
                    after_name = False

                elif not after_name and after_seq:
                    label = line
                    after_name = False
                    after_seq = False

    cdr_seq = get_cdr3_seq(seq, label)
    prev_cdr3s[cdr_seq] = ""

    all_functional_sequence_name_to_sequences = {}
    sequence_dict = {}
    cdr3_lengths = Histogram(minimum=0.0, maximum=40.0, buckets=20)
    cdr3_complexities_1 = Histogram(minimum=0.0, maximum=1.0, buckets=10)
    cdr3_complexities_2 = Histogram(minimum=0.0, maximum=1.0, buckets=10)

    name = ''
    num_seqs = 0
    num_valid_endings = 0
    num_unique_seqs = 0
    num_non_training_seqs = 0
    num_no_glycosylation_motifs = 0
    num_no_asparagine_deamination_motifs = 0
    num_no_sulfur_containing_amino_acids = 0

    filenames = glob.glob(ARGS.file_prefix_in)
    assert len(filenames) > 0
    for filename in filenames:
        r_seed = filename.split("-")[-1].split(".")[0]

        INPUT = open(filename, "r")

        for line in INPUT:
            line = line.rstrip()
            if line[0] == ">":
                name = line
            else:
                nanobody_seq = line
                num_seqs += 1

                # get rid of the end character
                nanobody_seq = nanobody_seq.rstrip("*")

                valid_ending = False
                # Make sure it has a valid sequence ending
                if nanobody_seq.endswith("YWGQGTQVTVS"):
                    # add an extra S at the end so it is consistent
                    nanobody_seq = nanobody_seq+"S"
                    valid_ending = True
                elif nanobody_seq.endswith("YWGQGTQVTVSS"):
                    valid_ending = True

                if not valid_ending:
                    continue

                cdr = nanobody_seq[ARGS.cdr3_start-1:-ARGS.cdr3_end]
                no_sulfur_aas = ("C" not in cdr and "M" not in cdr)

                glycosylation_motifs = re.findall("N[ACDEFGHIKLMNQRSTVWY][S|T]", nanobody_seq)
                no_glycosylation_motif = (glycosylation_motifs == [])

                asp_deamination_motif = re.findall("NG", nanobody_seq)
                no_asp_deamine_motif = (asp_deamination_motif == [])

                num_valid_endings += 1
                if nanobody_seq not in sequence_dict:
                    num_unique_seqs += 1
                    if cdr not in prev_cdr3s:
                        num_non_training_seqs += 1
                        if no_glycosylation_motif:
                            num_no_glycosylation_motifs += 1
                            if no_asp_deamine_motif:
                                num_no_asparagine_deamination_motifs += 1
                                if no_sulfur_aas:
                                    num_no_sulfur_containing_amino_acids += 1

                if no_glycosylation_motif and no_sulfur_aas and no_asp_deamine_motif and nanobody_seq not in sequence_dict and cdr not in prev_cdr3s:
                    sequence_dict[nanobody_seq] = ""
                    new_name = f"{name}_rseed-{r_seed}"
                    assert new_name not in all_functional_sequence_name_to_sequences
                    all_functional_sequence_name_to_sequences[new_name] = nanobody_seq
                    cdr3_lengths.add(len(cdr))
                    cdr3_complexities_1.add(
                        sum(cdr[i] == cdr[i+1] for i in range(len(cdr)-1)) / (len(cdr)-1)
                        if len(cdr)-1 > 0 else 0.0
                    )
                    cdr3_complexities_2.add(
                        sum(cdr[i] == cdr[i+2] for i in range(len(cdr)-2)) / (len(cdr)-2)
                        if len(cdr)-2 > 0 else 0.0
                    )

        INPUT.close()

    if len(all_functional_sequence_name_to_sequences) > 0:
        min_length = max_length = len(next(iter(all_functional_sequence_name_to_sequences.values())))
        for seq in all_functional_sequence_name_to_sequences.values():
            if len(seq) < min_length:
                min_length = len(seq)
            if len(seq) > max_length:
                max_length = len(seq)
        length_hist = histogram(
            (len(seq) for seq in all_functional_sequence_name_to_sequences.values()),
            minimum=min_length, maximum=max_length, buckets=20
        )
    else:
        length_hist = None

    output_sequences_description = f"""{ARGS.file_prefix_in} -> {ARGS.file_prefix_out}
num seqs:\t{num_seqs}
num valid endings:\t{num_valid_endings}\t({num_valid_endings/num_seqs:%})
num unique seqs:\t{num_unique_seqs}\t({num_unique_seqs/num_valid_endings:%})
num non-training seqs:\t{num_non_training_seqs}\t({num_non_training_seqs/num_unique_seqs:%})
num without glycosylation motifs:\t{num_no_glycosylation_motifs}\t({num_no_glycosylation_motifs/num_non_training_seqs:%})
num without asparagine deamination motifs:\t{num_no_asparagine_deamination_motifs}\t({num_no_asparagine_deamination_motifs/num_no_glycosylation_motifs:%})
num without sulfur containing amino acids:\t{num_no_sulfur_containing_amino_acids}\t({num_no_sulfur_containing_amino_acids/num_no_asparagine_deamination_motifs:%})
New nanobodies:\t{len(all_functional_sequence_name_to_sequences)}\t({len(all_functional_sequence_name_to_sequences)/num_seqs:%})
Length distribution:
{length_hist}
CDR3 length distribution:
{cdr3_lengths}
CDR3 low-complexity order 1 distribution:
{cdr3_complexities_1}
CDR3 low-complexity order 2 distribution:
{cdr3_complexities_2}

num_seqs	valid_endings	unique_seqs	nontraining_seqs	no_glycosylation_seqs	no_deamination_seqs	no_sulfur_seqs	num_out	invalid_endings_percent	duplicates_percent	training_duplicates_percent	glycosylation_percent	deamination_percent	sulfur_percent	postfilter_percent	length_mean	length_sd	cdr3_length_mean	cdr3_length_sd	cdr3_lowcomplexity_1_mean	cdr3_lowcomplexity_2_mean
{num_seqs}	{num_valid_endings}	{num_unique_seqs}	{num_non_training_seqs}	{num_no_glycosylation_motifs}	{num_no_asparagine_deamination_motifs}	{num_no_sulfur_containing_amino_acids}	{len(all_functional_sequence_name_to_sequences)}	{1-num_valid_endings/num_seqs:%}	{1-num_unique_seqs/num_valid_endings:%}	{1-num_non_training_seqs/num_unique_seqs:%}	{1-num_no_glycosylation_motifs/num_non_training_seqs:%}	{1-num_no_asparagine_deamination_motifs/num_no_glycosylation_motifs:%}	{1-num_no_sulfur_containing_amino_acids/num_no_asparagine_deamination_motifs:%}	{len(all_functional_sequence_name_to_sequences)/num_seqs:%}	{length_hist.mvsd_mean}	{length_hist.mvsd_sd}	{cdr3_lengths.mvsd_mean}	{cdr3_lengths.mvsd_sd}	{cdr3_complexities_1.mvsd_mean}	{cdr3_complexities_2.mvsd_mean}"""

    print(output_sequences_description)

    if ARGS.file_prefix_out != '/dev/null':
        ARGS.file_prefix_out = ARGS.file_prefix_out+"_unique_nanobodies.fa"
    with open(ARGS.file_prefix_out, "w") as out_f:
        for name,seq in all_functional_sequence_name_to_sequences.items():
            out_f.write(name+"\n"+seq+"\n")
    with open(ARGS.file_prefix_out.replace('.fa', '.txt'), 'w') as out_f:
        out_f.write(output_sequences_description)


if __name__ == "__main__":
    main()
