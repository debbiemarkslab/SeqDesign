#!/usr/bin/env python3
import glob
import re
import argparse


def main():
    parser = argparse.ArgumentParser(description="Preprocess generated nanobody sequences.")
    parser.add_argument("--file-prefix-in", type=str, required=True, metavar='P',
                        help="Prefix for input files (supports glob).")
    parser.add_argument("--file-prefix-out", type=str, required=True, metavar='P',
                        help="Prefix for output file.")
    parser.add_argument("--prev-sequences", type=str, required=True, metavar='P',
                        help="Location of previous nanobody sequences with labeled CDRs")
    parser.add_argument("--cdr3-start", type=int, default=97,
                        help="Start position for nanobody CDR3")
    parser.add_argument("--cdr3-end", type=int, default=11,
                        help="Length of final nanobody beta strand")
    ARGS = parser.parse_args()

    prev_cdr3s = {}
    prev_nanobody_data_file = open(ARGS.prev_sequences, "r")
    after_name = True
    after_seq = False
    seq = ""
    label = ""

    def get_cdr3_seq(seq,label):
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

    for i,line in enumerate(prev_nanobody_data_file):
        line = line.rstrip()
        if line[0] == ">":
            #print seq,label
            cdr_seq = get_cdr3_seq(seq,label)
            prev_cdr3s[cdr_seq] = ""
            name = line
            after_name = True
            after_seq = False

        elif after_name == True and after_seq == False:
            seq = line
            after_seq = True
            after_name = False

        elif after_name == False and after_seq == True:
            label = line
            after_name = False
            after_seq = False

    prev_nanobody_data_file.close()

    cdr_seq = get_cdr3_seq(seq,label)
    prev_cdr3s[cdr_seq] = ""

    all_functional_sequence_name_to_sequences = {}
    sequence_dict = {}

    num_seqs = 0
    num_valid_endings = 0
    num_unique_seqs = 0
    num_non_training_seqs = 0
    num_no_glycosylation_motifs = 0
    num_no_asparagine_deamination_motifs = 0
    num_no_sulfur_containing_amino_acids = 0
    for filename in glob.glob(ARGS.file_prefix_in):
        r_seed = filename.split("-")[-1].split(".")[0]

        INPUT = open(filename, "r")

        for line in INPUT:
            line = line.rstrip()
            if line[0] == ">":
                name = line
            else:
                nanobody_seq = line
                valid_ending = False
                # Make sure it has a valid sequence ending
                if "YWGQGTQVTVS*" in nanobody_seq:
                    nanobody_seq_list = list(nanobody_seq)

                    # get rid of the end character
                    nanobody_seq_list.pop()

                    # add an extra S at the end so it is consistent
                    nanobody_seq = "".join(nanobody_seq_list)+"S"

                    valid_ending = True


                if "YWGQGTQVTVSS*" in nanobody_seq:

                    nanobody_seq_list = list(nanobody_seq)

                    # get rid of the end character
                    nanobody_seq_list.pop()
                    nanobody_seq = "".join(nanobody_seq_list)

                    valid_ending = True

                if not valid_ending:
                    continue

                cdr = nanobody_seq[ARGS.nanobody_start-1:-ARGS.nanobody_end]
                no_sulfur_aas = False
                if "C" not in cdr and "M" not in cdr:
                    no_sulfur_aas = True

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
                    all_functional_sequence_name_to_sequences[name+"_rseed-"+str(r_seed)] = nanobody_seq

        INPUT.close()

    print("num seqs:", num_seqs)
    print("num valid endings: ", num_valid_endings)
    print("num unique seqs:", num_unique_seqs)
    print("num non-training seqs:", num_non_training_seqs)
    print("num without glycosylation motifs:", num_no_glycosylation_motifs)
    print("num without asparagine deamination motifs:", num_no_asparagine_deamination_motifs)
    print("num without sulfur containing amino acids:", num_no_sulfur_containing_amino_acids)

    print("New nanobodies:", len(all_functional_sequence_name_to_sequences))
    with open(ARGS.file_prefix_out+"_unique_nanobodies.fa", "w") as out_f:
        for name,seq in all_functional_sequence_name_to_sequences.items():
            out_f.write(name+"\n"+seq+"\n")


if __name__ == "__main__":
    main()
