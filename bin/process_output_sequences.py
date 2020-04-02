#!/usr/bin/env python
import glob
import re

file_prefix = "nanobody_id80*_temp-1.0_param-nanobody.ckpt-250000"
file_prefix_out = "nanobody_id80_temp-1.0_param-nanobody.ckpt-250000"


prev_cdr3s = {}
prev_nanobody_data_file = open("../datasets/Manglik_labelled_nanobodies.txt", "r")
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

for filename in glob.glob("../output/"+file_prefix+"*"):
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

            if valid_ending:
                cdr = nanobody_seq[96:-11]
                no_sulfur_aas = False
                if "C" not in cdr and "M" not in cdr:
                    no_sulfur_aas = True

                glycosylation_motifs = re.findall("N[ACDEFGHIKLMNQRSTVWY][S|T]", nanobody_seq)

		asp_deamination_motif = re.findall("NG", nanobody_seq)

                if glycosylation_motifs == []:
                    no_glycosylation_motif = True

                else:
                    no_glycosylation_motif = False

		if asp_deamination_motif == []:
		    no_asp_deamine_motif = True
		else:
		    no_asp_deamine_motif = False

                if no_glycosylation_motif and no_sulfur_aas and no_asp_deamine_motif and nanobody_seq not in sequence_dict and cdr not in prev_cdr3s:
                    sequence_dict[nanobody_seq] = ""
                    all_functional_sequence_name_to_sequences[name+"_rseed-"+str(r_seed)] = nanobody_seq

    INPUT.close()

print "New nanobodies:",len(all_functional_sequence_name_to_sequences)
OUTPUT = open(file_prefix_out+"unique_nanobodies.fa", "w")
for name,seq in all_functional_sequence_name_to_sequences.items():
    OUTPUT.write(name+"\n"+seq+"\n")

OUTPUT.close()
