#!/usr/bin/env python
from seqdesign import birch
import sys

# Hyperparameters
threshold = 0.575
branching_factor = 1000
minibatch_size = 1000

# TODO test this script
input_filename = '/n/groups/marks/users/adam/nanobody/generation/generate_test_sequences/process_output_sequences/make_feature_matrix/nanobody_id80_temp-1.0_param-nanobody.ckpt-250000unique_nanobodies_feat_matrix.csv'
def main(input_filename):
    data_helper = birch.NanobodyDataBirchCluster(input_filename=input_filename,
        minibatch_size=minibatch_size, r_seed=43)

    birch_inst = birch.BirchIter(threshold=threshold, branching_factor=branching_factor)

    birch_inst.fit(data_helper)

    output_name = "nanobody_id80_full_birch_thresh-"+str(threshold)+"_branch-"+str(branching_factor)+"_num_clusters-"+str(birch_inst.num_clusters)+".csv"

    print "\nPREDICTING LABELS\n"

    birch_inst.predict(data_helper,minibatch_size=minibatch_size,output_name=output_name)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print "library_selection_birch <input_filename>"
