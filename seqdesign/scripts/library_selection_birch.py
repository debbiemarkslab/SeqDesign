#!/usr/bin/env python3
from seqdesign import birch
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Make a feature matrix for BIRCH clustering from a fasta file.")
    parser.add_argument("--input", type=str, required=True,
                        help="Input feature matrix csv.")  # '/n/groups/marks/users/adam/nanobody/generation/generate_test_sequences/process_output_sequences/make_feature_matrix/nanobody_id80_temp-1.0_param-nanobody.ckpt-250000unique_nanobodies_feat_matrix.csv'
    parser.add_argument("--output-prefix", type=str, required=True,
                        help="Output cluster prefix.")  # 'nanobody_id80_temp-1.0_param-nanobody_18Apr18_1129PM.ckpt-250000unique_nanobodies_feat_matrix.csv'
    parser.add_argument("--threshold", type=float, default=0.575, help="Birch threshold.")
    parser.add_argument("--branching-factor", type=int, default=1000, help="Birch branching factor.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Birch batch size.")
    parser.add_argument("--r-seed", type=int, default=42, help="Birch branching factor.")
    args = parser.parse_args()

    data_helper = birch.NanobodyDataBirchCluster(
        input_filename=args.input,
        minibatch_size=args.minibatch_size,
        r_seed=args.r_seed)

    birch_inst = birch.BirchIter(threshold=args.threshold, branching_factor=args.branching_factor)
    birch_inst.fit(data_helper)

    output_name = (
        f"{args.output_prefix}_birch_thresh-{args.threshold}_branch-{args.branching_factor}_"
        f"num_clusters-{birch_inst.num_clusters}.csv"
    )

    print("\nPREDICTING LABELS\n")

    birch_inst.predict(data_helper, minibatch_size=args.batch_size, output_name=output_name)


if __name__ == "__main__":
    main()
