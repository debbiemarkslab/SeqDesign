#!/usr/bin/env python3
from seqdesign import birch
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Make a feature matrix for BIRCH clustering from a fasta file.")
    parser.add_argument("--input", type=str, required=True,
                        help="Input feature matrix csv.")
    parser.add_argument("--output-prefix", type=str, default=None,
                        help="Output cluster prefix.")
    parser.add_argument("--threshold", type=float, default=0.575, help="Birch threshold.")
    parser.add_argument("--branching-factor", type=int, default=1000, help="Birch branching factor.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Birch batch size.")
    parser.add_argument("--r-seed", type=int, default=42, help="Birch branching factor.")
    args = parser.parse_args()

    data_helper = birch.NanobodyDataBirchCluster(
        input_filename=args.input,
        minibatch_size=args.batch_size,
        r_seed=args.r_seed)

    birch_inst = birch.BirchIter(threshold=args.threshold, branching_factor=args.branching_factor)
    birch_inst.fit(data_helper)

    if args.output_prefix is None:
        os.makedirs('clusters')
        args.output_prefix = args.input.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    output_name = (
        f"clusters/{args.output_prefix}_birch_thresh-{args.threshold}_branch-{args.branching_factor}_"
        f"num_clusters-{birch_inst.num_clusters}.csv"
    )

    print("\nPREDICTING LABELS\n")
    birch_inst.predict(data_helper, minibatch_size=args.batch_size, output_name=output_name)


if __name__ == "__main__":
    main()
