#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import time
import os
import argparse
import glob
from seqdesign import hyper_conv_auto as model
from seqdesign import helper
from seqdesign import aws_utils
from seqdesign.version import VERSION


def main():
    parser = argparse.ArgumentParser(description="Generate novel sequences sampled from the model.")
    parser.add_argument("--sess", type=str, required=True, help="Session name for restoring a model.")
    parser.add_argument("--checkpoint", type=int, default=None, metavar='CKPT', help="Checkpoint step number.")
    parser.add_argument("--r-seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--temp", type=float, default=1.0, help="Generation temperature.")
    parser.add_argument("--batch-size", type=int, default=500, help="Number of sequences per generation batch.")
    parser.add_argument("--num-batches", type=int, default=1000000, help="Number of batches to generate.")
    parser.add_argument("--input-seq", type=str, default='default', help="Path to file with starting sequence.")
    parser.add_argument("--output-prefix", type=str, default='nanobody', help="Prefix for output fasta file.")
    parser.add_argument("--s3-path", type=str, default='', help="Base s3:// path (leave blank to disable syncing).")
    parser.add_argument("--s3-project", type=str, default=VERSION, help="Project name (subfolder of s3-path).")

    args = parser.parse_args()

    aws_util = aws_utils.AWSUtility(s3_project=args.s3_project, s3_base_path=args.s3_path) if args.s3_path else None

    working_dir = "."

    data_helper = helper.DataHelperDoubleWeightingNanobody(
        working_dir=working_dir,
        alignment_file='',
    )

    # Variables for runtime modification
    sess_name = args.sess
    batch_size = args.batch_size
    num_batches = args.num_batches
    temp = args.temp
    r_seed = args.r_seed

    print(r_seed)

    np.random.seed(r_seed)

    alphabet_list = list(data_helper.alphabet)

    os.makedirs(os.path.join(working_dir, 'generate_sequences', 'generated'), exist_ok=True)
    output_filename = (
        f"{working_dir}/generate_sequences/generated/"
        f"{args.output_prefix}_start-{args.input_seq.split('/')[-1].split('.')[0]}"
        f"_temp-{temp}_param-{sess_name}_ckpt-{args.checkpoint}_rseed-{r_seed}.fa"
    )
    OUTPUT = open(output_filename, "w")
    OUTPUT.close()

    # Provide the starting sequence to use for generation
    if args.input_seq != 'default':
        if not os.path.exists(args.input_seq) and aws_util:
            if '/' not in args.input_seq:
                args.input_seq = f'{working_dir}/generate_sequences/input/{args.input_seq}'
            aws_util.s3_get_file_grep(
                'generate_sequences/input',
                f'{working_dir}/generate_sequences/input',
                f"{args.input_seq.rsplit('/', 1)[-1]}"
            )
        with open(args.input_seq) as f:
            input_seq = f.read()
        input_seq = "*" + input_seq.strip()
    else:
        input_seq = "*EVQLVESGGGLVQAGGSLRLSCAASGFTFSSYAMGWYRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYC"

    if args.checkpoint is None:  # look for old-style flat session file structure
        glob_path = f"{working_dir}/sess/{sess_name}*"
        grep_path = f'{sess_name}.*'
        sess_namedir = f"{working_dir}/sess/{sess_name}"
    else:  # look for new folder-based session file structure
        glob_path = f"{working_dir}/sess/{sess_name}/{sess_name}.ckpt-{args.checkpoint}*"
        grep_path = f'{sess_name}.ckpt-{args.checkpoint}.*'
        sess_namedir = f"{working_dir}/sess/{sess_name}/{sess_name}.ckpt-{args.checkpoint}"

    if not glob.glob(glob_path) and aws_util:
        if not aws_util.s3_get_file_grep(
            f'sess/{sess_name}',
            f'{working_dir}/sess/{sess_name}',
            grep_path,
        ):
            raise Exception("Could not download session files from S3.")

    legacy_verison = model.AutoregressiveFR.get_checkpoint_legacy_version(sess_namedir)
    dims = {'alphabet': len(data_helper.alphabet)}
    conv_model = model.AutoregressiveFR(dims=dims, legacy_version=legacy_verison)

    params = tf.trainable_variables()
    p_counts = [np.prod(v.get_shape().as_list()) for v in params]
    p_total = sum(p_counts)
    print(f"Total parameter number:{p_total}\n")

    saver = tf.train.Saver()

    # with tf.Session(config=cfg) as sess:
    with tf.Session() as sess:

        # Initialization
        print("Initializing variables")
        init = tf.global_variables_initializer()
        sess.run(init)

        saver.restore(sess, sess_namedir)
        print("Loaded parameters")

        # Run optimization
        for i in range(num_batches):

            complete = False
            start = time.time()

            input_seq_list = batch_size * [input_seq]
            one_hot_seqs_f = data_helper.seq_list_to_one_hot(input_seq_list)
            one_hot_seq_mask = np.sum(one_hot_seqs_f, axis=-1, keepdims=True)

            # if the sequence is complete, set the value to zero,
            #   otherwise they should all be ones
            completed_seq_list = batch_size * [1.]
            decoding_steps = 0

            while not complete and decoding_steps < 50:

                feed_dict = {
                    conv_model.placeholders["sequences_start_f"]: one_hot_seqs_f,
                    conv_model.placeholders["sequences_stop_f"]: one_hot_seqs_f,
                    conv_model.placeholders["mask_decoder_1D_f"]: one_hot_seq_mask,
                    conv_model.placeholders["Neff_f"]: [1.],
                    conv_model.placeholders["sequences_start_r"]: one_hot_seqs_f,
                    conv_model.placeholders["sequences_stop_r"]: one_hot_seqs_f,
                    conv_model.placeholders["mask_decoder_1D_r"]: one_hot_seq_mask,
                    conv_model.placeholders["Neff_r"]: [1.],
                    conv_model.placeholders["step"]: [10.],
                    conv_model.placeholders["dropout"]: 1.0,
                }

                seq_logits_f = sess.run([conv_model.tensors["sequence_logits_f"]], feed_dict=feed_dict)[0]

                # slice off the last element of the list
                output_logits = seq_logits_f[:, :, -1] * temp

                # Safe exponents
                exp_output_logits = np.exp(output_logits)

                # Convert it to probabilities
                output_probs = exp_output_logits / np.sum(exp_output_logits, axis=-1, keepdims=True)

                # sample the sequences accordingly
                batch_aa_list = []

                for idx_batch in range(batch_size):
                    new_aa = np.random.choice(alphabet_list, 1, p=output_probs[idx_batch].flatten())[0]
                    input_seq_list[idx_batch] += new_aa

                    if new_aa == "*":
                        completed_seq_list[idx_batch] = 0.

                    batch_aa_list.append([new_aa])

                batch_one_hots = data_helper.seq_list_to_one_hot(batch_aa_list)
                batch_mask = np.reshape(np.asarray(completed_seq_list), [batch_size, 1, 1, 1])

                one_hot_seqs_f = np.concatenate([one_hot_seqs_f, batch_one_hots], axis=2)
                one_hot_seq_mask = np.concatenate([one_hot_seq_mask, batch_mask], axis=2)

                decoding_steps += 1

                if np.sum(completed_seq_list) == 0.0:
                    print("completed!")
                    complete = True

            OUTPUT = open(output_filename, "a")
            for idx_seq in range(batch_size):
                batch_seq = input_seq_list[idx_seq]
                out_seq = ""
                end_seq = False
                for idx_aa, aa in enumerate(batch_seq):
                    if idx_aa != 0:
                        if end_seq is False:
                            out_seq += aa
                        if aa == "*":
                            end_seq = True
                OUTPUT.write(f">{int(batch_size*i+idx_seq)}\n{out_seq}\n")
            OUTPUT.close()
            print(f"Batch {i+1} done in {time.time()-start} s")

    if aws_util:
        aws_util.s3_cp(
            local_file=output_filename,
            s3_file=f'generate_sequences/generated/{output_filename.rsplit("/", 1)[1]}',
            destination='s3'
        )


if __name__ == "__main__":
    main()
