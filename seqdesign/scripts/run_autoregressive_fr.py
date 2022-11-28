#!/usr/bin/env python3
import argparse
import os
import time
from datetime import timedelta
import sys
import tensorflow as tf
import numpy as np
from seqdesign.version import VERSION
from seqdesign import hyper_conv_auto as model
from seqdesign import helper
from seqdesign import utils
from seqdesign import aws_utils


def main(working_dir='.'):
    start_run_time = time.time()

    parser = argparse.ArgumentParser(description="Train an autoregressive model on a collection of sequences.")
    parser.add_argument("--s3-path", type=str, default='',
                        help="Base s3:// path (leave blank to disable syncing).")
    parser.add_argument("--s3-project", type=str, default=VERSION, metavar='P',
                        help="Project name (subfolder of s3-path).")
    parser.add_argument("--run-name-prefix", type=str, default=None, metavar='P',
                        help="Prefix for run name.")
    parser.add_argument("--channels", type=int, default=48, metavar='C',
                        help="Number of channels.")
    parser.add_argument("--num-iterations", type=int, default=250005, metavar='N',
                        help="Number of iterations to run the model.")
    parser.add_argument("--snapshot-interval", type=int, default=None, metavar='N',
                        help="Take a snapshot every N iterations.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name for fitting model. Alignment weights must be computed beforehand.")
    parser.add_argument("--restore", type=str, default='',
                        help="Session name for restoring a model to continue training.")
    parser.add_argument("--gpu", type=str, default='',
                        help="Which gpu to use. Usually  0, 1, 2, etc...")
    parser.add_argument("--r-seed", type=int, default=42, metavar='RSEED',
                        help="Random seed for parameter initialization")
    parser.add_argument("--alphabet-type", type=str, default='protein', metavar='T',
                        help="Type of data to model. Options = [protein, DNA, RNA]")
    ARGS = parser.parse_args()

    if ARGS.gpu != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu
    dataset_name = ARGS.dataset.rsplit('/', 1)[-1]
    if dataset_name.endswith(".fa"):
        dataset_name = dataset_name[:-len(".fa")]
    elif dataset_name.endswith(".fasta"):
        dataset_name = dataset_name[:-len(".fasta")]

    if ARGS.restore:
        # prevent from repeating batches/seed when restoring at intermediate point
        # script is repeatable as long as restored at same step
        # assumes restore arg of *.ckpt-(int step)
        restore_ckpt = ARGS.restore.split('.ckpt-')[-1]
        first_step = int(restore_ckpt)
        r_seed = ARGS.r_seed + first_step
        r_seed = r_seed % (2 ** 32 - 1)  # limit of np.random.seed
    else:
        first_step = 0
        r_seed = ARGS.r_seed

    print(ARGS.restore)

    if ARGS.restore == '':
        folder_time = (
            f"{dataset_name}_{ARGS.s3_project}_channels-{ARGS.channels}"
            f"_rseed-{ARGS.r_seed}_{time.strftime('%y%b%d_%I%M%p', time.gmtime())}"
        )
        if ARGS.run_name_prefix is not None:
            folder_time = ARGS.run_name_prefix + '_' + folder_time
    else:
        folder_time = ARGS.restore.split('/')[-1]
        folder_time = folder_time.split('.ckpt')[0]

    folder = f"{working_dir}/sess/{folder_time}"
    os.makedirs(folder, exist_ok=True)
    log_f = utils.Tee(f'{folder}/log.txt', 'a')  # log stdout to log.txt

    print(folder)
    print(ARGS)

    print("OS: ", sys.platform)
    print("Python: ", sys.version)
    print("TensorFlow: ", tf.__version__)
    print("Numpy: ", np.__version__)

    print('Using device:', ARGS.gpu)
    available_devices = utils.get_available_gpus()
    if available_devices:
        print('\t'.join(available_devices))
        print('\t'.join(utils.get_available_gpus_desc()))
        print(utils.get_cuda_version())
        print("CuDNN Version ", utils.get_cudnn_version())

    print("SeqDesign git hash:", str(utils.get_github_head_hash()))
    print()

    print("Run:", folder_time)

    tf.set_random_seed(r_seed)

    aws_util = aws_utils.AWSUtility(s3_base_path=ARGS.s3_path, s3_project=ARGS.s3_project) if ARGS.s3_path else None
    data_helper = helper.DataHelperSingleFamily(
        working_dir=working_dir, dataset=ARGS.dataset,
        r_seed=r_seed, alphabet_type=ARGS.alphabet_type,
        aws_util=aws_util,
    )
    data_helper.family_name = dataset_name

    # Variables for runtime modification
    batch_size = 30
    if ARGS.snapshot_interval is not None:
        fitness_check = ARGS.snapshot_interval
        fitness_start = 1
    else:
        fitness_check = 25000
        fitness_start = 29999
    num_iterations = int(ARGS.num_iterations)
    N_pred_iterations = 50

    plot_train = 100
    dropout_p_train = 0.5

    # fitness_check = 2
    # fitness_start = 2
    # num_iterations = 36


    # dims = {}
    dims = {"alphabet": len(data_helper.alphabet)}
    conv_model = model.AutoregressiveFR(dims=dims, channels=ARGS.channels)

    params = tf.trainable_variables()
    p_counts = [np.prod(v.get_shape().as_list()) for v in params]
    p_total = sum(p_counts)
    print(f"Total parameter number: {p_total}\n")

    saver = tf.train.Saver()
    run_metadata = tf.RunMetadata()

    with tf.Session() as sess:
        # Initialization
        print('Initializing variables')
        init = tf.global_variables_initializer()
        sess.run(init)

        print('Parameters initialized')

        if ARGS.restore != '':
            sess_namedir = f"{folder}/{ARGS.restore}"
            saver.restore(sess, sess_namedir)
            print("Parameters restored")
        else:
            # save initialization state
            save_path = saver.save(sess, f"{folder}/{folder_time}.ckpt", global_step=0)

        # Summary output
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(folder, sess.graph)
        global_step = first_step
        print('Created train writer')

        # Run optimization
        while global_step < num_iterations:
            start = time.time()

            prot_decoder_input_f, prot_decoder_output_f, prot_mask_decoder_f, \
            family_one_hot_f, Neff_f = data_helper.generate_one_family_minibatch_data(batch_size)

            prot_decoder_input_r, prot_decoder_output_r, prot_mask_decoder_r, \
            family_one_hot_r, Neff_r = data_helper.generate_one_family_minibatch_data(batch_size, reverse=True)

            data_load_time = time.time() - start

            feed_dict = {
                conv_model.placeholders["sequences_start_f"]: prot_decoder_input_f,
                conv_model.placeholders["sequences_stop_f"]: prot_decoder_output_f,
                conv_model.placeholders["mask_decoder_1D_f"]: prot_mask_decoder_f,
                conv_model.placeholders["Neff_f"]: [Neff_f],
                conv_model.placeholders["sequences_start_r"]: prot_decoder_input_r,
                conv_model.placeholders["sequences_stop_r"]: prot_decoder_output_r,
                conv_model.placeholders["mask_decoder_1D_r"]: prot_mask_decoder_r,
                conv_model.placeholders["Neff_r"]: [Neff_r],
                conv_model.placeholders["step"]: [global_step],
                conv_model.placeholders["dropout"]: dropout_p_train
            }

            summary, _, global_step, ce_loss, loss, KL_embedding_loss \
                = sess.run([merged, conv_model.opt_op, conv_model.global_step,
                            conv_model.tensors["cross_entropy_loss"], conv_model.tensors["loss"],
                            conv_model.tensors["KL_embedding_loss"]], feed_dict=feed_dict)

            print(f'{global_step:7} {time.time() - start:0.3f} {data_load_time:0.3f} '
                  f'{ce_loss:10.4f} {loss:10.4f} {KL_embedding_loss:10.4f}', flush=True)

            if global_step % plot_train == 0:
                train_writer.add_summary(summary, global_step)

            if global_step % fitness_check == 0 and global_step > fitness_start:
                save_path = saver.save(sess, f"{folder}/{folder_time}.ckpt", global_step=global_step)
                if working_dir != '.':
                    data_helper.generate_reset_script(
                        f"{folder_time}.ckpt-{global_step}", ARGS.channels, ARGS.dataset, ARGS.r_seed)
                if aws_util:
                    aws_util.s3_sync(local_folder=folder, s3_folder=f'sess/_inprogress/{folder_time}/', destination='s3')
        try:
            max_gpu_mem_used = sess.run(tf.contrib.memory_stats.MaxBytesInUse())
        except tf.errors.OpError:
            max_gpu_mem_used = None

        train_writer.close()

    print(f"Done! Total run time: {timedelta(seconds=time.time()-start_run_time)}")
    log_f.flush()
    if aws_util:
        aws_util.s3_sync(local_folder=folder, s3_folder=f'sess/{folder_time}/', destination='s3')

    if working_dir != '.':
        os.makedirs(f'{working_dir}/complete/', exist_ok=True)
        OUTPUT = open(f'{working_dir}/complete/{folder_time}.txt', 'w')
        OUTPUT.write(f"STEPS COMPLETED: {int(global_step)}")
        if max_gpu_mem_used is not None:
            OUTPUT.write(f"Max GPU memory used: {max_gpu_mem_used}")
        OUTPUT.close()


if __name__ == "__main__":
    main()
