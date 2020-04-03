#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import argparse
import platform
import time
import sys, os

from seqdesign import hyper_conv_auto as model
from seqdesign import helper


def main(working_dir='.'):
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser(description="Train an autoregressive model on a collection of sequences.")
    parser.add_argument("--dataset", type=str, default=None, required=True,
                        help="Dataset name for fitting model. Alignment weights must be computed beforehand.")
    parser.add_argument("--channels", type=int, default=48,
                        help="Number of channels.")
    parser.add_argument("--num_iterations", type=int, default=250005,
                        help="Number of iterations to run the model.")
    parser.add_argument("--snapshot_interval", type=int, default=None,
                        help="Number of iterations to run the model.")
    parser.add_argument("--restore", type=str, default='',
                        help="Session name for restoring a model to continue training.")
    parser.add_argument("--r_seed", type=int, default=42,
                        help="Random seed for parameter initialization and minibatch sampling.")
    ARGS = parser.parse_args()

    tf.set_random_seed(ARGS.r_seed)

    print ARGS.restore

    data_helper = helper.DataHelperSingleFamily(working_dir=working_dir, dataset=ARGS.dataset, r_seed=ARGS.r_seed)

    # Variables for runtime modification
    batch_size = 30
    if ARGS.snapshot_interval is not None:
        fitness_check = ARGS.snapshot_interval
        fitness_start = 1
    else:
        fitness_check = 25000
        fitness_start = 29999
    num_iterations = ARGS.num_iterations
    N_pred_iterations = 50

    plot_train = 100
    dropout_p_train = 0.5

    # fitness_check = 2
    # fitness_start = 2
    # num_iterations = 36


    dims = {"alphabet":len(data_helper.alphabet)}
    conv_model = model.AutoregressiveFR(dims=dims,channels=int(ARGS.channels))

    params = tf.trainable_variables()
    p_counts = [np.prod(v.get_shape().as_list()) for v in params]
    p_total = sum(p_counts)
    print "Total parameter number:", p_total, "\n"

    saver = tf.train.Saver()

    #with tf.Session(config=cfg) as sess:
    with tf.Session() as sess:

        # Initialization
        print 'Initializing variables'
        init = tf.global_variables_initializer()
        sess.run(init)

        print 'Parameters initialized'

        if ARGS.restore == '':
            folder_time = data_helper.family_name+'_elu_channels-'+str(ARGS.channels)+'_rseed-'+str(ARGS.r_seed)+'_'+time.strftime('%y%b%d_%I%M%p', time.gmtime())

        else:
            sess_namedir = working_dir+"/sess/"+ARGS.restore
            saver.restore(sess, sess_namedir)
            folder_time = ARGS.restore.split('/')[-1]
            folder_time_list = folder_time.split('.')
            folder_time_list.pop()
            folder_time = "-".join(folder_time_list)
            print "Parameters restored"

        folder = working_dir+'/log/'+folder_time+'/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        print folder

        # Summary output
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(folder, sess.graph)
        global_step = 0
        print 'Created train writer'

        # Run optimization
        while global_step < num_iterations:
            start = time.time()

            prot_decoder_input_f, prot_decoder_output_f, prot_mask_decoder_f, \
                family_one_hot_f, Neff_f = data_helper.generate_one_family_minibatch_data(batch_size)

            prot_decoder_input_r, prot_decoder_output_r, prot_mask_decoder_r, \
                family_one_hot_r, Neff_r = data_helper.generate_one_family_minibatch_data(batch_size, reverse=True)

            feed_dict = {conv_model.placeholders["sequences_start_f"]: prot_decoder_input_f,\
                conv_model.placeholders["sequences_stop_f"]: prot_decoder_output_f,\
                conv_model.placeholders["mask_decoder_1D_f"]: prot_mask_decoder_f,\
                conv_model.placeholders["Neff_f"]:[Neff_f],\
                conv_model.placeholders["sequences_start_r"]: prot_decoder_input_r,\
                conv_model.placeholders["sequences_stop_r"]: prot_decoder_output_r,\
                conv_model.placeholders["mask_decoder_1D_r"]: prot_mask_decoder_r,\
                conv_model.placeholders["Neff_r"]:[Neff_r],\
                conv_model.placeholders["step"]:[global_step],
                conv_model.placeholders["dropout"]: dropout_p_train}

            summary, _, global_step, ce_loss,loss,KL_embedding_loss \
                = sess.run([merged, conv_model.opt_op, conv_model.global_step,  \
                conv_model.tensors["cross_entropy_loss"],conv_model.tensors["loss"],\
                conv_model.tensors["KL_embedding_loss"]],feed_dict=feed_dict)

            print global_step, time.time() - start,ce_loss,loss,KL_embedding_loss

            if global_step % plot_train == 0:
                train_writer.add_summary(summary, global_step)

            print global_step
            print type(global_step)
            print global_step%2
            if global_step % fitness_check == 0 and global_step > fitness_start:
                save_path = saver.save(sess, working_dir+"/sess/"+folder_time+".ckpt", global_step=global_step)

        train_writer.close()


if __name__ == "__main__":
    main()
