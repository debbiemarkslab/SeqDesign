#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import argparse
import time
import sys, os

from seqdesign import hyper_conv_auto as model
from seqdesign import helper


def main():
    parser = argparse.ArgumentParser(description="Calculate the log probability of mutated sequences.")
    parser.add_argument("--channels", type=int, default=48, help="Number of channels.")
    parser.add_argument("--r_seed", type=int, default=-1, help="Random seed.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of iterations to run the model.")
    parser.add_argument("--minibatch_size", type=int, default=100, help="Minibatch size for inferring effect prediction.")
    parser.add_argument("--dropout_p", type=float, default=1., help="Dropout p while sampling log p(x).")
    parser.add_argument("--sess", type=str, default='', help="Session name for restoring a model to continue training.", required=True)
    parser.add_argument("--input", type=str, default='',  help="Directory and filename of the input data.", required=True)
    parser.add_argument("--output", type=str, default='',  help="Directory and filename of the outout data.", required=True)

    ARGS = parser.parse_args()

    data_helper = helper.DataHelperSingleFamily()

    # Variables for runtime modification
    minibatch_size = ARGS.minibatch_size
    alphabet_list = list(data_helper.alphabet)

    sess_name = ARGS.sess
    input_filename = ARGS.input
    output_filename = ARGS.output

    conv_model = model.AutoregressiveFR(dims={}, channels=ARGS.channels)

    params = tf.trainable_variables()
    p_counts = [np.prod(v.get_shape().as_list()) for v in params]
    p_total = sum(p_counts)
    print "Total parameter number:",p_total,"\n"

    saver = tf.train.Saver()

    #with tf.Session(config=cfg) as sess:
    with tf.Session() as sess:

        # Initialization
        print 'Initializing variables'
        init = tf.global_variables_initializer()
        sess.run(init)

        sess_namedir = "../sess/"+sess_name
        saver.restore(sess, sess_namedir)
        print "Loaded parameters."

        data_helper.read_in_test_data(input_filename)
        print "Read in test data."

        data_helper.output_log_probs(sess, conv_model, output_filename,
            ARGS.num_samples, ARGS.dropout_p, ARGS.r_seed,
            ARGS.channels, minibatch_size=minibatch_size)
        print "Done!"


if __name__ == "__main__":
    main()
