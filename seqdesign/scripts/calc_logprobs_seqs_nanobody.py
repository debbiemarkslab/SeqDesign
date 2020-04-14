#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

import time
import sys, os

from seqdesign import hyper_conv_auto as model
from seqdesign import helper


def main(input_filename, output_filename):
    tf.logging.set_verbosity(tf.logging.ERROR)

    data_helper = helper.DataHelperDoubleWeightingNanobody()

    # Variables for runtime modification
    minibatch_size = 100
    alphabet_list = list(data_helper.alphabet)

    sess_name = "nanobody.ckpt-250000"

    dims = {}
    conv_model = model.AutoregressiveFR(dims=dims)

    params = tf.trainable_variables()
    p_counts = [np.prod(v.get_shape().as_list()) for v in params]
    p_total = sum(p_counts)
    print("Total parameter number:", p_total, "\n")

    saver = tf.train.Saver()

    # with tf.Session(config=cfg) as sess:
    with tf.Session() as sess:

        # Initialization
        print("Initializing variables")
        init = tf.global_variables_initializer()
        sess.run(init)

        sess_namedir = "./sess/" + sess_name
        saver.restore(sess, sess_namedir)
        print("Loaded parameters.")

        data_helper.read_in_test_data(input_filename)
        print("Read in test data.")

        data_helper.output_log_probs(
            sess, conv_model, output_filename, minibatch_size=minibatch_size
        )
        print("Done!")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("python calc_logprobs_seq_nanobody.py <input_filename> <output_filename>")