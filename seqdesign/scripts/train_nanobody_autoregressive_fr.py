#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

import time
import os

from seqdesign import hyper_conv_auto as model
from seqdesign import helper

data_helper = helper.DataHelperDoubleWeightingNanobody(alignment_file='nanobodies/Manglik_filt_seq_id80_id90.fa')

# Variables for runtime modification
batch_size = 30
plot_train = 20
save_params = 50000

dropout_p_train = 0.5

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
    print('Initializing variables')
    init = tf.global_variables_initializer()
    sess.run(init)

    print('Parameters initialized')

    folder_time = time.strftime('%y%b%d_%I%M%p', time.gmtime())
    # folder_time = "18Jan09_0547AM"
    folder = 'log/' + folder_time + '/'

    if not os.path.exists(folder):
        os.makedirs(folder)

    # Summary output
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(folder, sess.graph)
    global_step = 0
    print('Created train writer')

    # Run optimization
    for i in range(100000000):
        start = time.time()

        prot_decoder_input_f, prot_decoder_output_f, prot_mask_decoder_f, \
            family_one_hot_f, Neff_f = data_helper.generate_one_family_minibatch_data(batch_size)

        prot_decoder_input_r, prot_decoder_output_r, prot_mask_decoder_r, \
            family_one_hot_r, Neff_r = data_helper.generate_one_family_minibatch_data(batch_size, reverse=True)

        feed_dict = {
            conv_model.placeholders["sequences_start_f"]: prot_decoder_input_f,
            conv_model.placeholders["sequences_stop_f"]: prot_decoder_output_f,
            conv_model.placeholders["mask_decoder_1D_f"]: prot_mask_decoder_f,
            conv_model.placeholders["Neff_f"]: [Neff_f],
            conv_model.placeholders["sequences_start_r"]: prot_decoder_input_r,
            conv_model.placeholders["sequences_stop_r"]: prot_decoder_output_r,
            conv_model.placeholders["mask_decoder_1D_r"]: prot_mask_decoder_r,
            conv_model.placeholders["Neff_r"]: [Neff_r],
            conv_model.placeholders["step"]: [i],
            conv_model.placeholders["dropout"]: dropout_p_train
        }

        summary, _, global_step, ce_loss, loss, KL_embedding_loss = sess.run(
            [merged, conv_model.opt_op, conv_model.global_step,
             conv_model.tensors["cross_entropy_loss"], conv_model.tensors["loss"],
             conv_model.tensors["KL_embedding_loss"]], feed_dict=feed_dict)

        print(i, time.time() - start, ce_loss, loss, KL_embedding_loss)

        if global_step % plot_train == 0:
            train_writer.add_summary(summary, global_step)

        if global_step % save_params == 0:
            save_path = saver.save(sess, "./sess/nanobody_" + folder_time + ".ckpt", global_step=global_step)

    train_writer.close()
