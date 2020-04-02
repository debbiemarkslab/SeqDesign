#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

import time
import sys, os
sys.path.append("../code/")

from SeqDesign import hyper_conv_auto as model
from SeqDesign import helper

working_dir = ".."

data_helper = helper.DataHelperDoubleWeightingNanobody(working_dir=working_dir,
    alignment_file="Manglik_filt_seq_id80_id90.fa")

# Variables for runtime modification
batch_size = 500
num_batches = 1000000
temp = float(sys.argv[1])
r_seed = int(sys.argv[2])

print(r_seed,type(r_seed))

np.random.seed(r_seed)

alphabet_list = list(data_helper.alphabet)


sess_name = "nanobody.ckpt-250000"
output_filename = working_dir+"/output/nanobody_temp-"+str(temp)+"_param-"+sess_name+"_rseed-"+str(r_seed)+".fa"
OUTPUT = open(output_filename, "w")
OUTPUT.close()

# Provide the starting sequence to use for generation
input_seq = "*EVQLVESGGGLVQAGGSLRLSCAASGFTFSSYAMGWYRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYC"


dims = {}
conv_model = model.AutoregressiveFR(dims=dims)

params = tf.trainable_variables()
p_counts = [np.prod(v.get_shape().as_list()) for v in params]
p_total = sum(p_counts)
print("Total parameter number:",p_total,"\n")

saver = tf.train.Saver()

#with tf.Session(config=cfg) as sess:
with tf.Session() as sess:

    # Initialization
    print("Initializing variables")
    init = tf.global_variables_initializer()
    sess.run(init)

    sess_namedir = working_dir+"/sess/"+sess_name
    saver.restore(sess, sess_namedir)
    print("Loaded parameters")

    # Run optimization
    for i in range(num_batches):

        complete = False
        start = time.time()

        input_seq_list = batch_size * [input_seq]
        one_hot_seqs_f = data_helper.seq_list_to_one_hot(input_seq_list)
        one_hot_seq_mask = np.sum(one_hot_seqs_f,axis=-1,keepdims=True)

        # if the sequence is complete, set the value to zero,
        #   otherwise they should all be ones
        completed_seq_list = batch_size * [1.]
        decoding_steps = 0

        while complete == False and decoding_steps < 50:

            feed_dict = {conv_model.placeholders["sequences_start_f"]: one_hot_seqs_f,\
                conv_model.placeholders["sequences_stop_f"]: one_hot_seqs_f,\
                conv_model.placeholders["mask_decoder_1D_f"]: one_hot_seq_mask,\
                conv_model.placeholders["Neff_f"]:[1.],\
                conv_model.placeholders["sequences_start_r"]: one_hot_seqs_f,\
                conv_model.placeholders["sequences_stop_r"]: one_hot_seqs_f,\
                conv_model.placeholders["mask_decoder_1D_r"]: one_hot_seq_mask,\
                conv_model.placeholders["Neff_r"]:[1.],\
                conv_model.placeholders["step"]:[10.],
                conv_model.placeholders["dropout"]: 1.0}

            seq_logits_f = sess.run([conv_model.tensors["sequence_logits_f"]],feed_dict=feed_dict)[0]

            # slice off the last element of the list
            output_logits = seq_logits_f[:,:,-1] * temp

            # Safe exponents
            exp_output_logits = np.exp(output_logits)

            # Convert it to probabilities
            output_probs = exp_output_logits / np.sum(exp_output_logits,axis=-1,keepdims=True)

            # sample the sequences accordingly
            batch_aa_list = []

            for idx_batch in range(batch_size):
                new_aa = np.random.choice(alphabet_list,1,p=output_probs[idx_batch].flatten())[0]
                input_seq_list[idx_batch] += new_aa

                if new_aa == "*":
                    completed_seq_list[idx_batch] = 0.

                batch_aa_list.append([new_aa])

            batch_one_hots = data_helper.seq_list_to_one_hot(batch_aa_list)
            batch_mask = np.reshape(np.asarray(completed_seq_list),[batch_size,1,1,1])

            one_hot_seqs_f = np.concatenate([one_hot_seqs_f,batch_one_hots],axis=2)
            one_hot_seq_mask = np.concatenate([one_hot_seq_mask,batch_mask],axis=2)

            decoding_steps += 1

            if np.sum(completed_seq_list) == 0.0:
                print("completed!")
                complete = True

        OUTPUT = open(output_filename, "a")
        for idx_seq in range(batch_size):
            batch_seq = input_seq_list[idx_seq]
            out_seq = ""
            end_seq = False
            for idx_aa,aa in enumerate(batch_seq):
                if idx_aa != 0:
                    if end_seq is False:
                        out_seq += aa
                    if aa == "*":
                        end_seq = True
            OUTPUT.write(">"+str(int(batch_size*i+idx_seq))+"\n"+out_seq+"\n")
        OUTPUT.close()
        print("Batch "+str(i+1)+" done in "+str(time.time()-start)+" s")
