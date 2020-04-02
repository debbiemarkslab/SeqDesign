import numpy as np
import tensorflow as tf
#import theano
import scipy
#import theano.tensor as T
from collections import defaultdict
import glob
import time


class DataHelperDoubleWeightingNanobody:
    def __init__(self, alignment_file='', focus_seq_name='',
        mutation_file='', calc_weights=True, working_dir='.',alphabet_type='protein'):

        np.random.seed(42)
        self.alignment_file = alignment_file
        self.focus_seq_name = focus_seq_name
        self.mutation_file = mutation_file
        self.working_dir = working_dir
        self.calc_weights = calc_weights
        self.alphabet_type = alphabet_type

        # Load up the alphabet type to use, whether that be DNA, RNA, or protein
        if self.alphabet_type == 'protein':
            self.alphabet = 'ACDEFGHIKLMNPQRSTVWY*'
            self.reorder_alphabet = 'DEKRHNQSTPGAVILMCFYW*'
        elif self.alphabet_type == 'RNA':
            self.alphabet = 'ACGU*'
            self.reorder_alphabet = 'ACGU*'
        elif self.alphabet_type == 'DNA':
            self.alphabet = 'ACGT*'
            self.reorder_alphabet = 'ACGT*'

        self.aa_dict = {}
        self.idx_to_aa = {}
        for i,aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i
            self.idx_to_aa[i] = aa

        # Do the inverse as well
        self.num_to_aa = {i:aa for aa,i in self.aa_dict.iteritems()}

        if alignment_file != '':
            #then generate the experimental data
            self.gen_alignment_mut_data()

    def one_hot_3D(self, s):
        # One-hot encode as row vector
        x = np.zeros((len(s), len(self.alphabet)))
        for i, letter in enumerate(s):
            if letter in self.aa_dict:
                x[i , self.aa_dict[letter]] = 1
        return x

    def gen_alignment_mut_data(self):

        # Make a dictionary that goes from aa to a number for one-hot
        self.aa_dict = {}
        self.idx_to_aa = {}
        for i,aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i
            self.idx_to_aa[i] = aa

        # Do the inverse as well
        self.num_to_aa = {i:aa for aa,i in self.aa_dict.iteritems()}

        ix = np.array([self.alphabet.find(s) for s in self.reorder_alphabet])


        self.name_to_sequence = {}
        self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists = {}
        self.cluster_id80_to_dict_of_cluster_id90_to_cluster_size = {}

        INPUT = open(self.working_dir+'/datasets/'+self.alignment_file,'r')
        for line in INPUT:
            line = line.rstrip()

            if line != '' and line[0] == '>':
                line = line.rstrip()
                name,cluster_id80,cluster_id90 = line.split(':')

            elif line != '' and line[0] != '>':
                valid = True
                for letter in line:
                    if letter not in self.aa_dict:
                        valid = False
                if valid:
                    self.name_to_sequence[name] = line
                    if cluster_id80 in self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists:
                        if cluster_id90 in self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists[cluster_id80]:
                            self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists[cluster_id80][cluster_id90] += [name]
                            self.cluster_id80_to_dict_of_cluster_id90_to_cluster_size[cluster_id80][cluster_id90] += 1

                        else:
                            self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists[cluster_id80][cluster_id90] = [name]
                            self.cluster_id80_to_dict_of_cluster_id90_to_cluster_size[cluster_id80][cluster_id90] = 1

                    else:
                        self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists[cluster_id80] = {cluster_id90:[name]}
                        self.cluster_id80_to_dict_of_cluster_id90_to_cluster_size[cluster_id80] = {cluster_id90:1}

        INPUT.close()

        self.cluster_id80_list = self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists.keys()
        print "Num clusters:",len(self.cluster_id80_list)

    def generate_one_family_minibatch_data(self, minibatch_size, use_embedding=True,
        reverse=False, matching=False, top_k=False):

        start = time.time()
        Neff = len(self.cluster_id80_list)
        cluster_name_list = np.random.choice(self.cluster_id80_list,minibatch_size).tolist()

        minibatch_max_seq_len = 0
        sequence_list = []
        for i,cluster_id80 in enumerate(cluster_name_list):

            # First pick a cluster id90 from the cluster id80s
            cluster_id90 = np.random.choice(self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists[cluster_id80].keys(),1)[0]

            # Then pick a random sequence all in those clusters
            if self.cluster_id80_to_dict_of_cluster_id90_to_cluster_size[cluster_id80][cluster_id90] > 1:
                seq_name = np.random.choice(self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists[cluster_id80][cluster_id90],1)[0]
            else:
                seq_name = self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists[cluster_id80][cluster_id90][0]

            # then grab the associated sequence
            seq = self.name_to_sequence[seq_name]

            sequence_list.append(seq)

            seq_len = len(seq)
            if seq_len > minibatch_max_seq_len:
                minibatch_max_seq_len = seq_len

        # Add 1 to compensate for the start and end character
        minibatch_max_seq_len += 1

        prot_decoder_output = np.zeros((minibatch_size,1,minibatch_max_seq_len,len(self.alphabet)))
        prot_decoder_input = np.zeros((minibatch_size,1,minibatch_max_seq_len,len(self.alphabet)))

        if matching:
            prot_decoder_output_r = np.zeros((minibatch_size,1,minibatch_max_seq_len,len(self.alphabet)))
            prot_decoder_input_r = np.zeros((minibatch_size,1,minibatch_max_seq_len,len(self.alphabet)))

        prot_mask_decoder = np.zeros((minibatch_size,1,minibatch_max_seq_len,1))
        family_one_hot = np.zeros(1)

        if use_embedding:
            family_one_hot[0] = 1.

        for i,sequence in enumerate(sequence_list):

            if reverse:
                sequence = sequence[::-1]

            decoder_input_seq = '*'+sequence
            decoder_output_seq = sequence+'*'

            if matching:
                sequence_r = sequence[::-1]
                decoder_input_seq_r = '*'+sequence_r
                decoder_output_seq_r = sequence_r+'*'

            for j in range(len(decoder_input_seq)):
                prot_decoder_input[i,0,j,self.aa_dict[decoder_input_seq[j]]] = 1
                prot_decoder_output[i,0,j,self.aa_dict[decoder_output_seq[j]]] = 1
                prot_mask_decoder[i,0,j,0] = 1

                if matching:
                    prot_decoder_input_r[i,0,j,self.aa_dict[decoder_input_seq_r[j]]] = 1
                    prot_decoder_output_r[i,0,j,self.aa_dict[decoder_output_seq_r[j]]] = 1

        if matching:
            return prot_decoder_input, prot_decoder_output, prot_mask_decoder,\
                prot_decoder_input_r, prot_decoder_output_r, family_one_hot, Neff
        else:
            return prot_decoder_input, prot_decoder_output, prot_mask_decoder, \
                family_one_hot, Neff


    def seq_list_to_one_hot(self, sequence_list):
        seq_arr = np.zeros((len(sequence_list), 1, len(sequence_list[0]), len(self.alphabet)))
        for i,seq in enumerate(sequence_list):
            for j,aa in enumerate(seq):
                k = self.aa_dict[aa]
                seq_arr[i,0,j,k] = 1.
        return seq_arr

    def one_hot_to_seq_list(self, one_hot_seqs):
        seq_aa_idx = np.argmax(one_hot_seqs, axis=-1)
        seq_list = []
        for i in range(one_hot_seqs.shape[0]):
            seq = ''
            for j in range(one_hot_seqs.shape[1]):
                seq += self.idx_to_aa[seq_aa_idx[i][j]]
            seq_list.append(seq)
        return seq_list


    def read_in_test_data(self, input_filename):
        INPUT = open(input_filename, 'r')

        self.test_name_to_sequence = {}
        self.test_name_list = []
        first_time = True
        seq = ''
        name = ''
        for i,line in enumerate(INPUT):
            line = line.rstrip()
            if line[0] == '>' and first_time == True:
                first_time = False
                name = line
            elif line[0] == '>' and first_time == False:
                self.test_name_to_sequence[name] = seq
                self.test_name_list.append(name)
                seq = ''
                name = line
            else:
                seq += line

        self.test_name_to_sequence[name] = seq
        self.test_name_list.append(name)

        INPUT.close()

    def output_log_probs(self, sess, model, output_filename, minibatch_size=100,
        Neff=100.):

        num_seqs = len(self.test_name_list)
        batch_order = np.arange(num_seqs)

        self.mean_log_probs_list = []
        self.forward_log_probs_list = []
        self.reverse_log_prob_list = []

        out_counter = 0
        for idx_batch in xrange(0,num_seqs,minibatch_size):
            start_time = time.time()
            batch_seq_name_list = self.test_name_list[idx_batch:idx_batch+minibatch_size]
            curr_minibatch_size = len(batch_seq_name_list)
            minibatch_max_seq_len = 0
            sequence_list = []
            for name in batch_seq_name_list:
                sequence_list.append(self.test_name_to_sequence[name])
                if len(self.test_name_to_sequence[name]) > minibatch_max_seq_len:
                    minibatch_max_seq_len = len(self.test_name_to_sequence[name])

            minibatch_max_seq_len += 1
            prot_decoder_input = np.zeros((curr_minibatch_size,1,minibatch_max_seq_len,len(self.alphabet)))
            prot_decoder_output = np.zeros((curr_minibatch_size,1,minibatch_max_seq_len,len(self.alphabet)))
            prot_decoder_input_r = np.zeros((curr_minibatch_size,1,minibatch_max_seq_len,len(self.alphabet)))
            prot_decoder_output_r = np.zeros((curr_minibatch_size,1,minibatch_max_seq_len,len(self.alphabet)))
            prot_mask_decoder = np.zeros((curr_minibatch_size,1,minibatch_max_seq_len,1))

            for i,sequence in enumerate(sequence_list):
                decoder_input_seq = '*'+sequence
                decoder_output_seq = sequence+'*'

                sequence_r = sequence[::-1]
                decoder_input_seq_r = '*'+sequence_r
                decoder_output_seq_r = sequence_r+'*'

                for j in range(len(decoder_input_seq)):
                    prot_decoder_input[i,0,j,self.aa_dict[decoder_input_seq[j]]] = 1
                    prot_decoder_output[i,0,j,self.aa_dict[decoder_output_seq[j]]] = 1
                    prot_decoder_input_r[i,0,j,self.aa_dict[decoder_input_seq_r[j]]] = 1
                    prot_decoder_output_r[i,0,j,self.aa_dict[decoder_output_seq_r[j]]] = 1
                    prot_mask_decoder[i,0,j,0] = 1

            feed_dict = {model.placeholders["sequences_start_f"]: prot_decoder_input,\
                model.placeholders["sequences_stop_f"]: prot_decoder_output,\
                model.placeholders["sequences_start_r"]: prot_decoder_input_r,\
                model.placeholders["sequences_stop_r"]: prot_decoder_output_r,\
                model.placeholders["mask_decoder_1D_f"]: prot_mask_decoder,\
                model.placeholders["mask_decoder_1D_r"]: prot_mask_decoder,\
                model.placeholders["Neff_f"]:[ Neff ],\
                model.placeholders["Neff_r"]:[ Neff ],\
                model.placeholders["step"]:[200],
                model.placeholders["dropout"]: 1.}

            ce_loss_f, ce_loss_r = sess.run([model.tensors["cross_entropy_per_seq_f"],\
                model.tensors["cross_entropy_per_seq_r"]], feed_dict=feed_dict)

            self.mean_log_probs_list += np.mean(np.concatenate(\
                [np.expand_dims(ce_loss_f,axis=1),np.expand_dims(ce_loss_r,axis=1),],axis=1),axis=1).tolist()
            self.forward_log_probs_list += ce_loss_f.tolist()
            self.reverse_log_prob_list += ce_loss_r.tolist()
            out_counter += curr_minibatch_size
            print str(out_counter)+" completed in "+str(time.time()-start_time)+" s"

        OUTPUT = open(output_filename, 'w')
        OUTPUT.write("name,mean,mean/L,forward,reverse,sequence\n")
        for i,name in enumerate(self.test_name_list):
            out_list = [name,self.mean_log_probs_list[i],self.mean_log_probs_list[i]/float(len(self.test_name_to_sequence[name])),\
                self.forward_log_probs_list[i],self.reverse_log_prob_list[i],self.test_name_to_sequence[name]]
            OUTPUT.write(",".join([str(val) for val in out_list])+'\n')
        OUTPUT.close()
