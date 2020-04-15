import os
import numpy as np
import tensorflow as tf
# import theano
import scipy
# import theano.tensor as T
from collections import defaultdict
import glob
import time

from seqdesign import aws_utils


class DataHelperSingleFamily:
    def __init__(self, dataset='', alignment_file='', focus_seq_name='',
                 mutation_file='', calc_weights=True, working_dir='.', theta=0.2,
                 load_all_sequences=True, alphabet_type='protein', max_seq_len=-1,
                 longest_entry=125, r_seed=42,
                 aws_util: aws_utils.AWSUtility = None):

        np.random.seed(r_seed)
        self.dataset = dataset
        self.alignment_file = alignment_file
        self.focus_seq_name = focus_seq_name
        self.mutation_file = mutation_file
        self.working_dir = working_dir
        self.calc_weights = calc_weights
        self.alphabet_type = alphabet_type
        self.max_seq_len = max_seq_len
        self.aws_util = aws_util
        # Alignment processing parameters
        self.theta = theta

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

        print(self.alphabet)
        # Make a dictionary that goes from aa to a number for one-hot
        self.aa_dict = {}
        for i, aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        # Do the inverse as well
        self.num_to_aa = {i: aa for aa, i in self.aa_dict.items()}

        # then generate the experimental data
        if dataset != '':
            self.gen_alignment_mut_data()

    def one_hot_3D(self, s):
        # One-hot encode as row vector
        x = np.zeros((len(s), len(self.alphabet)))
        for i, letter in enumerate(s):
            if letter in self.aa_dict:
                x[i, self.aa_dict[letter]] = 1
        return x

    def gen_alignment_mut_data(self):

        ix = np.array([self.alphabet.find(s) for s in self.reorder_alphabet])

        self.family_name_to_sequence_encoder_list = {}
        self.family_name_to_sequence_decoder_input_list = {}
        self.family_name_to_sequence_decoder_output_list = {}
        self.family_name_to_sequence_weight_list = {}
        self.family_name_to_Neff = {}
        self.family_name_list = []
        self.family_idx_list = []
        self.family_name = ''

        # for now, we will make all the sequences have the same length of
        #   encoded matrices, though this is wasteful
        filenames = glob.glob(f'{self.working_dir}/datasets/sequences/{self.dataset}*.fa')
        if not filenames and self.aws_util is not None:
            if not self.aws_util.s3_get_file_grep(
                s3_folder='datasets/sequences',
                dest_folder=f'{self.working_dir}/datasets/sequences/',
                search_pattern=f'{self.dataset}.*\\.fa',
            ):
                raise Exception("Could not download dataset files from S3.")
            filenames = glob.glob(f'{self.working_dir}/datasets/sequences/{self.dataset}*.fa')

        max_seq_len = 0
        max_family_size = 0
        for filename in filenames:
            print(f"Dataset: {filename}")
            INPUT = open(filename, 'r')

            encoder_sequence_list = []
            decoder_input_sequence_list = []
            decoder_output_sequence_list = []
            weight_list = []

            family_name_list = filename.split('/')[-1].split('_')
            family_name = family_name_list[0] + '_' + family_name_list[1]

            print(f"Family: {family_name}")

            family_size = 0
            ind_family_idx_list = []
            first_time = True
            seq = ''

            for line in INPUT:
                line = line.rstrip()

                if line != '':
                    if line[0] == '>' and first_time:
                        weight = float(line.split(':')[-1])
                        first_time = False

                    elif line[0] == '>' and first_time == False:
                        valid = True
                        for letter in seq:
                            if letter not in self.aa_dict:
                                valid = False
                        if valid:
                            encoder_sequence_list.append(seq)
                            decoder_input_sequence_list.append('*' + seq)
                            decoder_output_sequence_list.append(seq + '*')
                            ind_family_idx_list.append(family_size)
                            weight_list.append(weight)

                            family_size += 1
                            if len(seq) > max_seq_len:
                                max_seq_len = len(seq)

                        seq = ''
                        weight = float(line.split(':')[-1])

                    else:
                        seq += line

            INPUT.close()

            valid = True
            for letter in seq:
                if letter not in self.aa_dict:
                    valid = False
            if valid:
                encoder_sequence_list.append(seq)
                decoder_input_sequence_list.append('*' + seq)
                decoder_output_sequence_list.append(seq + '*')
                ind_family_idx_list.append(family_size)
                weight_list.append(weight)

                family_size += 1
                if len(seq) > max_seq_len:
                    max_seq_len = len(seq)

            if family_size > max_family_size:
                max_family_size = family_size

            self.family_name_to_sequence_encoder_list[family_name] = encoder_sequence_list
            self.family_name_to_sequence_decoder_input_list[family_name] = decoder_input_sequence_list
            self.family_name_to_sequence_decoder_output_list[family_name] = decoder_output_sequence_list
            self.family_name_to_sequence_weight_list[family_name] = (
                        np.asarray(weight_list) / np.sum(weight_list)).tolist()
            self.family_name_to_Neff[family_name] = np.sum(weight_list)
            self.family_name = family_name

            self.family_name_list.append(family_name)
            self.family_idx_list.append(ind_family_idx_list)

        self.family_name = family_name
        self.seq_len = max_seq_len
        self.num_families = len(self.family_name_list)
        self.max_family_size = max_family_size

        print("Number of families:", self.num_families)
        print("Neff:", np.sum(weight_list))

        print('Encoding sequences')
        self.family_name_to_idx = {}
        self.idx_to_family_name = {}

        for i, family_name in enumerate(self.family_name_list):
            self.family_name_to_idx[family_name] = i
            self.idx_to_family_name[i] = family_name

        # Then read in the mutation data so we can predict it later
        self.protein_mutation_names = []
        self.protein_names_to_uppercase_idx = {}
        self.protein_names_to_one_hot_seqs_encoder = {}
        self.mut_protein_names_to_seqs_encoder = {}
        self.protein_names_to_one_hot_seqs_decoder_input_f = {}
        self.protein_names_to_one_hot_seqs_decoder_output_f = {}
        self.protein_names_to_one_hot_seqs_decoder_input_r = {}
        self.protein_names_to_one_hot_seqs_decoder_output_r = {}
        self.protein_names_to_one_hot_seqs_encoder_mask = {}
        self.protein_names_to_one_hot_seqs_decoder_mask = {}
        self.protein_names_to_measurement_list = {}
        self.protein_names_to_mutation_list = {}

        for filename in glob.glob(self.working_dir + '/datasets/mutation_data/' + self.dataset + '*.csv'):

            INPUT = open(filename, 'r')

            encoder_sequence_list = []
            decoder_input_sequence_list = []
            decoder_output_sequence_list = []
            mutation_list = []
            uppercase_list = []
            measurement_list = []

            family_name_list = filename.split('/')[-1].split('_')
            family_name = family_name_list[0] + '_' + family_name_list[1]

            mutation_counter = 0
            for i, line in enumerate(INPUT):
                line = line.rstrip()
                line_list = line.split(',')
                if i != 0:
                    mutation, is_upper, measurement, sequence = line_list

                    measurement = float(measurement)
                    if np.isfinite(measurement):
                        if is_upper == 'True':
                            uppercase_list.append(mutation_counter)
                        mutation_list.append(mutation)
                        measurement_list.append(float(measurement))

                        encoder_sequence_list.append(sequence)
                        decoder_input_sequence_list.append('*' + sequence)
                        decoder_output_sequence_list.append(sequence + '*')
                        seq_len_mutations = len(sequence)
                        mutation_counter += 1

            INPUT.close()

            self.protein_mutation_names.append(family_name)
            self.protein_names_to_uppercase_idx[family_name] = np.asarray(uppercase_list)
            self.protein_names_to_measurement_list[family_name] = np.asarray(measurement_list)
            self.protein_names_to_mutation_list[family_name] = mutation_list
            self.mut_protein_names_to_seqs_encoder[family_name] = encoder_sequence_list

            prot_encoder = np.zeros((len(measurement_list), 1, seq_len_mutations, len(self.alphabet)))
            prot_mask_encoder = np.zeros((len(measurement_list), 1, seq_len_mutations, 1))
            prot_decoder_output_f = np.zeros((len(measurement_list), 1, seq_len_mutations + 1, len(self.alphabet)))
            prot_decoder_input_f = np.zeros((len(measurement_list), 1, seq_len_mutations + 1, len(self.alphabet)))
            prot_decoder_output_r = np.zeros((len(measurement_list), 1, seq_len_mutations + 1, len(self.alphabet)))
            prot_decoder_input_r = np.zeros((len(measurement_list), 1, seq_len_mutations + 1, len(self.alphabet)))
            prot_mask_decoder = np.zeros((len(measurement_list), 1, seq_len_mutations + 1, 1))

            for j, sequence in enumerate(encoder_sequence_list):
                for k, letter in enumerate(sequence):
                    if letter in self.aa_dict:
                        l = self.aa_dict[letter]
                        prot_encoder[j, 0, k, l] = 1.0
                        prot_mask_encoder[j, 0, k, 0] = 1.0

            for j, sequence in enumerate(decoder_output_sequence_list):
                for k, letter in enumerate(sequence):
                    if letter in self.aa_dict:
                        l = self.aa_dict[letter]
                        prot_decoder_output_f[j, 0, k, l] = 1.0
                        prot_mask_decoder[j, 0, k, 0] = 1.0

            for j, sequence in enumerate(decoder_input_sequence_list):
                for k, letter in enumerate(sequence):
                    if letter in self.aa_dict:
                        l = self.aa_dict[letter]
                        prot_decoder_input_f[j, 0, k, l] = 1.0

            for j, sequence in enumerate(encoder_sequence_list):
                sequence_r = "*" + sequence[::-1]
                for k, letter in enumerate(sequence_r):
                    if letter in self.aa_dict:
                        l = self.aa_dict[letter]
                        prot_decoder_input_r[j, 0, k, l] = 1.0

            for j, sequence in enumerate(encoder_sequence_list):
                sequence_r = sequence[::-1] + '*'
                for k, letter in enumerate(sequence_r):
                    if letter in self.aa_dict:
                        l = self.aa_dict[letter]
                        prot_decoder_output_r[j, 0, k, l] = 1.0

            self.protein_names_to_one_hot_seqs_encoder[family_name] = prot_encoder
            self.protein_names_to_one_hot_seqs_encoder_mask[family_name] = prot_mask_encoder
            self.protein_names_to_one_hot_seqs_decoder_input_f[family_name] = prot_decoder_input_f
            self.protein_names_to_one_hot_seqs_decoder_output_f[family_name] = prot_decoder_output_f
            self.protein_names_to_one_hot_seqs_decoder_input_r[family_name] = prot_decoder_input_r
            self.protein_names_to_one_hot_seqs_decoder_output_r[family_name] = prot_decoder_output_r
            self.protein_names_to_one_hot_seqs_decoder_mask[family_name] = prot_mask_decoder

    def generate_minibatch_idx(self, minibatch_size):

        # First choose
        family_idx = np.random.choice(len(self.family_idx_list), minibatch_size)

        one_hot_label = np.zeros((minibatch_size, len(self.family_idx_list)))

        sequence_idx = []
        for i, idx in enumerate(family_idx):
            sequence_idx.append(np.random.choice(self.family_idx_list[idx], 1)[0])
            one_hot_label[i][idx] = 1.

        return family_idx, sequence_idx, one_hot_label

    def gen_target_aa(self, x_input, x_input_lengths, dims):

        target_aa = []
        query_seq = np.copy(x_input)
        for i, input_len in enumerate(x_input_lengths):
            idx_rand = np.random.randint(input_len)
            target_aa.append(query_seq[i, 0, idx_rand].flatten().tolist())
            query_seq[i, 0, idx_rand] *= 0.
            query_seq[i, 0, idx_rand, dims["alphabet"] - 1] = 1.

        return query_seq, np.asarray(target_aa)

    def generate_reset_script(self, sess_filename, channels, dataset, r_seed):
        OUTPUT = open(
            f"{self.working_dir}/revive_executable/{dataset}_channels-{channels}_rseed-{r_seed}.sh",
            'w')
        OUTPUT.write('#!/bin/bash\n')
        OUTPUT.write("module load gcc/6.2.0 cuda/9.0\n")
        OUTPUT.write(
            f"/n/groups/marks/users/aaron/anaconda3/envs/tensorflow_gpuenv/bin/python3 "
            f"{self.working_dir}/run_autoregressive_fr.py "
            f"--dataset {dataset} --channels {channels} --restore {sess_filename} --r-seed {r_seed}"
            f"\n"
        )
        OUTPUT.close()

    def generate_one_family_minibatch_data(self, minibatch_size, use_embedding=True,
                                           reverse=False, matching=False, top_k=False):

        # First choose which families
        family_idx = np.random.choice(len(self.family_idx_list), 1)[0]
        family_name = self.idx_to_family_name[family_idx]
        batch_order = np.arange(len(self.family_name_to_sequence_decoder_output_list[family_name]))
        family_weights = self.family_name_to_sequence_weight_list[family_name]
        Neff = self.family_name_to_Neff[family_name]

        # Then choose which sequences to grab in those families
        if top_k:
            sequence_idx = minibatch_size * [0]
        else:
            sequence_idx = np.random.choice(batch_order, minibatch_size, p=family_weights).tolist()

        minibatch_max_seq_len = 0
        for i, idx_seq in enumerate(sequence_idx):
            # print i,idx_seq
            seq = self.family_name_to_sequence_decoder_output_list[family_name][idx_seq]

            seq_len = len(seq)
            if seq_len > minibatch_max_seq_len:
                minibatch_max_seq_len = seq_len

        prot_decoder_output = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
        prot_decoder_input = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))

        if matching:
            prot_decoder_output_r = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
            prot_decoder_input_r = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))

        prot_mask_decoder = np.zeros((minibatch_size, 1, minibatch_max_seq_len, 1))
        family_one_hot = np.zeros(self.num_families)

        if use_embedding:
            family_one_hot[family_idx] = 1.

        for i, idx_seq in enumerate(sequence_idx):
            sequence = self.family_name_to_sequence_encoder_list[family_name][idx_seq]
            if reverse:
                sequence = sequence[::-1]

            decoder_input_seq = '*' + sequence
            decoder_output_seq = sequence + '*'

            if matching:
                sequence_r = sequence[::-1]
                decoder_input_seq_r = '*' + sequence_r
                decoder_output_seq_r = sequence_r + '*'

            for j in range(len(decoder_input_seq)):
                prot_decoder_input[i, 0, j, self.aa_dict[decoder_input_seq[j]]] = 1
                prot_decoder_output[i, 0, j, self.aa_dict[decoder_output_seq[j]]] = 1
                prot_mask_decoder[i, 0, j, 0] = 1

                if matching:
                    prot_decoder_input_r[i, 0, j, self.aa_dict[decoder_input_seq_r[j]]] = 1
                    prot_decoder_output_r[i, 0, j, self.aa_dict[decoder_output_seq_r[j]]] = 1

        if matching:
            return prot_decoder_input, prot_decoder_output, prot_mask_decoder, \
                   prot_decoder_input_r, prot_decoder_output_r, family_one_hot, Neff
        else:
            return prot_decoder_input, prot_decoder_output, prot_mask_decoder, \
                   family_one_hot, Neff

    def generate_minibatch_data(self, minibatch_size):

        # First choose which families
        family_idx = np.random.choice(len(self.family_idx_list), minibatch_size)

        # Then choose which sequences to grab in those families
        sequence_idx = []
        minibatch_max_seq_len = 0
        for i, idx in enumerate(family_idx):

            idx_seq = np.random.choice(self.family_idx_list[idx], 1)[0]
            family_name = self.idx_to_family_name[idx]
            seq = self.family_name_to_sequence_decoder_output_list[family_name][idx_seq]

            seq_len = len(seq)
            if seq_len > minibatch_max_seq_len:
                minibatch_max_seq_len = seq_len

            sequence_idx.append(idx_seq)

        prot_decoder_output = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
        prot_decoder_input = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
        prot_mask_decoder = np.zeros((minibatch_size, 1, minibatch_max_seq_len, 1))

        for i, idx in enumerate(family_idx):

            family_name = self.idx_to_family_name[idx]
            idx_seq = sequence_idx[i]

            decoder_input_seq = self.family_name_to_sequence_decoder_input_list[family_name][idx_seq]
            decoder_output_seq = self.family_name_to_sequence_decoder_output_list[family_name][idx_seq]

            for j in range(len(decoder_input_seq)):
                prot_decoder_input[i, 0, j, self.aa_dict[decoder_input_seq[j]]] = 1
                prot_decoder_output[i, 0, j, self.aa_dict[decoder_output_seq[j]]] = 1
                prot_mask_decoder[i, 0, j, 0] = 1

        return prot_decoder_input, prot_decoder_output, prot_mask_decoder

    def get_experimental_error_single_family_fr(
            self, sess, conv_model, protein_name, step, batch_size=30,
            N_pred_iterations=10, dropout_p=1., use_only_uppercase=True):

        prot_encoder = self.protein_names_to_one_hot_seqs_encoder[protein_name]
        prot_mask_encoder = self.protein_names_to_one_hot_seqs_encoder_mask[protein_name]
        prot_decoder_input_f = self.protein_names_to_one_hot_seqs_decoder_input_f[protein_name]
        prot_decoder_output_f = self.protein_names_to_one_hot_seqs_decoder_output_f[protein_name]
        prot_decoder_input_r = self.protein_names_to_one_hot_seqs_decoder_input_r[protein_name]
        prot_decoder_output_r = self.protein_names_to_one_hot_seqs_decoder_output_r[protein_name]
        prot_mask_decoder = self.protein_names_to_one_hot_seqs_decoder_mask[protein_name]

        measurement_list = self.protein_names_to_measurement_list[protein_name]

        # if only using the mutations from uppercase columns as in previous pubs
        if use_only_uppercase:
            idx_uppercase = self.protein_names_to_uppercase_idx[protein_name]

            prot_encoder = prot_encoder[idx_uppercase]
            prot_mask_encoder = prot_mask_encoder[idx_uppercase]
            prot_decoder_input_f = prot_decoder_input_f[idx_uppercase]
            prot_decoder_output_f = prot_decoder_output_f[idx_uppercase]
            prot_decoder_input_r = prot_decoder_input_r[idx_uppercase]
            prot_decoder_output_r = prot_decoder_output_r[idx_uppercase]
            prot_mask_decoder = prot_mask_decoder[idx_uppercase]
            measurement_list = measurement_list[idx_uppercase]

        prediction_matrix_ce_f = np.zeros((len(prot_encoder), N_pred_iterations))
        prediction_matrix_all_f = np.zeros((len(prot_encoder), N_pred_iterations))

        prediction_matrix_ce_r = np.zeros((len(prot_encoder), N_pred_iterations))
        prediction_matrix_all_r = np.zeros((len(prot_encoder), N_pred_iterations))

        for i in range(N_pred_iterations):
            batch_order = np.arange(prot_encoder.shape[0])
            np.random.shuffle(batch_order)

            idx_family_one_hot = self.family_name_to_idx[protein_name]

            for j in range(0, prot_encoder.shape[0], batch_size):

                batch_index = batch_order[j:j + batch_size]
                batch_index_shape = prot_decoder_input_f[batch_index].shape[0]

                feed_dict = {conv_model.placeholders["sequences_start_f"]: prot_decoder_input_f[batch_index],
                             conv_model.placeholders["sequences_stop_f"]: prot_decoder_output_f[batch_index],
                             conv_model.placeholders["sequences_start_r"]: prot_decoder_input_r[batch_index],
                             conv_model.placeholders["sequences_stop_r"]: prot_decoder_output_r[batch_index],
                             conv_model.placeholders["mask_decoder_1D_f"]: prot_mask_decoder[batch_index],
                             conv_model.placeholders["mask_decoder_1D_r"]: prot_mask_decoder[batch_index],
                             conv_model.placeholders["Neff_f"]: [
                                 float(len(self.family_name_to_sequence_decoder_output_list[protein_name]))],
                             conv_model.placeholders["Neff_r"]: [
                                 float(len(self.family_name_to_sequence_decoder_output_list[protein_name]))],
                             conv_model.placeholders["step"]: [i],
                             conv_model.placeholders["dropout"]: dropout_p}

                ce_loss_f, loss_f, ce_loss_r, loss_r = sess.run([conv_model.tensors["cross_entropy_per_seq_f"],
                                                                 conv_model.tensors["loss_per_seq_f"],
                                                                 conv_model.tensors["cross_entropy_per_seq_r"],
                                                                 conv_model.tensors["loss_per_seq_r"]],
                                                                feed_dict=feed_dict)

                for k, idx_batch in enumerate(batch_index.tolist()):
                    prediction_matrix_ce_f[idx_batch][i] = ce_loss_f[k]
                    prediction_matrix_all_f[idx_batch][i] = loss_f[k]
                    prediction_matrix_ce_r[idx_batch][i] = ce_loss_r[k]
                    prediction_matrix_all_r[idx_batch][i] = loss_r[k]

        prediction_ce_f = np.mean(prediction_matrix_ce_f, axis=1)
        prediction_all_f = np.mean(prediction_matrix_all_f, axis=1)

        prediction_ce_r = np.mean(prediction_matrix_ce_r, axis=1)
        prediction_all_r = np.mean(prediction_matrix_all_r, axis=1)

        prediction_ce_fr = np.concatenate(
            [np.expand_dims(prediction_ce_f, axis=-1), np.expand_dims(prediction_ce_r, axis=-1)], axis=-1)
        prediction_all_fr = np.concatenate(
            [np.expand_dims(prediction_all_f, axis=-1), np.expand_dims(prediction_all_r, axis=-1)], axis=-1)

        prediction_ce_mean = np.mean(prediction_ce_fr, axis=1)
        prediction_all_mean = np.mean(prediction_ce_fr, axis=1)

        prediction_ce_min = np.min(prediction_ce_fr, axis=1)
        prediction_all_min = np.min(prediction_all_fr, axis=1)

        prediction_ce_max = np.max(prediction_ce_fr, axis=1)
        prediction_all_max = np.max(prediction_all_fr, axis=1)

        result_dict = {}

        result_dict["prediction_ce_f"] = scipy.stats.spearmanr(prediction_ce_f, measurement_list)[0]
        result_dict["prediction_all_f"] = scipy.stats.spearmanr(prediction_all_f, measurement_list)[0]

        result_dict["prediction_ce_r"] = scipy.stats.spearmanr(prediction_ce_r, measurement_list)[0]
        result_dict["prediction_all_r"] = scipy.stats.spearmanr(prediction_all_r, measurement_list)[0]

        result_dict["prediction_ce_mean"] = scipy.stats.spearmanr(prediction_ce_mean, measurement_list)[0]
        result_dict["prediction_all_mean"] = scipy.stats.spearmanr(prediction_all_mean, measurement_list)[0]

        result_dict["prediction_ce_min"] = scipy.stats.spearmanr(prediction_ce_min, measurement_list)[0]
        result_dict["prediction_all_min"] = scipy.stats.spearmanr(prediction_all_min, measurement_list)[0]

        result_dict["prediction_ce_max"] = scipy.stats.spearmanr(prediction_ce_max, measurement_list)[0]
        result_dict["prediction_all_max"] = scipy.stats.spearmanr(prediction_all_max, measurement_list)[0]

        return result_dict

    def get_experimental_error_fr(self, sess, conv_model, protein_name, step, N_pred_iterations=10, batch_size=4, use_only_uppercase=True,
                                  family_embedding_model=True, use_embedding=True):

        prot_encoder = self.protein_names_to_one_hot_seqs_encoder[protein_name]
        prot_mask_encoder = self.protein_names_to_one_hot_seqs_encoder_mask[protein_name]
        prot_decoder_input_f = self.protein_names_to_one_hot_seqs_decoder_input_f[protein_name]
        prot_decoder_output_f = self.protein_names_to_one_hot_seqs_decoder_output_f[protein_name]
        prot_decoder_input_r = self.protein_names_to_one_hot_seqs_decoder_input_r[protein_name]
        prot_decoder_output_r = self.protein_names_to_one_hot_seqs_decoder_output_r[protein_name]
        prot_mask_decoder = self.protein_names_to_one_hot_seqs_decoder_mask[protein_name]

        measurement_list = self.protein_names_to_measurement_list[protein_name]

        # if only using the mutations from uppercase columns as in previous pubs
        if use_only_uppercase:
            idx_uppercase = self.protein_names_to_uppercase_idx[protein_name]

            prot_encoder = prot_encoder[idx_uppercase]
            prot_mask_encoder = prot_mask_encoder[idx_uppercase]
            prot_decoder_input_f = prot_decoder_input_f[idx_uppercase]
            prot_decoder_output_f = prot_decoder_output_f[idx_uppercase]
            prot_decoder_input_r = prot_decoder_input_r[idx_uppercase]
            prot_decoder_output_r = prot_decoder_output_r[idx_uppercase]
            prot_mask_decoder = prot_mask_decoder[idx_uppercase]
            measurement_list = measurement_list[idx_uppercase]

        prediction_matrix_ce_f = np.zeros((len(prot_encoder), N_pred_iterations))
        prediction_matrix_all_f = np.zeros((len(prot_encoder), N_pred_iterations))

        prediction_matrix_ce_r = np.zeros((len(prot_encoder), N_pred_iterations))
        prediction_matrix_all_r = np.zeros((len(prot_encoder), N_pred_iterations))

        for i in range(N_pred_iterations):
            batch_order = np.arange(prot_encoder.shape[0])
            np.random.shuffle(batch_order)

            idx_family_one_hot = self.family_name_to_idx[protein_name]

            for j in range(0, prot_encoder.shape[0], batch_size):

                batch_index = batch_order[j:j + batch_size]
                batch_index_shape = prot_decoder_input_f[batch_index].shape[0]

                feed_dict = {conv_model.placeholders["sequences_start_f"]: prot_decoder_input_f[batch_index],
                             conv_model.placeholders["sequences_stop_f"]: prot_decoder_output_f[batch_index],
                             conv_model.placeholders["sequences_start_r"]: prot_decoder_input_r[batch_index],
                             conv_model.placeholders["sequences_stop_r"]: prot_decoder_output_r[batch_index],
                             conv_model.placeholders["mask_decoder_1D_f"]: prot_mask_decoder[batch_index],
                             conv_model.placeholders["mask_decoder_1D_r"]: prot_mask_decoder[batch_index],
                             conv_model.placeholders["Neff_f"]: [
                                 float(len(self.family_name_to_sequence_decoder_output_list[protein_name]))],
                             conv_model.placeholders["Neff_r"]: [
                                 float(len(self.family_name_to_sequence_decoder_output_list[protein_name]))],
                             conv_model.placeholders["step"]: [i]}

                if family_embedding_model:
                    family_one_hot = np.zeros(self.num_families)
                    if use_embedding:
                        family_one_hot[idx_family_one_hot] = 1.
                    feed_dict[conv_model.placeholders["family_embedding_f"]] = family_one_hot
                    feed_dict[conv_model.placeholders["family_embedding_r"]] = family_one_hot

                ce_loss_f, loss_f, ce_loss_r, loss_r = sess.run([conv_model.tensors["cross_entropy_per_seq_f"],
                                                                 conv_model.tensors["loss_per_seq_f"],
                                                                 conv_model.tensors["cross_entropy_per_seq_r"],
                                                                 conv_model.tensors["loss_per_seq_r"]],
                                                                feed_dict=feed_dict)

                for k, idx_batch in enumerate(batch_index.tolist()):
                    prediction_matrix_ce_f[idx_batch][i] = ce_loss_f[k]
                    prediction_matrix_all_f[idx_batch][i] = loss_f[k]
                    prediction_matrix_ce_r[idx_batch][i] = ce_loss_r[k]
                    prediction_matrix_all_r[idx_batch][i] = loss_r[k]

        prediction_ce_f = np.mean(prediction_matrix_ce_f, axis=1)
        prediction_all_f = np.mean(prediction_matrix_all_f, axis=1)

        prediction_ce_r = np.mean(prediction_matrix_ce_r, axis=1)
        prediction_all_r = np.mean(prediction_matrix_all_r, axis=1)

        prediction_ce_fr = np.concatenate(
            [np.expand_dims(prediction_ce_f, axis=-1), np.expand_dims(prediction_ce_r, axis=-1)], axis=-1)
        prediction_all_fr = np.concatenate(
            [np.expand_dims(prediction_all_f, axis=-1), np.expand_dims(prediction_all_r, axis=-1)], axis=-1)

        prediction_ce_mean = np.mean(prediction_ce_fr, axis=1)
        prediction_all_mean = np.mean(prediction_ce_fr, axis=1)

        prediction_ce_min = np.min(prediction_ce_fr, axis=1)
        prediction_all_min = np.min(prediction_all_fr, axis=1)

        prediction_ce_max = np.max(prediction_ce_fr, axis=1)
        prediction_all_max = np.max(prediction_all_fr, axis=1)

        result_dict = {
            "prediction_ce_f": scipy.stats.spearmanr(prediction_ce_f, measurement_list)[0],
            "prediction_all_f": scipy.stats.spearmanr(prediction_all_f, measurement_list)[0],
            "prediction_ce_r": scipy.stats.spearmanr(prediction_ce_r, measurement_list)[0],
            "prediction_all_r": scipy.stats.spearmanr(prediction_all_r, measurement_list)[0],
            "prediction_ce_mean": scipy.stats.spearmanr(prediction_ce_mean, measurement_list)[0],
            "prediction_all_mean": scipy.stats.spearmanr(prediction_all_mean, measurement_list)[0],
            "prediction_ce_min": scipy.stats.spearmanr(prediction_ce_min, measurement_list)[0],
            "prediction_all_min": scipy.stats.spearmanr(prediction_all_min, measurement_list)[0],
            "prediction_ce_max": scipy.stats.spearmanr(prediction_ce_max, measurement_list)[0],
            "prediction_all_max": scipy.stats.spearmanr(prediction_all_max, measurement_list)[0]
        }

        return result_dict

    def get_experimental_error(self, sess, conv_model, protein_name, step,
                               N_pred_iterations=10, batch_size=4, use_only_uppercase=True, use_family_embedding=False):

        prot_encoder = self.protein_names_to_one_hot_seqs_encoder[protein_name]
        prot_mask_encoder = self.protein_names_to_one_hot_seqs_encoder_mask[protein_name]
        prot_decoder_input = self.protein_names_to_one_hot_seqs_decoder_input[protein_name]
        prot_decoder_output = self.protein_names_to_one_hot_seqs_decoder_output[protein_name]
        prot_mask_decoder = self.protein_names_to_one_hot_seqs_decoder_mask[protein_name]

        measurement_list = self.protein_names_to_measurement_list[protein_name]

        # if only using the mutations from uppercase columns as in previous pubs
        if use_only_uppercase:
            idx_uppercase = self.protein_names_to_uppercase_idx[protein_name]

            prot_encoder = prot_encoder[idx_uppercase]
            prot_mask_encoder = prot_mask_encoder[idx_uppercase]
            prot_decoder_input = prot_decoder_input[idx_uppercase]
            prot_decoder_output = prot_decoder_output[idx_uppercase]
            prot_mask_decoder = prot_mask_decoder[idx_uppercase]
            measurement_list = measurement_list[idx_uppercase]

        prediction_matrix_ce = np.zeros((prot_encoder.shape[0], N_pred_iterations))
        prediction_matrix_all = np.zeros((prot_encoder.shape[0], N_pred_iterations))

        for i in range(N_pred_iterations):
            batch_order = np.arange(prot_encoder.shape[0])
            np.random.shuffle(batch_order)

            idx_family_one_hot = self.family_name_to_idx[protein_name]

            for j in range(0, prot_encoder.shape[0], batch_size):

                batch_index = batch_order[j:j + batch_size]
                batch_index_shape = prot_decoder_input[batch_index].shape[0]

                feed_dict = {conv_model.placeholders["sequences_start"]: prot_decoder_input[batch_index],
                             conv_model.placeholders["sequences_stop"]: prot_decoder_output[batch_index],
                             conv_model.placeholders["mask_decoder_1D"]: prot_mask_decoder[batch_index],
                             conv_model.placeholders["Neff"]: [
                                 float(len(self.family_name_to_sequence_decoder_output_list[protein_name]))],
                             conv_model.placeholders["step"]: [i]}

                if use_family_embedding:
                    family_one_hot = np.zeros(self.num_families)
                    family_one_hot[idx_family_one_hot] = 1.
                    feed_dict[conv_model.placeholders["family_embedding"]] = family_one_hot

                ce_loss, loss = sess.run([conv_model.tensors["cross_entropy_per_seq"],
                                          conv_model.tensors["loss_per_seq"]], feed_dict=feed_dict)

                for k, idx_batch in enumerate(batch_index.tolist()):
                    prediction_matrix_ce[idx_batch][i] = ce_loss[k]
                    prediction_matrix_all[idx_batch][i] = loss[k]

        prediction_ce = np.mean(prediction_matrix_ce, axis=1)
        prediction_all = np.mean(prediction_matrix_all, axis=1)

        return scipy.stats.spearmanr(prediction_ce, measurement_list)[0], \
               scipy.stats.spearmanr(prediction_all, measurement_list)[0]

    def read_in_test_data(self, input_filename):
        if not os.path.exists(input_filename) and input_filename.startswith('input/') and self.aws_util is not None:
            folder, filename = input_filename.rsplit('/', 1)
            if not self.aws_util.s3_get_file_grep(
                    s3_folder=f'calc_logprobs/{folder}',
                    dest_folder=f'{self.working_dir}/calc_logprobs/{folder}',
                    search_pattern=f'{filename}',
            ):
                raise Exception("Could not download test data from S3.")
        INPUT = open(input_filename, 'r')

        self.test_name_to_sequence = {}
        self.test_name_list = []
        first_time = True
        seq = ''
        name = ''
        for i, line in enumerate(INPUT):
            line = line.rstrip()
            if line[0] == '>' and first_time == True:
                first_time = False
                name = line
            elif line[0] == '>' and first_time == False:

                valid_seq = True
                for aa in seq:
                    if aa not in self.aa_dict:
                        valid_seq = False
                if valid_seq:
                    self.test_name_to_sequence[name] = seq
                    self.test_name_list.append(name)
                seq = ''
                name = line
            else:
                seq += line

        valid_seq = True
        for aa in seq:
            if aa not in self.aa_dict:
                valid_seq = False
        if valid_seq:
            self.test_name_to_sequence[name] = seq
            self.test_name_list.append(name)
        # self.test_name_to_sequence[name] = seq
        # self.test_name_list.append(name)

        INPUT.close()

    def output_log_probs(self, sess, model, output_filename, num_samples,
                         dropout_p, random_seed, channels, minibatch_size=100, Neff=100., step=200):

        num_seqs = len(self.test_name_list)
        batch_order = np.arange(num_seqs)

        self.mean_log_probs_arr = np.zeros((num_seqs, num_samples))
        self.forward_log_probs_arr = np.zeros((num_seqs, num_samples))
        self.reverse_log_probs_arr = np.zeros((num_seqs, num_samples))

        out_counter = 0
        for idx_iteration in range(num_samples):
            print("Iteration:", (idx_iteration + 1))
            mean_log_probs_list = []
            forward_log_probs_list = []
            reverse_log_probs_list = []
            for idx_batch in range(0, num_seqs, minibatch_size):
                start_time = time.time()
                batch_seq_name_list = self.test_name_list[idx_batch:idx_batch + minibatch_size]
                curr_minibatch_size = len(batch_seq_name_list)
                minibatch_max_seq_len = 0
                sequence_list = []
                for name in batch_seq_name_list:
                    sequence_list.append(self.test_name_to_sequence[name])
                    if len(self.test_name_to_sequence[name]) > minibatch_max_seq_len:
                        minibatch_max_seq_len = len(self.test_name_to_sequence[name])

                minibatch_max_seq_len += 1
                prot_decoder_input = np.zeros((curr_minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
                prot_decoder_output = np.zeros((curr_minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
                prot_decoder_input_r = np.zeros((curr_minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
                prot_decoder_output_r = np.zeros((curr_minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
                prot_mask_decoder = np.zeros((curr_minibatch_size, 1, minibatch_max_seq_len, 1))

                for i, sequence in enumerate(sequence_list):
                    decoder_input_seq = '*' + sequence
                    decoder_output_seq = sequence + '*'

                    sequence_r = sequence[::-1]
                    decoder_input_seq_r = '*' + sequence_r
                    decoder_output_seq_r = sequence_r + '*'

                    for j in range(len(decoder_input_seq)):
                        prot_decoder_input[i, 0, j, self.aa_dict[decoder_input_seq[j]]] = 1
                        prot_decoder_output[i, 0, j, self.aa_dict[decoder_output_seq[j]]] = 1
                        prot_decoder_input_r[i, 0, j, self.aa_dict[decoder_input_seq_r[j]]] = 1
                        prot_decoder_output_r[i, 0, j, self.aa_dict[decoder_output_seq_r[j]]] = 1
                        prot_mask_decoder[i, 0, j, 0] = 1

                feed_dict = {model.placeholders["sequences_start_f"]: prot_decoder_input,
                             model.placeholders["sequences_stop_f"]: prot_decoder_output,
                             model.placeholders["sequences_start_r"]: prot_decoder_input_r,
                             model.placeholders["sequences_stop_r"]: prot_decoder_output_r,
                             model.placeholders["mask_decoder_1D_f"]: prot_mask_decoder,
                             model.placeholders["mask_decoder_1D_r"]: prot_mask_decoder,
                             model.placeholders["Neff_f"]: [Neff], model.placeholders["Neff_r"]: [Neff],
                             model.placeholders["step"]: [step],
                             model.placeholders["dropout"]: dropout_p}

                ce_loss_f, ce_loss_r = sess.run([model.tensors["cross_entropy_per_seq_f"],
                                                 model.tensors["cross_entropy_per_seq_r"]], feed_dict=feed_dict)

                mean_log_probs_list += np.mean(np.concatenate(
                    [np.expand_dims(ce_loss_f, axis=1), np.expand_dims(ce_loss_r, axis=1), ], axis=1), axis=1).tolist()
                forward_log_probs_list += ce_loss_f.tolist()
                reverse_log_probs_list += ce_loss_r.tolist()
                out_counter += curr_minibatch_size

                if num_samples == 1:
                    print(str(out_counter) + " completed in " + str(time.time() - start_time) + " s")

            for idx_pred in range(num_seqs):
                self.mean_log_probs_arr[idx_pred, idx_iteration] = mean_log_probs_list[idx_pred]
                self.forward_log_probs_arr[idx_pred, idx_iteration] = forward_log_probs_list[idx_pred]
                self.reverse_log_probs_arr[idx_pred, idx_iteration] = reverse_log_probs_list[idx_pred]

        self.mean_log_probs_list = np.mean(self.mean_log_probs_arr, axis=1).tolist()
        self.forward_log_probs_list = np.mean(self.forward_log_probs_arr, axis=1).tolist()
        self.reverse_log_probs_list = np.mean(self.reverse_log_probs_arr, axis=1).tolist()

        OUTPUT = open(output_filename, 'w')
        header_list = ["mean", "bitperchar", "forward", "reverse"]
        header_list = [val + "-channels_" + str(channels) for val in header_list]
        if random_seed != -1:
            header_list = [val + "-rseed_" + str(random_seed) for val in header_list]
        header_list = ["name"] + header_list + ["sequence"]
        OUTPUT.write(",".join(header_list) + "\n")
        for i, name in enumerate(self.test_name_list):
            out_list = [name, self.mean_log_probs_list[i],
                        self.mean_log_probs_list[i] / float(len(self.test_name_to_sequence[name])),
                        self.forward_log_probs_list[i], self.reverse_log_probs_list[i],
                        self.test_name_to_sequence[name]]
            OUTPUT.write(",".join([str(val) for val in out_list]) + '\n')
        OUTPUT.close()


class DataHelperDoubleWeightingNanobody:
    def __init__(self, alignment_file='', focus_seq_name='',
                 mutation_file='', calc_weights=True, working_dir='.', alphabet_type='protein'):

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

        # then generate the experimental data
        self.gen_alignment_mut_data()

    def one_hot_3D(self, s):
        # One-hot encode as row vector
        x = np.zeros((len(s), len(self.alphabet)))
        for i, letter in enumerate(s):
            if letter in self.aa_dict:
                x[i, self.aa_dict[letter]] = 1
        return x

    def gen_alignment_mut_data(self):

        # Make a dictionary that goes from aa to a number for one-hot
        self.aa_dict = {}
        self.idx_to_aa = {}
        for i, aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i
            self.idx_to_aa[i] = aa

        # Do the inverse as well
        self.num_to_aa = {i: aa for aa, i in self.aa_dict.items()}

        ix = np.array([self.alphabet.find(s) for s in self.reorder_alphabet])

        self.name_to_sequence = {}
        self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists = {}
        self.cluster_id80_to_dict_of_cluster_id90_to_cluster_size = {}

        INPUT = open(self.working_dir + '/datasets/nanobodies/' + self.alignment_file, 'r')
        for line in INPUT:
            line = line.rstrip()

            if line != '' and line[0] == '>':
                line = line.rstrip()
                name, cluster_id80, cluster_id90 = line.split(':')

            elif line != '' and line[0] != '>':
                valid = True
                for letter in line:
                    if letter not in self.aa_dict:
                        valid = False
                if valid:
                    self.name_to_sequence[name] = line
                    if cluster_id80 in self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists:
                        if cluster_id90 in self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists[
                            cluster_id80]:
                            self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists[cluster_id80][
                                cluster_id90] += [name]
                            self.cluster_id80_to_dict_of_cluster_id90_to_cluster_size[cluster_id80][cluster_id90] += 1

                        else:
                            self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists[cluster_id80][
                                cluster_id90] = [name]
                            self.cluster_id80_to_dict_of_cluster_id90_to_cluster_size[cluster_id80][cluster_id90] = 1

                    else:
                        self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists[cluster_id80] = {
                            cluster_id90: [name]}
                        self.cluster_id80_to_dict_of_cluster_id90_to_cluster_size[cluster_id80] = {cluster_id90: 1}

        INPUT.close()

        self.cluster_id80_list = list(self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists.keys())
        print("Num clusters:", len(self.cluster_id80_list))

    def generate_one_family_minibatch_data(self, minibatch_size, use_embedding=True,
                                           reverse=False, matching=False, top_k=False):

        start = time.time()
        Neff = len(self.cluster_id80_list)
        cluster_name_list = np.random.choice(self.cluster_id80_list, minibatch_size).tolist()

        minibatch_max_seq_len = 0
        sequence_list = []
        for i, cluster_id80 in enumerate(cluster_name_list):

            # First pick a cluster id90 from the cluster id80s
            cluster_id90 = np.random.choice(
                list(self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists[cluster_id80].keys()), 1)[0]

            # Then pick a random sequence all in those clusters
            if self.cluster_id80_to_dict_of_cluster_id90_to_cluster_size[cluster_id80][cluster_id90] > 1:
                seq_name = np.random.choice(
                    self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists[cluster_id80][cluster_id90], 1)[0]
            else:
                seq_name = self.cluster_id80_to_dict_of_cluster_id90_to_sequence_name_lists[cluster_id80][cluster_id90][
                    0]

            # then grab the associated sequence
            seq = self.name_to_sequence[seq_name]

            sequence_list.append(seq)

            seq_len = len(seq)
            if seq_len > minibatch_max_seq_len:
                minibatch_max_seq_len = seq_len

        # Add 1 to compensate for the start and end character
        minibatch_max_seq_len += 1

        prot_decoder_output = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
        prot_decoder_input = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))

        if matching:
            prot_decoder_output_r = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
            prot_decoder_input_r = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))

        prot_mask_decoder = np.zeros((minibatch_size, 1, minibatch_max_seq_len, 1))
        family_one_hot = np.zeros(1)

        if use_embedding:
            family_one_hot[0] = 1.

        for i, sequence in enumerate(sequence_list):

            if reverse:
                sequence = sequence[::-1]

            decoder_input_seq = '*' + sequence
            decoder_output_seq = sequence + '*'

            if matching:
                sequence_r = sequence[::-1]
                decoder_input_seq_r = '*' + sequence_r
                decoder_output_seq_r = sequence_r + '*'

            for j in range(len(decoder_input_seq)):
                prot_decoder_input[i, 0, j, self.aa_dict[decoder_input_seq[j]]] = 1
                prot_decoder_output[i, 0, j, self.aa_dict[decoder_output_seq[j]]] = 1
                prot_mask_decoder[i, 0, j, 0] = 1

                if matching:
                    prot_decoder_input_r[i, 0, j, self.aa_dict[decoder_input_seq_r[j]]] = 1
                    prot_decoder_output_r[i, 0, j, self.aa_dict[decoder_output_seq_r[j]]] = 1

        if matching:
            return prot_decoder_input, prot_decoder_output, prot_mask_decoder, \
                   prot_decoder_input_r, prot_decoder_output_r, family_one_hot, Neff
        else:
            return prot_decoder_input, prot_decoder_output, prot_mask_decoder, \
                   family_one_hot, Neff

    def seq_list_to_one_hot(self, sequence_list):
        seq_arr = np.zeros((len(sequence_list), 1, len(sequence_list[0]), len(self.alphabet)))
        for i, seq in enumerate(sequence_list):
            for j, aa in enumerate(seq):
                k = self.aa_dict[aa]
                seq_arr[i, 0, j, k] = 1.
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
        for i, line in enumerate(INPUT):
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
        for idx_batch in range(0, num_seqs, minibatch_size):
            start_time = time.time()
            batch_seq_name_list = self.test_name_list[idx_batch:idx_batch + minibatch_size]
            curr_minibatch_size = len(batch_seq_name_list)
            minibatch_max_seq_len = 0
            sequence_list = []
            for name in batch_seq_name_list:
                sequence_list.append(self.test_name_to_sequence[name])
                if len(self.test_name_to_sequence[name]) > minibatch_max_seq_len:
                    minibatch_max_seq_len = len(self.test_name_to_sequence[name])

            minibatch_max_seq_len += 1
            prot_decoder_input = np.zeros((curr_minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
            prot_decoder_output = np.zeros((curr_minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
            prot_decoder_input_r = np.zeros((curr_minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
            prot_decoder_output_r = np.zeros((curr_minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
            prot_mask_decoder = np.zeros((curr_minibatch_size, 1, minibatch_max_seq_len, 1))

            for i, sequence in enumerate(sequence_list):
                decoder_input_seq = '*' + sequence
                decoder_output_seq = sequence + '*'

                sequence_r = sequence[::-1]
                decoder_input_seq_r = '*' + sequence_r
                decoder_output_seq_r = sequence_r + '*'

                for j in range(len(decoder_input_seq)):
                    prot_decoder_input[i, 0, j, self.aa_dict[decoder_input_seq[j]]] = 1
                    prot_decoder_output[i, 0, j, self.aa_dict[decoder_output_seq[j]]] = 1
                    prot_decoder_input_r[i, 0, j, self.aa_dict[decoder_input_seq_r[j]]] = 1
                    prot_decoder_output_r[i, 0, j, self.aa_dict[decoder_output_seq_r[j]]] = 1
                    prot_mask_decoder[i, 0, j, 0] = 1

            feed_dict = {model.placeholders["sequences_start_f"]: prot_decoder_input,
                         model.placeholders["sequences_stop_f"]: prot_decoder_output,
                         model.placeholders["sequences_start_r"]: prot_decoder_input_r,
                         model.placeholders["sequences_stop_r"]: prot_decoder_output_r,
                         model.placeholders["mask_decoder_1D_f"]: prot_mask_decoder,
                         model.placeholders["mask_decoder_1D_r"]: prot_mask_decoder,
                         model.placeholders["Neff_f"]: [Neff], model.placeholders["Neff_r"]: [Neff],
                         model.placeholders["step"]: [200],
                         model.placeholders["dropout"]: 1.}

            ce_loss_f, ce_loss_r = sess.run([model.tensors["cross_entropy_per_seq_f"],
                                             model.tensors["cross_entropy_per_seq_r"]], feed_dict=feed_dict)

            self.mean_log_probs_list += np.mean(np.concatenate(
                [np.expand_dims(ce_loss_f, axis=1), np.expand_dims(ce_loss_r, axis=1), ], axis=1), axis=1).tolist()
            self.forward_log_probs_list += ce_loss_f.tolist()
            self.reverse_log_prob_list += ce_loss_r.tolist()
            out_counter += curr_minibatch_size
            print(str(out_counter) + " completed in " + str(time.time() - start_time) + " s")

        OUTPUT = open(output_filename, 'w')
        OUTPUT.write("name,mean,mean/L,forward,reverse,sequence\n")
        for i, name in enumerate(self.test_name_list):
            out_list = [name, self.mean_log_probs_list[i],
                        self.mean_log_probs_list[i] / float(len(self.test_name_to_sequence[name])),
                        self.forward_log_probs_list[i], self.reverse_log_prob_list[i], self.test_name_to_sequence[name]]
            OUTPUT.write(",".join([str(val) for val in out_list]) + '\n')
        OUTPUT.close()


class DataHelperNanobody:
    def __init__(self, alignment_file='', focus_seq_name='',
                 mutation_file='', calc_weights=True, working_dir='.', alphabet_type='protein'):

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

        # then generate the experimental data
        self.gen_alignment_mut_data()

    def one_hot_3D(self, s):
        # One-hot encode as row vector
        x = np.zeros((len(s), len(self.alphabet)))
        for i, letter in enumerate(s):
            if letter in self.aa_dict:
                x[i, self.aa_dict[letter]] = 1
        return x

    def gen_alignment_mut_data(self):

        # Make a dictionary that goes from aa to a number for one-hot
        self.aa_dict = {}
        self.idx_to_aa = {}
        for i, aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i
            self.idx_to_aa[i] = aa

        # Do the inverse as well
        self.num_to_aa = {i: aa for aa, i in self.aa_dict.items()}

        ix = np.array([self.alphabet.find(s) for s in self.reorder_alphabet])

        self.name_to_sequence = {}
        self.cluster_id_to_sequence_name_lists = {}

        INPUT = open(self.working_dir + '/datasets/' + self.alignment_file, 'r')
        for line in INPUT:
            line = line.rstrip()

            if line != '' and line[0] == '>':
                line = line.rstrip()
                name, cluster = line.split(':')

            elif line != '' and line[0] != '>':
                valid = True
                for letter in line:
                    if letter not in self.aa_dict:
                        valid = False
                if valid:
                    self.name_to_sequence[name] = line
                    if cluster in self.cluster_id_to_sequence_name_lists:
                        self.cluster_id_to_sequence_name_lists[cluster].append(name)
                    else:
                        self.cluster_id_to_sequence_name_lists[cluster] = [name]
        INPUT.close()

        self.cluster_list = list(self.cluster_id_to_sequence_name_lists.keys())
        print("Num clusters:", len(self.cluster_list))

    def generate_one_family_minibatch_data(self, minibatch_size, use_embedding=True,
                                           reverse=False, matching=False, top_k=False):

        Neff = len(self.cluster_list)
        cluster_name_list = np.random.choice(self.cluster_list, minibatch_size).tolist()

        minibatch_max_seq_len = 0
        sequence_list = []
        for i, clust in enumerate(cluster_name_list):

            # pick a sequence name from the cluster
            seq_name = np.random.choice(self.cluster_id_to_sequence_name_lists[clust], 1)[0]

            # then grab the associated sequence
            seq = self.name_to_sequence[seq_name]

            sequence_list.append(seq)

            seq_len = len(seq)
            if seq_len > minibatch_max_seq_len:
                minibatch_max_seq_len = seq_len

        # Add 1 to compensate for the start and end character
        minibatch_max_seq_len += 1

        prot_decoder_output = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
        prot_decoder_input = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))

        if matching:
            prot_decoder_output_r = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
            prot_decoder_input_r = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))

        prot_mask_decoder = np.zeros((minibatch_size, 1, minibatch_max_seq_len, 1))
        family_one_hot = np.zeros(1)

        if use_embedding:
            family_one_hot[0] = 1.

        for i, sequence in enumerate(sequence_list):

            if reverse:
                sequence = sequence[::-1]

            decoder_input_seq = '*' + sequence
            decoder_output_seq = sequence + '*'

            if matching:
                sequence_r = sequence[::-1]
                decoder_input_seq_r = '*' + sequence_r
                decoder_output_seq_r = sequence_r + '*'

            for j in range(len(decoder_input_seq)):
                prot_decoder_input[i, 0, j, self.aa_dict[decoder_input_seq[j]]] = 1
                prot_decoder_output[i, 0, j, self.aa_dict[decoder_output_seq[j]]] = 1
                prot_mask_decoder[i, 0, j, 0] = 1

                if matching:
                    prot_decoder_input_r[i, 0, j, self.aa_dict[decoder_input_seq_r[j]]] = 1
                    prot_decoder_output_r[i, 0, j, self.aa_dict[decoder_output_seq_r[j]]] = 1

        if matching:
            return prot_decoder_input, prot_decoder_output, prot_mask_decoder, \
                   prot_decoder_input_r, prot_decoder_output_r, family_one_hot, Neff
        else:
            return prot_decoder_input, prot_decoder_output, prot_mask_decoder, \
                   family_one_hot, Neff

    def seq_list_to_one_hot(self, sequence_list):
        seq_arr = np.zeros((len(sequence_list), 1, len(sequence_list[0]), len(self.alphabet)))
        for i, seq in enumerate(sequence_list):
            for j, aa in enumerate(seq):
                k = self.aa_dict[aa]
                seq_arr[i, 0, j, k] = 1.
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


class DataHelperMultifamDynamicBatches:
    def __init__(self, dataset='', alignment_file='', focus_seq_name='',
                 mutation_file='', calc_weights=True, working_dir='.', theta=0.2,
                 load_all_sequences=True, alphabet_type='protein', max_seq_len=-1,
                 longest_entry=125):

        np.random.seed(42)
        self.dataset = dataset
        self.alignment_file = alignment_file
        self.focus_seq_name = focus_seq_name
        self.mutation_file = mutation_file
        self.working_dir = working_dir
        self.calc_weights = calc_weights
        self.alphabet_type = alphabet_type
        self.max_seq_len = max_seq_len
        # Alignment processing parameters
        self.theta = theta

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

        # then generate the experimental data
        self.gen_alignment_mut_data()

    def one_hot_3D(self, s):
        # One-hot encode as row vector
        x = np.zeros((len(s), len(self.alphabet)))
        for i, letter in enumerate(s):
            if letter in self.aa_dict:
                x[i, self.aa_dict[letter]] = 1
        return x

    def gen_alignment_mut_data(self):

        # Make a dictionary that goes from aa to a number for one-hot
        self.aa_dict = {}
        for i, aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        # Do the inverse as well
        self.num_to_aa = {i: aa for aa, i in self.aa_dict.items()}

        ix = np.array([self.alphabet.find(s) for s in self.reorder_alphabet])

        self.family_name_to_sequence_encoder_list = {}
        self.family_name_to_sequence_decoder_input_list = {}
        self.family_name_to_sequence_decoder_output_list = {}
        self.family_name_to_sequence_weight_list = {}
        self.family_name_to_Neff = {}
        self.family_name_list = []
        self.family_idx_list = []
        self.family_name = ''

        # for now, we will make all the sequences have the same length of
        #   encoded matrices, though this is wasteful

        max_seq_len = 0
        max_family_size = 0
        for filename in glob.glob(self.working_dir + '/datasets/sequences/*.fa'):
            INPUT = open(filename, 'r')

            encoder_sequence_list = []
            decoder_input_sequence_list = []
            decoder_output_sequence_list = []
            weight_list = []

            family_name_list = filename.split('/')[-1].split('_')
            family_name = family_name_list[0] + '_' + family_name_list[1]

            print(family_name)

            family_size = 0
            ind_family_idx_list = []
            first_time = True
            seq = ''

            for line in INPUT:
                line = line.rstrip()

                if line != '':
                    if line[0] == '>' and first_time:
                        weight = float(line.split(':')[-1])
                        first_time = False

                    elif line[0] == '>' and first_time == False:
                        valid = True
                        for letter in seq:
                            if letter not in self.aa_dict:
                                valid = False
                        if valid:
                            encoder_sequence_list.append(seq)
                            decoder_input_sequence_list.append('*' + seq)
                            decoder_output_sequence_list.append(seq + '*')
                            ind_family_idx_list.append(family_size)
                            weight_list.append(weight)

                            family_size += 1
                            if len(seq) > max_seq_len:
                                max_seq_len = len(seq)

                        seq = ''
                        weight = float(line.split(':')[-1])

                    else:
                        seq += line

            INPUT.close()

            valid = True
            for letter in seq:
                if letter not in self.aa_dict:
                    valid = False
            if valid:
                encoder_sequence_list.append(seq)
                decoder_input_sequence_list.append('*' + seq)
                decoder_output_sequence_list.append(seq + '*')
                ind_family_idx_list.append(family_size)
                weight_list.append(weight)

                family_size += 1
                if len(seq) > max_seq_len:
                    max_seq_len = len(seq)

            if family_size > max_family_size:
                max_family_size = family_size

            self.family_name_to_sequence_encoder_list[family_name] = encoder_sequence_list
            self.family_name_to_sequence_decoder_input_list[family_name] = decoder_input_sequence_list
            self.family_name_to_sequence_decoder_output_list[family_name] = decoder_output_sequence_list
            self.family_name_to_sequence_weight_list[family_name] = (
                        np.asarray(weight_list) / np.sum(weight_list)).tolist()
            self.family_name_to_Neff[family_name] = np.sum(weight_list)
            self.family_name = family_name

            self.family_name_list.append(family_name)
            self.family_idx_list.append(ind_family_idx_list)

        self.seq_len = max_seq_len
        self.num_families = len(self.family_name_list)
        self.max_family_size = max_family_size

        print("Number of families:", self.num_families)
        print("Neff:", np.sum(weight_list))

        print('Encoding sequences')
        self.family_name_to_idx = {}
        self.idx_to_family_name = {}

        for i, family_name in enumerate(self.family_name_list):
            self.family_name_to_idx[family_name] = i
            self.idx_to_family_name[i] = family_name

        # Then read in the mutation data so we can predict it later
        self.protein_mutation_names = []
        self.protein_names_to_uppercase_idx = {}
        self.protein_names_to_one_hot_seqs_encoder = {}
        self.mut_protein_names_to_seqs_encoder = {}
        self.protein_names_to_one_hot_seqs_decoder_input_f = {}
        self.protein_names_to_one_hot_seqs_decoder_output_f = {}
        self.protein_names_to_one_hot_seqs_decoder_input_r = {}
        self.protein_names_to_one_hot_seqs_decoder_output_r = {}
        self.protein_names_to_one_hot_seqs_encoder_mask = {}
        self.protein_names_to_one_hot_seqs_decoder_mask = {}
        self.protein_names_to_measurement_list = {}
        self.protein_names_to_mutation_list = {}

        for filename in glob.glob(self.working_dir + '/datasets/mutation_data/*.csv'):

            INPUT = open(filename, 'r')

            encoder_sequence_list = []
            decoder_input_sequence_list = []
            decoder_output_sequence_list = []
            mutation_list = []
            uppercase_list = []
            measurement_list = []

            family_name_list = filename.split('/')[-1].split('_')
            family_name = family_name_list[0] + '_' + family_name_list[1]

            mutation_counter = 0
            for i, line in enumerate(INPUT):
                line = line.rstrip()
                line_list = line.split(',')
                if i != 0:
                    mutation, is_upper, measurement, sequence = line_list

                    measurement = float(measurement)
                    if np.isfinite(measurement):
                        if is_upper == 'True':
                            uppercase_list.append(mutation_counter)
                        mutation_list.append(mutation)
                        measurement_list.append(float(measurement))

                        encoder_sequence_list.append(sequence)
                        decoder_input_sequence_list.append('*' + sequence)
                        decoder_output_sequence_list.append(sequence + '*')
                        seq_len_mutations = len(sequence)
                        mutation_counter += 1

            INPUT.close()

            self.protein_mutation_names.append(family_name)
            self.protein_names_to_uppercase_idx[family_name] = np.asarray(uppercase_list)
            self.protein_names_to_measurement_list[family_name] = np.asarray(measurement_list)
            self.protein_names_to_mutation_list[family_name] = mutation_list
            self.mut_protein_names_to_seqs_encoder[family_name] = encoder_sequence_list

            prot_encoder = np.zeros((len(measurement_list), 1, seq_len_mutations, len(self.alphabet)))
            prot_mask_encoder = np.zeros((len(measurement_list), 1, seq_len_mutations, 1))
            prot_decoder_output_f = np.zeros((len(measurement_list), 1, seq_len_mutations + 1, len(self.alphabet)))
            prot_decoder_input_f = np.zeros((len(measurement_list), 1, seq_len_mutations + 1, len(self.alphabet)))
            prot_decoder_output_r = np.zeros((len(measurement_list), 1, seq_len_mutations + 1, len(self.alphabet)))
            prot_decoder_input_r = np.zeros((len(measurement_list), 1, seq_len_mutations + 1, len(self.alphabet)))
            prot_mask_decoder = np.zeros((len(measurement_list), 1, seq_len_mutations + 1, 1))

            for j, sequence in enumerate(encoder_sequence_list):
                for k, letter in enumerate(sequence):
                    if letter in self.aa_dict:
                        l = self.aa_dict[letter]
                        prot_encoder[j, 0, k, l] = 1.0
                        prot_mask_encoder[j, 0, k, 0] = 1.0

            for j, sequence in enumerate(decoder_output_sequence_list):
                for k, letter in enumerate(sequence):
                    if letter in self.aa_dict:
                        l = self.aa_dict[letter]
                        prot_decoder_output_f[j, 0, k, l] = 1.0
                        prot_mask_decoder[j, 0, k, 0] = 1.0

            for j, sequence in enumerate(decoder_input_sequence_list):
                for k, letter in enumerate(sequence):
                    if letter in self.aa_dict:
                        l = self.aa_dict[letter]
                        prot_decoder_input_f[j, 0, k, l] = 1.0

            for j, sequence in enumerate(encoder_sequence_list):
                sequence_r = "*" + sequence[::-1]
                for k, letter in enumerate(sequence_r):
                    if letter in self.aa_dict:
                        l = self.aa_dict[letter]
                        prot_decoder_input_r[j, 0, k, l] = 1.0

            for j, sequence in enumerate(encoder_sequence_list):
                sequence_r = sequence[::-1] + '*'
                for k, letter in enumerate(sequence_r):
                    if letter in self.aa_dict:
                        l = self.aa_dict[letter]
                        prot_decoder_output_r[j, 0, k, l] = 1.0

            self.protein_names_to_one_hot_seqs_encoder[family_name] = prot_encoder
            self.protein_names_to_one_hot_seqs_encoder_mask[family_name] = prot_mask_encoder
            self.protein_names_to_one_hot_seqs_decoder_input_f[family_name] = prot_decoder_input_f
            self.protein_names_to_one_hot_seqs_decoder_output_f[family_name] = prot_decoder_output_f
            self.protein_names_to_one_hot_seqs_decoder_input_r[family_name] = prot_decoder_input_r
            self.protein_names_to_one_hot_seqs_decoder_output_r[family_name] = prot_decoder_output_r
            self.protein_names_to_one_hot_seqs_decoder_mask[family_name] = prot_mask_decoder

    def generate_minibatch_idx(self, minibatch_size):

        # First choose
        family_idx = np.random.choice(len(self.family_idx_list), minibatch_size)

        one_hot_label = np.zeros((minibatch_size, len(self.family_idx_list)))

        sequence_idx = []
        for i, idx in enumerate(family_idx):
            sequence_idx.append(np.random.choice(self.family_idx_list[idx], 1)[0])
            one_hot_label[i][idx] = 1.

        return family_idx, sequence_idx, one_hot_label

    def gen_target_aa(self, x_input, x_input_lengths, dims):

        target_aa = []
        query_seq = np.copy(x_input)
        for i, input_len in enumerate(x_input_lengths):
            idx_rand = np.random.randint(input_len)
            target_aa.append(query_seq[i, 0, idx_rand].flatten().tolist())
            query_seq[i, 0, idx_rand] *= 0.
            query_seq[i, 0, idx_rand, dims["alphabet"] - 1] = 1.

        return query_seq, np.asarray(target_aa)

    def generate_one_family_minibatch_data(self, minibatch_size, use_embedding=True,
                                           reverse=False, matching=False, top_k=False):

        # First choose which families
        family_idx = np.random.choice(len(self.family_idx_list), 1)[0]
        family_name = self.idx_to_family_name[family_idx]
        batch_order = np.arange(len(self.family_name_to_sequence_decoder_output_list[family_name]))
        family_weights = self.family_name_to_sequence_weight_list[family_name]
        Neff = self.family_name_to_Neff[family_name]

        # Then choose which sequences to grab in those families
        if top_k:
            sequence_idx = minibatch_size * [0]
        else:
            sequence_idx = np.random.choice(batch_order, minibatch_size, p=family_weights).tolist()

        minibatch_max_seq_len = 0
        for i, idx_seq in enumerate(sequence_idx):
            # print i,idx_seq
            seq = self.family_name_to_sequence_decoder_output_list[family_name][idx_seq]

            seq_len = len(seq)
            if seq_len > minibatch_max_seq_len:
                minibatch_max_seq_len = seq_len

        prot_decoder_output = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
        prot_decoder_input = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))

        if matching:
            prot_decoder_output_r = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
            prot_decoder_input_r = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))

        prot_mask_decoder = np.zeros((minibatch_size, 1, minibatch_max_seq_len, 1))
        family_one_hot = np.zeros(self.num_families)

        if use_embedding:
            family_one_hot[family_idx] = 1.

        for i, idx_seq in enumerate(sequence_idx):
            sequence = self.family_name_to_sequence_encoder_list[family_name][idx_seq]
            if reverse:
                sequence = sequence[::-1]

            decoder_input_seq = '*' + sequence
            decoder_output_seq = sequence + '*'

            if matching:
                sequence_r = sequence[::-1]
                decoder_input_seq_r = '*' + sequence_r
                decoder_output_seq_r = sequence_r + '*'

            for j in range(len(decoder_input_seq)):
                prot_decoder_input[i, 0, j, self.aa_dict[decoder_input_seq[j]]] = 1
                prot_decoder_output[i, 0, j, self.aa_dict[decoder_output_seq[j]]] = 1
                prot_mask_decoder[i, 0, j, 0] = 1

                if matching:
                    prot_decoder_input_r[i, 0, j, self.aa_dict[decoder_input_seq_r[j]]] = 1
                    prot_decoder_output_r[i, 0, j, self.aa_dict[decoder_output_seq_r[j]]] = 1

        if matching:
            return prot_decoder_input, prot_decoder_output, prot_mask_decoder, \
                   prot_decoder_input_r, prot_decoder_output_r, family_one_hot, Neff
        else:
            return prot_decoder_input, prot_decoder_output, prot_mask_decoder, \
                   family_one_hot, Neff

    def generate_minibatch_data(self, minibatch_size):

        # First choose which families
        family_idx = np.random.choice(len(self.family_idx_list), minibatch_size)

        # Then choose which sequences to grab in those families
        sequence_idx = []
        minibatch_max_seq_len = 0
        for i, idx in enumerate(family_idx):

            idx_seq = np.random.choice(self.family_idx_list[idx], 1)[0]
            family_name = self.idx_to_family_name[idx]
            seq = self.family_name_to_sequence_decoder_output_list[family_name][idx_seq]

            seq_len = len(seq)
            if seq_len > minibatch_max_seq_len:
                minibatch_max_seq_len = seq_len

            sequence_idx.append(idx_seq)

        prot_decoder_output = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
        prot_decoder_input = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
        prot_mask_decoder = np.zeros((minibatch_size, 1, minibatch_max_seq_len, 1))

        for i, idx in enumerate(family_idx):

            family_name = self.idx_to_family_name[idx]
            idx_seq = sequence_idx[i]

            decoder_input_seq = self.family_name_to_sequence_decoder_input_list[family_name][idx_seq]
            decoder_output_seq = self.family_name_to_sequence_decoder_output_list[family_name][idx_seq]

            for j in range(len(decoder_input_seq)):
                prot_decoder_input[i, 0, j, self.aa_dict[decoder_input_seq[j]]] = 1
                prot_decoder_output[i, 0, j, self.aa_dict[decoder_output_seq[j]]] = 1
                prot_mask_decoder[i, 0, j, 0] = 1

        return prot_decoder_input, prot_decoder_output, prot_mask_decoder

    def get_experimental_error_single_family_fr(self, sess, conv_model, protein_name, step, batch_size=30, N_pred_iterations=10,
                                                dropout_p=1., use_only_uppercase=True):

        prot_encoder = self.protein_names_to_one_hot_seqs_encoder[protein_name]
        prot_mask_encoder = self.protein_names_to_one_hot_seqs_encoder_mask[protein_name]
        prot_decoder_input_f = self.protein_names_to_one_hot_seqs_decoder_input_f[protein_name]
        prot_decoder_output_f = self.protein_names_to_one_hot_seqs_decoder_output_f[protein_name]
        prot_decoder_input_r = self.protein_names_to_one_hot_seqs_decoder_input_r[protein_name]
        prot_decoder_output_r = self.protein_names_to_one_hot_seqs_decoder_output_r[protein_name]
        prot_mask_decoder = self.protein_names_to_one_hot_seqs_decoder_mask[protein_name]

        measurement_list = self.protein_names_to_measurement_list[protein_name]

        # if only using the mutations from uppercase columns as in previous pubs
        if use_only_uppercase:
            idx_uppercase = self.protein_names_to_uppercase_idx[protein_name]

            prot_encoder = prot_encoder[idx_uppercase]
            prot_mask_encoder = prot_mask_encoder[idx_uppercase]
            prot_decoder_input_f = prot_decoder_input_f[idx_uppercase]
            prot_decoder_output_f = prot_decoder_output_f[idx_uppercase]
            prot_decoder_input_r = prot_decoder_input_r[idx_uppercase]
            prot_decoder_output_r = prot_decoder_output_r[idx_uppercase]
            prot_mask_decoder = prot_mask_decoder[idx_uppercase]
            measurement_list = measurement_list[idx_uppercase]

        prediction_matrix_ce_f = np.zeros((len(prot_encoder), N_pred_iterations))
        prediction_matrix_all_f = np.zeros((len(prot_encoder), N_pred_iterations))

        prediction_matrix_ce_r = np.zeros((len(prot_encoder), N_pred_iterations))
        prediction_matrix_all_r = np.zeros((len(prot_encoder), N_pred_iterations))

        for i in range(N_pred_iterations):
            batch_order = np.arange(prot_encoder.shape[0])
            np.random.shuffle(batch_order)

            idx_family_one_hot = self.family_name_to_idx[protein_name]

            for j in range(0, prot_encoder.shape[0], batch_size):

                batch_index = batch_order[j:j + batch_size]
                batch_index_shape = prot_decoder_input_f[batch_index].shape[0]

                feed_dict = {conv_model.placeholders["sequences_start_f"]: prot_decoder_input_f[batch_index],
                             conv_model.placeholders["sequences_stop_f"]: prot_decoder_output_f[batch_index],
                             conv_model.placeholders["sequences_start_r"]: prot_decoder_input_r[batch_index],
                             conv_model.placeholders["sequences_stop_r"]: prot_decoder_output_r[batch_index],
                             conv_model.placeholders["mask_decoder_1D_f"]: prot_mask_decoder[batch_index],
                             conv_model.placeholders["mask_decoder_1D_r"]: prot_mask_decoder[batch_index],
                             conv_model.placeholders["Neff_f"]: [
                                 float(len(self.family_name_to_sequence_decoder_output_list[protein_name]))],
                             conv_model.placeholders["Neff_r"]: [
                                 float(len(self.family_name_to_sequence_decoder_output_list[protein_name]))],
                             conv_model.placeholders["step"]: [i],
                             conv_model.placeholders["dropout"]: dropout_p}

                ce_loss_f, loss_f, ce_loss_r, loss_r = sess.run([conv_model.tensors["cross_entropy_per_seq_f"],
                                                                 conv_model.tensors["loss_per_seq_f"],
                                                                 conv_model.tensors["cross_entropy_per_seq_r"],
                                                                 conv_model.tensors["loss_per_seq_r"]],
                                                                feed_dict=feed_dict)

                for k, idx_batch in enumerate(batch_index.tolist()):
                    prediction_matrix_ce_f[idx_batch][i] = ce_loss_f[k]
                    prediction_matrix_all_f[idx_batch][i] = loss_f[k]
                    prediction_matrix_ce_r[idx_batch][i] = ce_loss_r[k]
                    prediction_matrix_all_r[idx_batch][i] = loss_r[k]

        prediction_ce_f = np.mean(prediction_matrix_ce_f, axis=1)
        prediction_all_f = np.mean(prediction_matrix_all_f, axis=1)

        prediction_ce_r = np.mean(prediction_matrix_ce_r, axis=1)
        prediction_all_r = np.mean(prediction_matrix_all_r, axis=1)

        prediction_ce_fr = np.concatenate(
            [np.expand_dims(prediction_ce_f, axis=-1), np.expand_dims(prediction_ce_r, axis=-1)], axis=-1)
        prediction_all_fr = np.concatenate(
            [np.expand_dims(prediction_all_f, axis=-1), np.expand_dims(prediction_all_r, axis=-1)], axis=-1)

        prediction_ce_mean = np.mean(prediction_ce_fr, axis=1)
        prediction_all_mean = np.mean(prediction_ce_fr, axis=1)

        prediction_ce_min = np.min(prediction_ce_fr, axis=1)
        prediction_all_min = np.min(prediction_all_fr, axis=1)

        prediction_ce_max = np.max(prediction_ce_fr, axis=1)
        prediction_all_max = np.max(prediction_all_fr, axis=1)

        result_dict = {}

        result_dict["prediction_ce_f"] = scipy.stats.spearmanr(prediction_ce_f, measurement_list)[0]
        result_dict["prediction_all_f"] = scipy.stats.spearmanr(prediction_all_f, measurement_list)[0]

        result_dict["prediction_ce_r"] = scipy.stats.spearmanr(prediction_ce_r, measurement_list)[0]
        result_dict["prediction_all_r"] = scipy.stats.spearmanr(prediction_all_r, measurement_list)[0]

        result_dict["prediction_ce_mean"] = scipy.stats.spearmanr(prediction_ce_mean, measurement_list)[0]
        result_dict["prediction_all_mean"] = scipy.stats.spearmanr(prediction_all_mean, measurement_list)[0]

        result_dict["prediction_ce_min"] = scipy.stats.spearmanr(prediction_ce_min, measurement_list)[0]
        result_dict["prediction_all_min"] = scipy.stats.spearmanr(prediction_all_min, measurement_list)[0]

        result_dict["prediction_ce_max"] = scipy.stats.spearmanr(prediction_ce_max, measurement_list)[0]
        result_dict["prediction_all_max"] = scipy.stats.spearmanr(prediction_all_max, measurement_list)[0]

        return result_dict

    def get_experimental_error_fr(self, sess, conv_model, protein_name, step, N_pred_iterations=10, batch_size=4, use_only_uppercase=True,
                                  family_embedding_model=True, use_embedding=True):

        prot_encoder = self.protein_names_to_one_hot_seqs_encoder[protein_name]
        prot_mask_encoder = self.protein_names_to_one_hot_seqs_encoder_mask[protein_name]
        prot_decoder_input_f = self.protein_names_to_one_hot_seqs_decoder_input_f[protein_name]
        prot_decoder_output_f = self.protein_names_to_one_hot_seqs_decoder_output_f[protein_name]
        prot_decoder_input_r = self.protein_names_to_one_hot_seqs_decoder_input_r[protein_name]
        prot_decoder_output_r = self.protein_names_to_one_hot_seqs_decoder_output_r[protein_name]
        prot_mask_decoder = self.protein_names_to_one_hot_seqs_decoder_mask[protein_name]

        measurement_list = self.protein_names_to_measurement_list[protein_name]

        # if only using the mutations from uppercase columns as in previous pubs
        if use_only_uppercase:
            idx_uppercase = self.protein_names_to_uppercase_idx[protein_name]

            prot_encoder = prot_encoder[idx_uppercase]
            prot_mask_encoder = prot_mask_encoder[idx_uppercase]
            prot_decoder_input_f = prot_decoder_input_f[idx_uppercase]
            prot_decoder_output_f = prot_decoder_output_f[idx_uppercase]
            prot_decoder_input_r = prot_decoder_input_r[idx_uppercase]
            prot_decoder_output_r = prot_decoder_output_r[idx_uppercase]
            prot_mask_decoder = prot_mask_decoder[idx_uppercase]
            measurement_list = measurement_list[idx_uppercase]

        prediction_matrix_ce_f = np.zeros((len(prot_encoder), N_pred_iterations))
        prediction_matrix_all_f = np.zeros((len(prot_encoder), N_pred_iterations))

        prediction_matrix_ce_r = np.zeros((len(prot_encoder), N_pred_iterations))
        prediction_matrix_all_r = np.zeros((len(prot_encoder), N_pred_iterations))

        for i in range(N_pred_iterations):
            batch_order = np.arange(prot_encoder.shape[0])
            np.random.shuffle(batch_order)

            idx_family_one_hot = self.family_name_to_idx[protein_name]

            for j in range(0, prot_encoder.shape[0], batch_size):

                batch_index = batch_order[j:j + batch_size]
                batch_index_shape = prot_decoder_input_f[batch_index].shape[0]

                feed_dict = {conv_model.placeholders["sequences_start_f"]: prot_decoder_input_f[batch_index],
                             conv_model.placeholders["sequences_stop_f"]: prot_decoder_output_f[batch_index],
                             conv_model.placeholders["sequences_start_r"]: prot_decoder_input_r[batch_index],
                             conv_model.placeholders["sequences_stop_r"]: prot_decoder_output_r[batch_index],
                             conv_model.placeholders["mask_decoder_1D_f"]: prot_mask_decoder[batch_index],
                             conv_model.placeholders["mask_decoder_1D_r"]: prot_mask_decoder[batch_index],
                             conv_model.placeholders["Neff_f"]: [
                                 float(len(self.family_name_to_sequence_decoder_output_list[protein_name]))],
                             conv_model.placeholders["Neff_r"]: [
                                 float(len(self.family_name_to_sequence_decoder_output_list[protein_name]))],
                             conv_model.placeholders["step"]: [i]}

                if family_embedding_model:
                    family_one_hot = np.zeros(self.num_families)
                    if use_embedding:
                        family_one_hot[idx_family_one_hot] = 1.
                    feed_dict[conv_model.placeholders["family_embedding_f"]] = family_one_hot
                    feed_dict[conv_model.placeholders["family_embedding_r"]] = family_one_hot

                ce_loss_f, loss_f, ce_loss_r, loss_r = sess.run([conv_model.tensors["cross_entropy_per_seq_f"],
                                                                 conv_model.tensors["loss_per_seq_f"],
                                                                 conv_model.tensors["cross_entropy_per_seq_r"],
                                                                 conv_model.tensors["loss_per_seq_r"]],
                                                                feed_dict=feed_dict)

                for k, idx_batch in enumerate(batch_index.tolist()):
                    prediction_matrix_ce_f[idx_batch][i] = ce_loss_f[k]
                    prediction_matrix_all_f[idx_batch][i] = loss_f[k]
                    prediction_matrix_ce_r[idx_batch][i] = ce_loss_r[k]
                    prediction_matrix_all_r[idx_batch][i] = loss_r[k]

        prediction_ce_f = np.mean(prediction_matrix_ce_f, axis=1)
        prediction_all_f = np.mean(prediction_matrix_all_f, axis=1)

        prediction_ce_r = np.mean(prediction_matrix_ce_r, axis=1)
        prediction_all_r = np.mean(prediction_matrix_all_r, axis=1)

        prediction_ce_fr = np.concatenate(
            [np.expand_dims(prediction_ce_f, axis=-1), np.expand_dims(prediction_ce_r, axis=-1)], axis=-1)
        prediction_all_fr = np.concatenate(
            [np.expand_dims(prediction_all_f, axis=-1), np.expand_dims(prediction_all_r, axis=-1)], axis=-1)

        prediction_ce_mean = np.mean(prediction_ce_fr, axis=1)
        prediction_all_mean = np.mean(prediction_ce_fr, axis=1)

        prediction_ce_min = np.min(prediction_ce_fr, axis=1)
        prediction_all_min = np.min(prediction_all_fr, axis=1)

        prediction_ce_max = np.max(prediction_ce_fr, axis=1)
        prediction_all_max = np.max(prediction_all_fr, axis=1)

        result_dict = {}

        result_dict["prediction_ce_f"] = scipy.stats.spearmanr(prediction_ce_f, measurement_list)[0]
        result_dict["prediction_all_f"] = scipy.stats.spearmanr(prediction_all_f, measurement_list)[0]

        result_dict["prediction_ce_r"] = scipy.stats.spearmanr(prediction_ce_r, measurement_list)[0]
        result_dict["prediction_all_r"] = scipy.stats.spearmanr(prediction_all_r, measurement_list)[0]

        result_dict["prediction_ce_mean"] = scipy.stats.spearmanr(prediction_ce_mean, measurement_list)[0]
        result_dict["prediction_all_mean"] = scipy.stats.spearmanr(prediction_all_mean, measurement_list)[0]

        result_dict["prediction_ce_min"] = scipy.stats.spearmanr(prediction_ce_min, measurement_list)[0]
        result_dict["prediction_all_min"] = scipy.stats.spearmanr(prediction_all_min, measurement_list)[0]

        result_dict["prediction_ce_max"] = scipy.stats.spearmanr(prediction_ce_max, measurement_list)[0]
        result_dict["prediction_all_max"] = scipy.stats.spearmanr(prediction_all_max, measurement_list)[0]

        return result_dict

    def get_experimental_error(self, sess, conv_model, protein_name, step,
                               N_pred_iterations=10, batch_size=4, use_only_uppercase=True, use_family_embedding=False):

        prot_encoder = self.protein_names_to_one_hot_seqs_encoder[protein_name]
        prot_mask_encoder = self.protein_names_to_one_hot_seqs_encoder_mask[protein_name]
        prot_decoder_input = self.protein_names_to_one_hot_seqs_decoder_input[protein_name]
        prot_decoder_output = self.protein_names_to_one_hot_seqs_decoder_output[protein_name]
        prot_mask_decoder = self.protein_names_to_one_hot_seqs_decoder_mask[protein_name]

        measurement_list = self.protein_names_to_measurement_list[protein_name]

        # if only using the mutations from uppercase columns as in previous pubs
        if use_only_uppercase:
            idx_uppercase = self.protein_names_to_uppercase_idx[protein_name]

            prot_encoder = prot_encoder[idx_uppercase]
            prot_mask_encoder = prot_mask_encoder[idx_uppercase]
            prot_decoder_input = prot_decoder_input[idx_uppercase]
            prot_decoder_output = prot_decoder_output[idx_uppercase]
            prot_mask_decoder = prot_mask_decoder[idx_uppercase]
            measurement_list = measurement_list[idx_uppercase]

        prediction_matrix_ce = np.zeros((prot_encoder.shape[0], N_pred_iterations))
        prediction_matrix_all = np.zeros((prot_encoder.shape[0], N_pred_iterations))

        for i in range(N_pred_iterations):
            batch_order = np.arange(prot_encoder.shape[0])
            np.random.shuffle(batch_order)

            idx_family_one_hot = self.family_name_to_idx[protein_name]

            for j in range(0, prot_encoder.shape[0], batch_size):

                batch_index = batch_order[j:j + batch_size]
                batch_index_shape = prot_decoder_input[batch_index].shape[0]

                feed_dict = {conv_model.placeholders["sequences_start"]: prot_decoder_input[batch_index],
                             conv_model.placeholders["sequences_stop"]: prot_decoder_output[batch_index],
                             conv_model.placeholders["mask_decoder_1D"]: prot_mask_decoder[batch_index],
                             conv_model.placeholders["Neff"]: [
                                 float(len(self.family_name_to_sequence_decoder_output_list[protein_name]))],
                             conv_model.placeholders["step"]: [i]}

                if use_family_embedding:
                    family_one_hot = np.zeros(self.num_families)
                    family_one_hot[idx_family_one_hot] = 1.
                    feed_dict[conv_model.placeholders["family_embedding"]] = family_one_hot

                ce_loss, loss = sess.run([conv_model.tensors["cross_entropy_per_seq"],
                                          conv_model.tensors["loss_per_seq"]], feed_dict=feed_dict)

                for k, idx_batch in enumerate(batch_index.tolist()):
                    prediction_matrix_ce[idx_batch][i] = ce_loss[k]
                    prediction_matrix_all[idx_batch][i] = loss[k]

        prediction_ce = np.mean(prediction_matrix_ce, axis=1)
        prediction_all = np.mean(prediction_matrix_all, axis=1)

        return scipy.stats.spearmanr(prediction_ce, measurement_list)[0], \
               scipy.stats.spearmanr(prediction_all, measurement_list)[0]


class DataHelperDynamicBatches:
    def __init__(self, dataset='', alignment_file='', focus_seq_name='',
                 mutation_file='', calc_weights=True, working_dir='.', theta=0.2,
                 load_all_sequences=True, alphabet_type='protein', max_seq_len=-1,
                 autoregressive=True):

        np.random.seed(42)
        self.dataset = dataset
        self.alignment_file = alignment_file
        self.focus_seq_name = focus_seq_name
        self.mutation_file = mutation_file
        self.working_dir = working_dir
        self.calc_weights = calc_weights
        self.alphabet_type = alphabet_type
        self.max_seq_len = max_seq_len
        self.autoregressive = autoregressive

        # Alignment processing parameters
        self.theta = theta

        # If I am running tests with the model, I don't need all the
        #    sequences loaded
        self.load_all_sequences = load_all_sequences

        # Load necessary information for preloaded datasets
        if self.dataset != '':
            self.configure_datasets()

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

        # then generate the experimental data
        self.gen_alignment()

    def configure_datasets(self):

        if self.dataset == 'amp_2500':
            self.alignment_file = self.working_dir + '/datasets/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m'
            self.focus_seq_name = '>BLAT_ECOLX/24-286'
            self.mutation_file = self.working_dir + '/datasets/mutation_data/BLAT_ECOLX_Ranganathan2015_BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105_mutant_sequences.csv'
            self.full_sequence_file = self.working_dir + '/datasets/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105_full_seq.fa'
            self.protein_name = 'BLAT_ECOLX'

        elif self.dataset == 'methyltransferase':
            self.alignment_file = self.working_dir + '/datasets/MTH3_HAEAESTABILIZED_hmmerbit_plmc_n5_m30_f50_t0.2_r1-330_id100_b165.a2m'
            self.focus_seq_name = '>MTH3_HAEAESTABILIZED/1-330'
            self.mutation_file = self.working_dir + '/datasets/MTH3_HAEAESTABILIZED_Tawfik2015_MTH3_HAEAESTABILIZED_hmmerbit_plmc_n5_m30_f50_t0.2_r1-330_id100_b165_filtered_w17_seq.csv'
            self.protein_name = 'MTH3_HAEAESTABILIZED'

        elif self.dataset == 'PABP':
            self.alignment_file = self.working_dir + '/datasets/PABP_YEAST_hmmerbit_plmc_n5_m30_f50_t0.2_r115-210_id100_b48.a2m'
            self.focus_seq_name = '>PABP_YEAST/115-210'
            self.full_sequence_file = self.working_dir + '/datasets/hhfilter_full_length/PABP_YEAST_hmmerbit_plmc_n5_m30_f50_t0.2_r115-210_id100_b48_id80_upper_full_seq.fa'
            self.mutation_file = self.working_dir + '/datasets/mutation_data/unused/PABP_YEAST_Fields2013-singles_PABP_YEAST_hmmerbit_plmc_n5_m30_f50_t0.2_r115-210_id100_b48_mutant_sequences.csv'
            self.protein_name = 'PABP_YEAST'

        elif self.dataset == 'DLG':
            self.alignment_file = self.working_dir + '/datasets/DLG4_RAT_hmmerbit_plmc_n5_m30_f50_t0.2_r300-400_id100_b50.a2m'
            self.focus_seq_name = '>DLG4_RAT/300-400'
            self.mutation_file = self.working_dir + '/datasets/DLG4_RAT_Ranganathan2012_DLG4_RAT_hmmerbit_plmc_n5_m30_f50_t0.2_r300-400_id100_b50_seq.csv'
            self.protein_name = 'DLG4_RAT'

    def one_hot_3D(self, s):
        # One-hot encode as row vector
        x = np.zeros((len(s), len(self.alphabet)))
        for i, letter in enumerate(s):
            if letter in self.aa_dict:
                x[i, self.aa_dict[letter]] = 1
        return x

    def gen_alignment(self):

        # Make a dictionary that goes from aa to a number for one-hot
        self.aa_dict = {}
        for i, aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        # Do the inverse as well
        self.num_to_aa = {i: aa for aa, i in self.aa_dict.items()}

        ix = np.array([self.alphabet.find(s) for s in self.reorder_alphabet])

        # Read alignment
        self.seq_name_to_sequence = defaultdict(str)
        self.seq_names = []

        name = ''
        INPUT = open(self.alignment_file, 'r')
        for i, line in enumerate(INPUT):
            line = line.rstrip()
            if line.startswith('>'):
                line_list = list(line)
                line_list.pop(0)
                line = "".join(line_list)
                name = '>' + line.split('|')[-1]
                self.seq_names.append(name)
            else:
                self.seq_name_to_sequence[name] += line
        INPUT.close()

        name = ''
        self.seq_name_to_full_sequence = {}
        self.seq_names_full_sequence = []
        INPUT = open(self.full_sequence_file, 'r')
        for i, line in enumerate(INPUT):
            line = line.rstrip()
            if line.startswith('>'):
                name = line
                self.seq_names_full_sequence.append(name)
            else:
                self.seq_name_to_full_sequence[name] = line
        INPUT.close()

        self.seq_name_to_full_sequence[name] = line

        self.seq_name_to_sequence_aligned = {}

        if self.focus_seq_name == '':
            self.focus_seq_name = self.seq_names[0]

        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]

        # make sure my focus sequence is in my full alignment
        self.seq_name_to_full_sequence[self.focus_seq_name] = self.focus_seq.upper()

        self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper()]

        # print self.focus_cols

        alphabet_set = set(list(self.alphabet))
        seq_names_to_remove = []
        # print self.seq_names
        # Add the end character
        self.seq_names_aligned = []
        # print self.seq_name_to_sequence
        for i, name in enumerate(self.seq_names):
            # print name
            sequence = self.seq_name_to_sequence[name]

            ungapped_seq = sequence.replace('.', '')
            ungapped_seq = ungapped_seq.replace('-', '')
            ungapped_seq = ungapped_seq.upper()

            valid_sequence = True

            for letter in ungapped_seq:
                if letter not in alphabet_set:
                    valid_sequence = False

            if name in self.seq_name_to_full_sequence:
                unaligned_seq = self.seq_name_to_full_sequence[name]
                if unaligned_seq == '':
                    valid_sequence = False

                for letter in unaligned_seq:
                    if letter not in alphabet_set:
                        valid_sequence = False

            if valid_sequence and name in self.seq_name_to_full_sequence:
                self.seq_name_to_sequence_aligned[name] = "".join([sequence[ix] for ix in self.focus_cols])
                self.seq_names_aligned.append(name)

        self.seq_name_to_idx_full_sequence_encoder = {}
        self.seq_name_to_idx_full_sequence_input = {}
        self.seq_name_to_idx_full_sequence_output = {}
        self.seq_name_to_seq_len = {}

        print("Num valid sequences:", len(self.seq_names_aligned))

        self.max_seq_len = 0
        for name in self.seq_names_aligned:
            sequence = self.seq_name_to_full_sequence[name]

            seq_input = '*' + sequence
            seq_output = sequence + '*'

            seq_input_idx_list = []
            seq_output_idx_list = []
            seq_encoder_idx_list = []

            for i, letter in enumerate(seq_input):
                l = self.aa_dict[letter]
                seq_input_idx_list.append(l)

                if i != 0:
                    seq_encoder_idx_list.append(l)

            for letter in seq_output:
                l = self.aa_dict[letter]
                seq_output_idx_list.append(l)

            self.seq_name_to_idx_full_sequence_encoder[name] = seq_encoder_idx_list
            self.seq_name_to_idx_full_sequence_input[name] = seq_input_idx_list
            self.seq_name_to_idx_full_sequence_output[name] = seq_output_idx_list
            self.seq_name_to_seq_len[name] = len(sequence)
            if len(sequence) > self.max_seq_len:
                self.max_seq_len = len(sequence)

        # If we don't have a focus sequence, pick the one that
        #   we used to generate the alignment
        if self.focus_seq_name == '':
            self.focus_seq_name = self.seq_names_aligned[0]

        self.alphabet_size = len(self.alphabet)

        # Encode the sequences
        print('Encoding sequences')
        self.x_train_aligned = np.zeros(
            (len(list(self.seq_name_to_sequence_aligned.keys())), len(self.focus_cols), len(self.alphabet)))
        for i, seq_name in enumerate(self.seq_names_aligned):
            sequence = self.seq_name_to_sequence_aligned[seq_name]
            for j, letter in enumerate(sequence):
                if letter in self.aa_dict:
                    k = self.aa_dict[letter]
                    self.x_train_aligned[i, j, k] = 1.0

        # Fast sequence weights with Theano
        if self.calc_weights:
            print('Computing sequence weights')
            # Numpy version
            # self.weights = scale / np.sum(squareform(pdist(seq_index_array, metric='hamming')) < theta, axis=0)
            #
            # Theano weights
            X = T.tensor3("x")
            cutoff = T.scalar("theta")
            X_flat = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
            N_list, updates = theano.map(lambda x: 1.0 / T.sum(T.dot(X_flat, x) / T.dot(x, x) > 1 - cutoff), X_flat)
            weightfun = theano.function(inputs=[X, cutoff], outputs=[N_list], allow_input_downcast=True)
            #
            self.weights = weightfun(self.x_train_aligned, self.theta)[0]
        else:
            self.weights = np.ones(self.x_train_aligned.shape[0])

        self.Neff = np.sum(self.weights)
        self.seq_sample_probs = self.weights / self.Neff

        print('Neff =', str(self.Neff))
        print("Data Shape =", self.x_train_aligned.shape)

        # Then read in the mutation data so we can predict it later
        self.protein_mutation_names = []
        self.protein_names_to_uppercase_idx = {}
        self.protein_names_to_one_hot_seqs_encoder = {}
        self.protein_names_to_one_hot_seqs_decoder_input = {}
        self.protein_names_to_one_hot_seqs_decoder_output = {}
        self.protein_names_to_one_hot_seqs_encoder_mask = {}
        self.protein_names_to_one_hot_seqs_decoder_mask = {}
        self.protein_names_to_measurement_list = {}
        self.protein_names_to_mutation_list = {}

        INPUT = open(self.mutation_file, 'r')

        encoder_sequence_list = []
        decoder_input_sequence_list = []
        decoder_output_sequence_list = []
        mutation_list = []
        uppercase_list = []
        measurement_list = []

        for i, line in enumerate(INPUT):
            line = line.rstrip()
            line_list = line.split(';')
            if i != 0:
                mutation, is_upper, measurement, sequence = line_list
                if is_upper == 'True':
                    uppercase_list.append(i - 1)
                mutation_list.append(mutation)
                measurement_list.append(float(measurement))

                sequence = "".join([sequence[ix] for ix in self.focus_cols])

                encoder_sequence_list.append(sequence)
                decoder_input_sequence_list.append('*' + sequence)
                decoder_output_sequence_list.append(sequence + '*')
                mutant_seq_len = len(sequence)

        INPUT.close()

        self.protein_mutation_names.append(self.protein_name)
        self.protein_names_to_uppercase_idx[self.protein_name] = np.asarray(uppercase_list)
        self.protein_names_to_measurement_list[self.protein_name] = np.asarray(measurement_list)
        self.protein_names_to_mutation_list[self.protein_name] = mutation_list

        prot_encoder = np.zeros((len(measurement_list), 1, mutant_seq_len, len(self.alphabet)))
        prot_mask_encoder = np.zeros((len(measurement_list), 1, mutant_seq_len, 1))
        prot_decoder_output = np.zeros((len(measurement_list), 1, mutant_seq_len + 1, len(self.alphabet)))
        prot_decoder_input = np.zeros((len(measurement_list), 1, mutant_seq_len + 1, len(self.alphabet)))
        prot_mask_decoder = np.zeros((len(measurement_list), 1, mutant_seq_len + 1, 1))

        for j, sequence in enumerate(encoder_sequence_list):
            for k, letter in enumerate(sequence):
                if letter in self.aa_dict:
                    l = self.aa_dict[letter]
                    prot_encoder[j, 0, k, l] = 1.0
                    prot_mask_encoder[j, 0, k, 0] = 1.0

        for j, sequence in enumerate(decoder_output_sequence_list):
            for k, letter in enumerate(sequence):
                if letter in self.aa_dict:
                    l = self.aa_dict[letter]
                    prot_decoder_output[j, 0, k, l] = 1.0
                    prot_mask_decoder[j, 0, k, 0] = 1.0

        for j, sequence in enumerate(decoder_input_sequence_list):
            for k, letter in enumerate(sequence):
                if letter in self.aa_dict:
                    l = self.aa_dict[letter]
                    prot_decoder_input[j, 0, k, l] = 1.0

        self.protein_names_to_one_hot_seqs_encoder[self.protein_name] = prot_encoder
        self.protein_names_to_one_hot_seqs_encoder_mask[self.protein_name] = prot_mask_encoder
        self.protein_names_to_one_hot_seqs_decoder_input[self.protein_name] = prot_decoder_input
        self.protein_names_to_one_hot_seqs_decoder_output[self.protein_name] = prot_decoder_output
        self.protein_names_to_one_hot_seqs_decoder_mask[self.protein_name] = prot_mask_decoder

    def get_experimental_error(self, sess, conv_model, protein_name, step,
                               N_pred_iterations=10, batch_size=4, use_only_uppercase=True):

        prot_encoder = self.protein_names_to_one_hot_seqs_encoder[protein_name]
        prot_mask_encoder = self.protein_names_to_one_hot_seqs_encoder_mask[protein_name]
        prot_decoder_input = self.protein_names_to_one_hot_seqs_decoder_input[protein_name]
        prot_decoder_output = self.protein_names_to_one_hot_seqs_decoder_output[protein_name]
        prot_mask_decoder = self.protein_names_to_one_hot_seqs_decoder_mask[protein_name]

        measurement_list = self.protein_names_to_measurement_list[protein_name]

        # if only using the mutations from uppercase columns as in previous pubs
        if use_only_uppercase:
            idx_uppercase = self.protein_names_to_uppercase_idx[protein_name]

            prot_encoder = prot_encoder[idx_uppercase]
            prot_mask_encoder = prot_mask_encoder[idx_uppercase]
            prot_decoder_input = prot_decoder_input[idx_uppercase]
            prot_decoder_output = prot_decoder_output[idx_uppercase]
            prot_mask_decoder = prot_mask_decoder[idx_uppercase]
            measurement_list = measurement_list[idx_uppercase]

        prediction_matrix_ce = np.zeros((prot_encoder.shape[0], N_pred_iterations))
        prediction_matrix_all = np.zeros((prot_encoder.shape[0], N_pred_iterations))

        for i in range(N_pred_iterations):
            batch_order = np.arange(prot_encoder.shape[0])
            np.random.shuffle(batch_order)

            for j in range(0, prot_encoder.shape[0], batch_size):

                batch_index = batch_order[j:j + batch_size]
                batch_index_shape = prot_decoder_input[batch_index].shape[0]

                if self.autoregressive:
                    feed_dict = {conv_model.placeholders["sequences_start"]: prot_decoder_input[batch_index],
                                 conv_model.placeholders["sequences_stop"]: prot_decoder_output[batch_index],
                                 conv_model.placeholders["mask_decoder_1D"]: prot_mask_decoder[batch_index],
                                 conv_model.placeholders["Neff"]: [self.Neff], conv_model.placeholders["step"]: [i]}

                else:
                    feed_dict = {
                        conv_model.placeholders["sequences"]: prot_encoder[batch_index],
                        conv_model.placeholders["mask_1D"]: prot_mask_encoder[batch_index],
                        conv_model.placeholders["Neff"]: [self.Neff], conv_model.placeholders["step"]: [i]}

                ce_loss, loss = sess.run([conv_model.tensors["cross_entropy_per_seq"],
                                          conv_model.tensors["loss_per_seq"]], feed_dict=feed_dict)

                for k, idx_batch in enumerate(batch_index.tolist()):
                    prediction_matrix_ce[idx_batch][i] = ce_loss[k]
                    prediction_matrix_all[idx_batch][i] = loss[k]

        prediction_ce = np.mean(prediction_matrix_ce, axis=1)
        prediction_all = np.mean(prediction_matrix_all, axis=1)

        return scipy.stats.spearmanr(prediction_ce, measurement_list)[0], \
               scipy.stats.spearmanr(prediction_all, measurement_list)[0]

    def generate_minibatch_idx(self, minibatch_size):

        sequence_idx = np.random.choice(np.arange(self.x_train_encoder.shape[0]), minibatch_size, p=self.seq_sample_probs).tolist()
        # sequence_idx = np.random.choice(self.x_train_encoder.shape[0],minibatch_size)

        return sequence_idx

    def generate_minibatch_data(self, minibatch_size):

        sequence_idx = np.random.choice(np.arange(len(self.seq_names_aligned)), minibatch_size, p=self.seq_sample_probs).tolist()

        minibatch_seq_names = [self.seq_names_aligned[idx] for idx in sequence_idx]
        minibatch_seq_lengths = [self.seq_name_to_seq_len[name] for name in minibatch_seq_names]

        minibatch_max_seq_len = np.amax(minibatch_seq_lengths)

        prot_encoder = np.zeros((minibatch_size, 1, minibatch_max_seq_len, len(self.alphabet)))
        # prot_encoder = np.zeros((minibatch_size,1,self.max_seq_len,len(self.alphabet)))
        prot_mask_encoder = np.zeros((minibatch_size, 1, minibatch_max_seq_len, 1))
        # prot_mask_encoder = np.zeros((minibatch_size,1,self.max_seq_len,1))

        prot_decoder_output = np.zeros((minibatch_size, 1, minibatch_max_seq_len + 1, len(self.alphabet)))
        prot_decoder_input = np.zeros((minibatch_size, 1, minibatch_max_seq_len + 1, len(self.alphabet)))
        prot_mask_decoder = np.zeros((minibatch_size, 1, minibatch_max_seq_len + 1, 1))

        for i, name in enumerate(minibatch_seq_names):

            idx_encoder = self.seq_name_to_idx_full_sequence_encoder[name]
            idx_decoder_in = self.seq_name_to_idx_full_sequence_input[name]
            idx_decoder_out = self.seq_name_to_idx_full_sequence_output[name]

            for j in np.arange(minibatch_seq_lengths[i]):
                prot_encoder[i, 0, j, idx_encoder[j]] = 1
                prot_mask_encoder[i, 0, j, 0] = 1
                prot_decoder_input[i, 0, j, idx_decoder_in[j]] = 1
                prot_decoder_output[i, 0, j, idx_decoder_out[j]] = 1
                prot_mask_decoder[i, 0, j, 0] = 1

            # put in the final letter, as the decoder values are one AA longer
            # print i,name
            prot_decoder_input[i, 0, j + 1, idx_decoder_in[j + 1]] = 1
            prot_decoder_output[i, 0, j + 1, idx_decoder_out[j + 1]] = 1
            prot_mask_decoder[i, 0, j + 1, 0] = 1

        return prot_encoder, prot_mask_encoder, prot_decoder_input, \
               prot_decoder_output, prot_mask_decoder

    def gen_target_aa(self, x_input, x_input_lengths, dims):

        target_aa = []
        query_seq = np.copy(x_input)
        for i, input_len in enumerate(x_input_lengths):
            idx_rand = np.random.randint(input_len)
            target_aa.append(query_seq[i, 0, idx_rand].flatten().tolist())
            query_seq[i, 0, idx_rand] *= 0.
            query_seq[i, 0, idx_rand, dims["alphabet"] - 1] = 1.

        return query_seq, np.asarray(target_aa)

    def get_elbo_samples(self, model, N_pred_iterations=100, minibatch_size=2000):

        print(self.one_hot_mut_array_with_wt.shape)
        self.prediction_matrix = np.zeros((self.one_hot_mut_array_with_wt.shape[0], N_pred_iterations))

        batch_order = np.arange(self.one_hot_mut_array_with_wt.shape[0])

        for i in range(N_pred_iterations):
            np.random.shuffle(batch_order)

            for j in range(0, self.one_hot_mut_array_with_wt.shape[0], minibatch_size):

                batch_index = batch_order[j:j + minibatch_size]
                batch_preds, _, _ = model.all_likelihood_components(self.one_hot_mut_array_with_wt[batch_index])

                for k, idx_batch in enumerate(batch_index.tolist()):
                    self.prediction_matrix[idx_batch][i] = batch_preds[k]


class DataHelperAutoregressive:
    def __init__(self, dataset='', alignment_file='', focus_seq_name='',
                 mutation_file='', calc_weights=True, working_dir='.', theta=0.2,
                 load_all_sequences=True, alphabet_type='protein', max_seq_len=-1):

        np.random.seed(42)
        self.dataset = dataset
        self.alignment_file = alignment_file
        self.focus_seq_name = focus_seq_name
        self.mutation_file = mutation_file
        self.working_dir = working_dir
        self.calc_weights = calc_weights
        self.alphabet_type = alphabet_type
        self.max_seq_len = max_seq_len
        # Alignment processing parameters
        self.theta = theta

        # If I am running tests with the model, I don't need all the
        #    sequences loaded
        self.load_all_sequences = load_all_sequences

        # Load necessary information for preloaded datasets
        if self.dataset != '':
            self.configure_datasets()

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

        # then generate the experimental data
        self.gen_basic_alignment()

        if self.load_all_sequences:
            self.gen_full_alignment()

    def configure_datasets(self):

        if self.dataset == 'amp_2500':
            self.alignment_file = self.working_dir + '/datasets/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m'
            self.focus_seq_name = '>BLAT_ECOLX/24-286'
            self.mutation_file = self.working_dir + '/datasets/mutation_data/BLAT_ECOLX_Ranganathan2015_BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105_mutant_sequences.csv'
            self.pdb_id = '1axb'
            self.chain_color_list = ['A']
            self.chain_show_list = ['A']
            self.show_organic = True
            self.protein_name = 'BLAT_ECOLX'

        elif self.dataset == 'methyltransferase':
            self.alignment_file = self.working_dir + '/datasets/MTH3_HAEAESTABILIZED_hmmerbit_plmc_n5_m30_f50_t0.2_r1-330_id100_b165.a2m'
            self.focus_seq_name = '>MTH3_HAEAESTABILIZED/1-330'
            self.mutation_file = self.working_dir + '/datasets/MTH3_HAEAESTABILIZED_Tawfik2015_MTH3_HAEAESTABILIZED_hmmerbit_plmc_n5_m30_f50_t0.2_r1-330_id100_b165_filtered_w17_seq.csv'
            self.pdb_id = '1dct'
            self.chain_color_list = ['A']
            self.chain_show_list = ['A', 'F', 'M']
            self.protein_name = 'MTH3_HAEAESTABILIZED'

        elif self.dataset == 'PABP':
            self.alignment_file = self.working_dir + '/datasets/PABP_YEAST_hmmerbit_plmc_n5_m30_f50_t0.2_r115-210_id100_b48.a2m'
            self.focus_seq_name = '>PABP_YEAST/115-210'
            self.mutation_file = self.working_dir + '/datasets/PABP_YEAST_Fields2013-all_PABP_YEAST_hmmerbit_plmc_n5_m30_f50_t0.2_r115-210_id100_b48_seq.csv'
            self.pdb_id = '1cvj'
            self.chain_color_list = ['A']
            self.chain_show_list = ['A', 'M']
            self.protein_name = 'PABP_YEAST'

        elif self.dataset == 'DLG':
            self.alignment_file = self.working_dir + '/datasets/DLG4_RAT_hmmerbit_plmc_n5_m30_f50_t0.2_r300-400_id100_b50.a2m'
            self.focus_seq_name = '>DLG4_RAT/300-400'
            self.mutation_file = self.working_dir + '/datasets/DLG4_RAT_Ranganathan2012_DLG4_RAT_hmmerbit_plmc_n5_m30_f50_t0.2_r300-400_id100_b50_seq.csv'
            self.pdb_id = '1be9'
            self.chain_color_list = ['A']
            self.chain_show_list = ['A']
            self.protein_name = 'DLG4_RAT'

        elif self.dataset == 'DLG_small':
            self.alignment_file = self.working_dir + '/datasets/DLG4_RAT_small.a2m'
            self.focus_seq_name = '>DLG4_RAT/300-400'
            self.mutation_file = self.working_dir + '/datasets/DLG4_RAT_Ranganathan2012_DLG4_RAT_hmmerbit_plmc_n5_m30_f50_t0.2_r300-400_id100_b50_seq.csv'
            self.pdb_id = '1be9'
            self.chain_color_list = ['A']
            self.chain_show_list = ['A']

    def one_hot_3D(self, s):
        # One-hot encode as row vector
        x = np.zeros((len(s), len(self.alphabet)))
        for i, letter in enumerate(s):
            if letter in self.aa_dict:
                x[i, self.aa_dict[letter]] = 1
        return x

    def gen_basic_alignment(self):

        # Make a dictionary that goes from aa to a number for one-hot
        self.aa_dict = {}
        for i, aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        # Do the inverse as well
        self.num_to_aa = {i: aa for aa, i in self.aa_dict.items()}

        ix = np.array([self.alphabet.find(s) for s in self.reorder_alphabet])

        # Read alignment
        self.seq_name_to_sequence = defaultdict(str)
        self.seq_names = []

        name = ''
        INPUT = open(self.alignment_file, 'r')
        for i, line in enumerate(INPUT):
            line = line.rstrip()
            if line.startswith('>'):
                name = line
                self.seq_names.append(name)
            else:
                self.seq_name_to_sequence[name] += line
        INPUT.close()

        self.seq_name_to_sequence_aligned = {}
        self.seq_name_to_sequence_encoder = {}
        self.seq_name_to_sequence_decoder_input = {}
        self.seq_name_to_sequence_decoder_output = {}

        if self.focus_seq_name == '':
            self.focus_seq_name = self.seq_names[0]

        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
        self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper()]

        alphabet_set = set(list(self.alphabet))
        seq_names_to_remove = []

        # Add the end character
        for name, sequence in self.seq_name_to_sequence.items():

            ungapped_seq = sequence.replace('.', '')
            ungapped_seq = ungapped_seq.replace('-', '')
            ungapped_seq = ungapped_seq.upper()

            valid_sequence = True

            for letter in ungapped_seq:
                if letter not in alphabet_set:
                    valid_sequence = False
                    seq_names_to_remove.append(name)

            if valid_sequence:
                self.seq_name_to_sequence_aligned[name] = "".join([sequence[ix] for ix in self.focus_cols])
                self.seq_name_to_sequence_encoder[name] = ungapped_seq
                self.seq_name_to_sequence_decoder_input[name] = '*' + ungapped_seq
                self.seq_name_to_sequence_decoder_output[name] = ungapped_seq + '*'

        # Remove sequences that have bad characters
        seq_names_to_remove = list(set(seq_names_to_remove))
        for seq_name in seq_names_to_remove:
            del self.seq_name_to_sequence[seq_name]

        # If we don't have a focus sequence, pick the one that
        #   we used to generate the alignment
        if self.focus_seq_name == '':
            self.focus_seq_name = self.seq_names[0]

        # Select focus columns

        if self.max_seq_len == -1:
            self.seq_len = len(self.seq_name_to_sequence[self.focus_seq_name])
        else:
            self.seq_len = self.max_seq_len

        self.alphabet_size = len(self.alphabet)

    def gen_full_alignment(self):

        # Encode the sequences
        print('Encoding sequences')
        self.x_train_aligned = np.zeros(
            (len(list(self.seq_name_to_sequence.keys())), len(self.focus_cols), len(self.alphabet)))
        self.x_train_name_list = []
        for i, seq_name in enumerate(self.seq_name_to_sequence.keys()):
            sequence = self.seq_name_to_sequence_aligned[seq_name]
            self.x_train_name_list.append(seq_name)
            for j, letter in enumerate(sequence):
                if letter in self.aa_dict:
                    k = self.aa_dict[letter]
                    self.x_train_aligned[i, j, k] = 1.0

        # Make a non-gapped version
        self.x_train_encoder = np.zeros(
            (len(list(self.seq_name_to_sequence.keys())), 1, self.seq_len, len(self.alphabet)))
        self.x_train_lengths = np.zeros((len(list(self.seq_name_to_sequence.keys()))))
        self.x_train_mask_encoder = np.zeros((len(list(self.seq_name_to_sequence.keys())), 1, self.seq_len, 1))

        for i, seq_name in enumerate(self.x_train_name_list):
            sequence = self.seq_name_to_sequence_encoder[seq_name]
            for j, letter in enumerate(sequence):
                if j < self.seq_len:
                    if letter in self.aa_dict:
                        k = self.aa_dict[letter]
                        self.x_train_encoder[i, 0, j, k] = 1.0
                        self.x_train_mask_encoder[i, 0, j, 0] = 1.0
                self.x_train_lengths[i] = len(sequence)

        self.x_train_decoder_output = np.zeros(
            (len(list(self.seq_name_to_sequence.keys())), 1, self.seq_len + 1, len(self.alphabet)))
        self.x_train_decoder_input = np.zeros(
            (len(list(self.seq_name_to_sequence.keys())), 1, self.seq_len + 1, len(self.alphabet)))
        self.x_train_mask_decoder = np.zeros((len(list(self.seq_name_to_sequence.keys())), 1, self.seq_len + 1, 1))

        for i, seq_name in enumerate(self.x_train_name_list):
            sequence = self.seq_name_to_sequence_decoder_output[seq_name]
            for j, letter in enumerate(sequence):
                if j < self.seq_len:
                    if letter in self.aa_dict:
                        k = self.aa_dict[letter]
                        self.x_train_decoder_output[i, 0, j, k] = 1.0
                        self.x_train_mask_decoder[i, 0, j, 0] = 1.0

        for i, seq_name in enumerate(self.x_train_name_list):
            sequence = self.seq_name_to_sequence_decoder_input[seq_name]
            for j, letter in enumerate(sequence):
                if j < self.seq_len:
                    if letter in self.aa_dict:
                        k = self.aa_dict[letter]
                        self.x_train_decoder_input[i, 0, j, k] = 1.0

        # Fast sequence weights with Theano
        if self.calc_weights:
            print('Computing sequence weights')
            # Numpy version
            # self.weights = scale / np.sum(squareform(pdist(seq_index_array, metric='hamming')) < theta, axis=0)
            #
            # Theano weights
            X = T.tensor3("x")
            cutoff = T.scalar("theta")
            X_flat = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
            N_list, updates = theano.map(lambda x: 1.0 / T.sum(T.dot(X_flat, x) / T.dot(x, x) > 1 - cutoff), X_flat)
            weightfun = theano.function(inputs=[X, cutoff], outputs=[N_list], allow_input_downcast=True)
            #
            self.weights = weightfun(self.x_train_aligned, self.theta)[0]
        else:
            self.weights = np.ones(self.x_train_aligned.shape[0])

        self.Neff = np.sum(self.weights)
        self.seq_sample_probs = self.weights / self.Neff

        print('Neff =', str(self.Neff))
        print("Data Shape =", self.x_train_aligned.shape)

        # Then read in the mutation data so we can predict it later
        self.protein_mutation_names = []
        self.protein_names_to_uppercase_idx = {}
        self.protein_names_to_one_hot_seqs_encoder = {}
        self.protein_names_to_one_hot_seqs_decoder_input = {}
        self.protein_names_to_one_hot_seqs_decoder_output = {}
        self.protein_names_to_one_hot_seqs_encoder_mask = {}
        self.protein_names_to_one_hot_seqs_decoder_mask = {}
        self.protein_names_to_measurement_list = {}
        self.protein_names_to_mutation_list = {}

        INPUT = open(self.mutation_file, 'r')

        encoder_sequence_list = []
        decoder_input_sequence_list = []
        decoder_output_sequence_list = []
        mutation_list = []
        uppercase_list = []
        measurement_list = []

        for i, line in enumerate(INPUT):
            line = line.rstrip()
            line_list = line.split(';')
            if i != 0:
                mutation, is_upper, measurement, sequence = line_list
                if is_upper == 'True':
                    uppercase_list.append(i - 1)
                mutation_list.append(mutation)
                measurement_list.append(float(measurement))

                sequence = "".join([sequence[ix] for ix in self.focus_cols])

                encoder_sequence_list.append(sequence)
                decoder_input_sequence_list.append('*' + sequence)
                decoder_output_sequence_list.append(sequence + '*')

        INPUT.close()

        self.protein_mutation_names.append(self.protein_name)
        self.protein_names_to_uppercase_idx[self.protein_name] = np.asarray(uppercase_list)
        self.protein_names_to_measurement_list[self.protein_name] = np.asarray(measurement_list)
        self.protein_names_to_mutation_list[self.protein_name] = mutation_list

        prot_encoder = np.zeros((len(measurement_list), 1, self.seq_len, len(self.alphabet)))
        prot_mask_encoder = np.zeros((len(measurement_list), 1, self.seq_len, 1))
        prot_decoder_output = np.zeros((len(measurement_list), 1, self.seq_len + 1, len(self.alphabet)))
        prot_decoder_input = np.zeros((len(measurement_list), 1, self.seq_len + 1, len(self.alphabet)))
        prot_mask_decoder = np.zeros((len(measurement_list), 1, self.seq_len + 1, 1))

        for j, sequence in enumerate(encoder_sequence_list):
            for k, letter in enumerate(sequence):
                if letter in self.aa_dict:
                    l = self.aa_dict[letter]
                    prot_encoder[j, 0, k, l] = 1.0
                    prot_mask_encoder[j, 0, k, 0] = 1.0

        for j, sequence in enumerate(decoder_output_sequence_list):
            for k, letter in enumerate(sequence):
                if letter in self.aa_dict:
                    l = self.aa_dict[letter]
                    prot_decoder_output[j, 0, k, l] = 1.0
                    prot_mask_decoder[j, 0, k, 0] = 1.0

        for j, sequence in enumerate(decoder_input_sequence_list):
            for k, letter in enumerate(sequence):
                if letter in self.aa_dict:
                    l = self.aa_dict[letter]
                    prot_decoder_input[j, 0, k, l] = 1.0

        self.protein_names_to_one_hot_seqs_encoder[self.protein_name] = prot_encoder
        self.protein_names_to_one_hot_seqs_encoder_mask[self.protein_name] = prot_mask_encoder
        self.protein_names_to_one_hot_seqs_decoder_input[self.protein_name] = prot_decoder_input
        self.protein_names_to_one_hot_seqs_decoder_output[self.protein_name] = prot_decoder_output
        self.protein_names_to_one_hot_seqs_decoder_mask[self.protein_name] = prot_mask_decoder

    def get_experimental_error(self, sess, conv_model, protein_name, step,
                               N_pred_iterations=10, batch_size=4, use_only_uppercase=True):

        prot_encoder = self.protein_names_to_one_hot_seqs_encoder[protein_name]
        prot_mask_encoder = self.protein_names_to_one_hot_seqs_encoder_mask[protein_name]
        prot_decoder_input = self.protein_names_to_one_hot_seqs_decoder_input[protein_name]
        prot_decoder_output = self.protein_names_to_one_hot_seqs_decoder_output[protein_name]
        prot_mask_decoder = self.protein_names_to_one_hot_seqs_decoder_mask[protein_name]

        measurement_list = self.protein_names_to_measurement_list[protein_name]

        # if only using the mutations from uppercase columns as in previous pubs
        if use_only_uppercase:
            idx_uppercase = self.protein_names_to_uppercase_idx[protein_name]

            prot_encoder = prot_encoder[idx_uppercase]
            prot_mask_encoder = prot_mask_encoder[idx_uppercase]
            prot_decoder_input = prot_decoder_input[idx_uppercase]
            prot_decoder_output = prot_decoder_output[idx_uppercase]
            prot_mask_decoder = prot_mask_decoder[idx_uppercase]
            measurement_list = measurement_list[idx_uppercase]

        prediction_matrix_ce = np.zeros((prot_encoder.shape[0], N_pred_iterations))
        prediction_matrix_all = np.zeros((prot_encoder.shape[0], N_pred_iterations))

        for i in range(N_pred_iterations):
            batch_order = np.arange(prot_encoder.shape[0])
            np.random.shuffle(batch_order)

            for j in range(0, prot_encoder.shape[0], batch_size):

                batch_index = batch_order[j:j + batch_size]
                batch_index_shape = prot_decoder_input[batch_index].shape[0]

                feed_dict = {conv_model.placeholders["sequences_start"]: prot_decoder_input[batch_index],
                             conv_model.placeholders["sequences_stop"]: prot_decoder_output[batch_index],
                             conv_model.placeholders["mask_decoder_1D"]: prot_mask_decoder[batch_index],
                             conv_model.placeholders["Neff"]: [self.Neff], conv_model.placeholders["step"]: [i]}

                # Add the encoder values if using a vae
                if conv_model.model_type == 'vae':
                    feed_dict[conv_model.placeholders["sequences"]] = prot_encoder[batch_index]
                    feed_dict[conv_model.placeholders["mask_encoder_1D"]] = prot_mask_encoder[batch_index]

                ce_loss, loss = sess.run([conv_model.tensors["cross_entropy_per_seq"],
                                          conv_model.tensors["loss_per_seq"]], feed_dict=feed_dict)

                for k, idx_batch in enumerate(batch_index.tolist()):
                    prediction_matrix_ce[idx_batch][i] = ce_loss[k]
                    prediction_matrix_all[idx_batch][i] = loss[k]

        prediction_ce = np.mean(prediction_matrix_ce, axis=1)
        prediction_all = np.mean(prediction_matrix_all, axis=1)

        return scipy.stats.spearmanr(prediction_ce, measurement_list)[0], \
               scipy.stats.spearmanr(prediction_all, measurement_list)[0]

    def generate_minibatch_idx(self, minibatch_size):

        sequence_idx = np.random.choice(np.arange(self.x_train_encoder.shape[0]), minibatch_size, p=self.seq_sample_probs).tolist()
        # sequence_idx = np.random.choice(self.x_train_encoder.shape[0],minibatch_size)

        return sequence_idx

    def gen_target_aa(self, x_input, x_input_lengths, dims):

        target_aa = []
        query_seq = np.copy(x_input)
        for i, input_len in enumerate(x_input_lengths):
            idx_rand = np.random.randint(input_len)
            target_aa.append(query_seq[i, 0, idx_rand].flatten().tolist())
            query_seq[i, 0, idx_rand] *= 0.
            query_seq[i, 0, idx_rand, dims["alphabet"] - 1] = 1.

        return query_seq, np.asarray(target_aa)

    def get_elbo_samples(self, model, N_pred_iterations=100, minibatch_size=2000):

        print(self.one_hot_mut_array_with_wt.shape)
        self.prediction_matrix = np.zeros((self.one_hot_mut_array_with_wt.shape[0], N_pred_iterations))

        batch_order = np.arange(self.one_hot_mut_array_with_wt.shape[0])

        for i in range(N_pred_iterations):
            np.random.shuffle(batch_order)

            for j in range(0, self.one_hot_mut_array_with_wt.shape[0], minibatch_size):

                batch_index = batch_order[j:j + minibatch_size]
                batch_preds, _, _ = model.all_likelihood_components(self.one_hot_mut_array_with_wt[batch_index])

                for k, idx_batch in enumerate(batch_index.tolist()):
                    self.prediction_matrix[idx_batch][i] = batch_preds[k]


class DataHelperMultifam:
    def __init__(self, dataset='', alignment_file='', focus_seq_name='',
                 mutation_file='', calc_weights=True, working_dir='.', theta=0.2,
                 load_all_sequences=True, alphabet_type='protein', max_seq_len=-1,
                 longest_entry=125):

        np.random.seed(42)
        self.dataset = dataset
        self.alignment_file = alignment_file
        self.focus_seq_name = focus_seq_name
        self.mutation_file = mutation_file
        self.working_dir = working_dir
        self.calc_weights = calc_weights
        self.alphabet_type = alphabet_type
        self.max_seq_len = max_seq_len
        # Alignment processing parameters
        self.theta = theta

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

        # then generate the experimental data
        self.gen_alignment_mut_data()

    def configure_datasets(self):

        if self.dataset == 'amp_2500':
            self.alignment_file = self.working_dir + '/datasets/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m'
            self.focus_seq_name = '>BLAT_ECOLX/24-286'
            self.mutation_file = self.working_dir + '/datasets/BLAT_ECOLX_Ranganathan2015_BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105_amp_2500_seq.csv'
            self.pdb_id = '1axb'
            self.chain_color_list = ['A']
            self.chain_show_list = ['A']
            self.show_organic = True

        elif self.dataset == 'methyltransferase':
            self.alignment_file = self.working_dir + '/datasets/MTH3_HAEAESTABILIZED_hmmerbit_plmc_n5_m30_f50_t0.2_r1-330_id100_b165.a2m'
            self.focus_seq_name = '>MTH3_HAEAESTABILIZED/1-330'
            self.mutation_file = self.working_dir + '/datasets/MTH3_HAEAESTABILIZED_Tawfik2015_MTH3_HAEAESTABILIZED_hmmerbit_plmc_n5_m30_f50_t0.2_r1-330_id100_b165_filtered_w17_seq.csv'
            self.pdb_id = '1dct'
            self.chain_color_list = ['A']
            self.chain_show_list = ['A', 'F', 'M']

        elif self.dataset == 'PABP':
            self.alignment_file = self.working_dir + '/datasets/PABP_YEAST_hmmerbit_plmc_n5_m30_f50_t0.2_r115-210_id100_b48.a2m'
            self.focus_seq_name = '>PABP_YEAST/115-210'
            self.mutation_file = self.working_dir + '/datasets/PABP_YEAST_Fields2013-all_PABP_YEAST_hmmerbit_plmc_n5_m30_f50_t0.2_r115-210_id100_b48_seq.csv'
            self.pdb_id = '1cvj'
            self.chain_color_list = ['A']
            self.chain_show_list = ['A', 'M']

        elif self.dataset == 'DLG':
            self.alignment_file = self.working_dir + '/datasets/DLG4_RAT_hmmerbit_plmc_n5_m30_f50_t0.2_r300-400_id100_b50.a2m'
            self.focus_seq_name = '>DLG4_RAT/300-400'
            self.mutation_file = self.working_dir + '/datasets/DLG4_RAT_Ranganathan2012_DLG4_RAT_hmmerbit_plmc_n5_m30_f50_t0.2_r300-400_id100_b50_seq.csv'
            self.pdb_id = '1be9'
            self.chain_color_list = ['A']
            self.chain_show_list = ['A']

        elif self.dataset == 'DLG_small':
            self.alignment_file = self.working_dir + '/datasets/DLG4_RAT_small.a2m'
            self.focus_seq_name = '>DLG4_RAT/300-400'
            self.mutation_file = self.working_dir + '/datasets/DLG4_RAT_Ranganathan2012_DLG4_RAT_hmmerbit_plmc_n5_m30_f50_t0.2_r300-400_id100_b50_seq.csv'
            self.pdb_id = '1be9'
            self.chain_color_list = ['A']
            self.chain_show_list = ['A']

    def one_hot_3D(self, s):
        # One-hot encode as row vector
        x = np.zeros((len(s), len(self.alphabet)))
        for i, letter in enumerate(s):
            if letter in self.aa_dict:
                x[i, self.aa_dict[letter]] = 1
        return x

    def gen_alignment_mut_data(self):

        # Make a dictionary that goes from aa to a number for one-hot
        self.aa_dict = {}
        for i, aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        # Do the inverse as well
        self.num_to_aa = {i: aa for aa, i in self.aa_dict.items()}

        ix = np.array([self.alphabet.find(s) for s in self.reorder_alphabet])

        self.family_name_to_sequence_encoder_list = {}
        self.family_name_to_sequence_decoder_input_list = {}
        self.family_name_to_sequence_decoder_output_list = {}
        self.family_name_list = []
        self.family_idx_list = []

        # for now, we will make all the sequences have the same length of
        #   encoded matrices, though this is wasteful

        max_seq_len = 0
        max_family_size = 0
        for filename in glob.glob('datasets/hhfilter_full_length/*.fa'):
            INPUT = open(filename, 'r')

            encoder_sequence_list = []
            decoder_input_sequence_list = []
            decoder_output_sequence_list = []

            family_name_list = filename.split('/')[-1].split('_')
            family_name = family_name_list[0] + '_' + family_name_list[1]

            print(family_name)

            family_size = 0
            ind_family_idx_list = []

            for line in INPUT:
                line = line.rstrip()
                if line[0] != '>':
                    encoder_sequence_list.append(line)
                    decoder_input_sequence_list.append('*' + line)
                    decoder_output_sequence_list.append(line + '*')
                    ind_family_idx_list.append(family_size)
                    family_size += 1
                    if len(line) > max_seq_len:
                        max_seq_len = len(line)
            INPUT.close()

            if family_size > max_family_size:
                max_family_size = family_size

            self.family_name_to_sequence_encoder_list[family_name] = encoder_sequence_list
            self.family_name_to_sequence_decoder_input_list[family_name] = decoder_input_sequence_list
            self.family_name_to_sequence_decoder_output_list[family_name] = decoder_output_sequence_list
            self.family_name_list.append(family_name)
            self.family_idx_list.append(ind_family_idx_list)

        self.seq_len = max_seq_len
        self.num_families = len(self.family_name_list)
        self.max_family_size = max_family_size

        print('Encoding sequences')
        self.x_train_encoder = np.zeros((self.num_families, self.max_family_size, 1, self.seq_len, len(self.alphabet)))
        self.x_train_mask_encoder = np.zeros((self.num_families, self.max_family_size, 1, self.seq_len, 1))
        self.x_train_decoder_output = np.zeros(
            (self.num_families, self.max_family_size, 1, self.seq_len + 1, len(self.alphabet)))
        self.x_train_decoder_input = np.zeros(
            (self.num_families, self.max_family_size, 1, self.seq_len + 1, len(self.alphabet)))
        self.x_train_mask_decoder = np.zeros((self.num_families, self.max_family_size, 1, self.seq_len + 1, 1))
        self.family_name_to_idx = {}

        print("All training data size:", self.x_train_encoder.shape)
        for i, family_name in enumerate(self.family_name_list):
            self.family_name_to_idx[family_name] = i
            sequence_list = self.family_name_to_sequence_encoder_list[family_name]
            for j, sequence in enumerate(sequence_list):
                for k, letter in enumerate(sequence):
                    if letter in self.aa_dict:
                        l = self.aa_dict[letter]
                        self.x_train_encoder[i, j, 0, k, l] = 1.0
                        self.x_train_mask_encoder[i, j, 0, k, 0] = 1.0

        for i, family_name in enumerate(self.family_name_list):
            sequence_list = self.family_name_to_sequence_decoder_output_list[family_name]
            for j, sequence in enumerate(sequence_list):
                for k, letter in enumerate(sequence):
                    if letter in self.aa_dict:
                        l = self.aa_dict[letter]
                        self.x_train_decoder_output[i, j, 0, k, l] = 1.0
                        self.x_train_mask_decoder[i, j, 0, k, 0] = 1.0

        for i, family_name in enumerate(self.family_name_list):
            sequence_list = self.family_name_to_sequence_decoder_input_list[family_name]
            for j, sequence in enumerate(sequence_list):
                for k, letter in enumerate(sequence):
                    if letter in self.aa_dict:
                        l = self.aa_dict[letter]
                        self.x_train_decoder_input[i, j, 0, k, l] = 1.0

        # Then read in the mutation data so we can predict it later
        self.protein_mutation_names = []
        self.protein_names_to_uppercase_idx = {}
        self.protein_names_to_one_hot_seqs_encoder = {}
        self.protein_names_to_one_hot_seqs_decoder_input = {}
        self.protein_names_to_one_hot_seqs_decoder_output = {}
        self.protein_names_to_one_hot_seqs_encoder_mask = {}
        self.protein_names_to_one_hot_seqs_decoder_mask = {}
        self.protein_names_to_measurement_list = {}
        self.protein_names_to_mutation_list = {}

        for filename in glob.glob('./datasets/mutation_data/*.csv'):

            INPUT = open(filename, 'r')

            encoder_sequence_list = []
            decoder_input_sequence_list = []
            decoder_output_sequence_list = []
            mutation_list = []
            uppercase_list = []
            measurement_list = []

            family_name_list = filename.split('/')[-1].split('_')
            family_name = family_name_list[0] + '_' + family_name_list[1]

            for i, line in enumerate(INPUT):
                line = line.rstrip()
                line_list = line.split(';')
                if i != 0:
                    mutation, is_upper, measurement, sequence = line_list
                    if is_upper == 'True':
                        uppercase_list.append(i - 1)
                    mutation_list.append(mutation)
                    measurement_list.append(float(measurement))

                    encoder_sequence_list.append(sequence)
                    decoder_input_sequence_list.append('*' + sequence)
                    decoder_output_sequence_list.append(sequence + '*')

            INPUT.close()

            self.protein_mutation_names.append(family_name)
            self.protein_names_to_uppercase_idx[family_name] = np.asarray(uppercase_list)
            self.protein_names_to_measurement_list[family_name] = np.asarray(measurement_list)
            self.protein_names_to_mutation_list[family_name] = mutation_list

            prot_encoder = np.zeros((len(measurement_list), 1, self.seq_len, len(self.alphabet)))
            prot_mask_encoder = np.zeros((len(measurement_list), 1, self.seq_len, 1))
            prot_decoder_output = np.zeros((len(measurement_list), 1, self.seq_len + 1, len(self.alphabet)))
            prot_decoder_input = np.zeros((len(measurement_list), 1, self.seq_len + 1, len(self.alphabet)))
            prot_mask_decoder = np.zeros((len(measurement_list), 1, self.seq_len + 1, 1))

            for j, sequence in enumerate(encoder_sequence_list):
                for k, letter in enumerate(sequence):
                    if letter in self.aa_dict:
                        l = self.aa_dict[letter]
                        prot_encoder[j, 0, k, l] = 1.0
                        prot_mask_encoder[j, 0, k, 0] = 1.0

            for j, sequence in enumerate(decoder_output_sequence_list):
                for k, letter in enumerate(sequence):
                    if letter in self.aa_dict:
                        l = self.aa_dict[letter]
                        prot_decoder_output[j, 0, k, l] = 1.0
                        prot_mask_decoder[j, 0, k, 0] = 1.0

            for j, sequence in enumerate(decoder_input_sequence_list):
                for k, letter in enumerate(sequence):
                    if letter in self.aa_dict:
                        l = self.aa_dict[letter]
                        prot_decoder_input[j, 0, k, l] = 1.0

            self.protein_names_to_one_hot_seqs_encoder[family_name] = prot_encoder
            self.protein_names_to_one_hot_seqs_encoder_mask[family_name] = prot_mask_encoder
            self.protein_names_to_one_hot_seqs_decoder_input[family_name] = prot_decoder_input
            self.protein_names_to_one_hot_seqs_decoder_output[family_name] = prot_decoder_output
            self.protein_names_to_one_hot_seqs_decoder_mask[family_name] = prot_mask_decoder

    def generate_minibatch_idx(self, minibatch_size):

        # First choose
        family_idx = np.random.choice(len(self.family_idx_list), minibatch_size)

        one_hot_label = np.zeros((minibatch_size, len(self.family_idx_list)))

        sequence_idx = []
        for i, idx in enumerate(family_idx):
            sequence_idx.append(np.random.choice(self.family_idx_list[idx], 1)[0])
            one_hot_label[i][idx] = 1.

        return family_idx, sequence_idx, one_hot_label

    def gen_target_aa(self, x_input, x_input_lengths, dims):

        target_aa = []
        query_seq = np.copy(x_input)
        for i, input_len in enumerate(x_input_lengths):
            idx_rand = np.random.randint(input_len)
            target_aa.append(query_seq[i, 0, idx_rand].flatten().tolist())
            query_seq[i, 0, idx_rand] *= 0.
            query_seq[i, 0, idx_rand, dims["alphabet"] - 1] = 1.

        return query_seq, np.asarray(target_aa)

    def get_experimental_error(self, sess, conv_model, protein_name, step,
                               N_pred_iterations=10, batch_size=4, use_only_uppercase=True):

        prot_encoder = self.protein_names_to_one_hot_seqs_encoder[protein_name]
        prot_mask_encoder = self.protein_names_to_one_hot_seqs_encoder_mask[protein_name]
        prot_decoder_input = self.protein_names_to_one_hot_seqs_decoder_input[protein_name]
        prot_decoder_output = self.protein_names_to_one_hot_seqs_decoder_output[protein_name]
        prot_mask_decoder = self.protein_names_to_one_hot_seqs_decoder_mask[protein_name]

        measurement_list = self.protein_names_to_measurement_list[protein_name]

        # if only using the mutations from uppercase columns as in previous pubs
        if use_only_uppercase:
            idx_uppercase = self.protein_names_to_uppercase_idx[protein_name]

            prot_encoder = prot_encoder[idx_uppercase]
            prot_mask_encoder = prot_mask_encoder[idx_uppercase]
            prot_decoder_input = prot_decoder_input[idx_uppercase]
            prot_decoder_output = prot_decoder_output[idx_uppercase]
            prot_mask_decoder = prot_mask_decoder[idx_uppercase]
            measurement_list = measurement_list[idx_uppercase]

        prediction_matrix_ce = np.zeros((prot_encoder.shape[0], N_pred_iterations))
        prediction_matrix_all = np.zeros((prot_encoder.shape[0], N_pred_iterations))

        for i in range(N_pred_iterations):
            batch_order = np.arange(prot_encoder.shape[0])
            np.random.shuffle(batch_order)

            idx_family_one_hot = self.family_name_to_idx[protein_name]

            for j in range(0, prot_encoder.shape[0], batch_size):

                batch_index = batch_order[j:j + batch_size]
                batch_index_shape = prot_decoder_input[batch_index].shape[0]

                one_hot_family_label = np.zeros((batch_index_shape, self.x_train_encoder.shape[0]))
                for idx_len_minibatch in range(batch_index_shape):
                    one_hot_family_label[idx_len_minibatch][idx_family_one_hot] = 1.

                feed_dict = {conv_model.placeholders["sequences_start"]: prot_decoder_input[batch_index],
                             conv_model.placeholders["sequences_stop"]: prot_decoder_output[batch_index],
                             conv_model.placeholders["label"]: prot_mask_encoder[batch_index],
                             conv_model.placeholders["mask_decoder_1D"]: prot_mask_decoder[batch_index],
                             conv_model.placeholders["step"]: [step]}

                ce_loss, loss = sess.run([conv_model.tensors["cross_entropy_per_seq"],
                                          conv_model.tensors["loss_per_seq"]], feed_dict=feed_dict)

                for k, idx_batch in enumerate(batch_index.tolist()):
                    prediction_matrix_ce[idx_batch][i] = ce_loss[k]
                    prediction_matrix_all[idx_batch][i] = loss[k]

        prediction_ce = np.mean(prediction_matrix_ce, axis=1)
        prediction_all = np.mean(prediction_matrix_all, axis=1)

        return scipy.stats.spearmanr(prediction_ce, measurement_list)[0], \
               scipy.stats.spearmanr(prediction_all, measurement_list)[0]


class DataHelper:
    def __init__(self, dataset='', alignment_file='', focus_seq_name='',
                 mutation_file='', calc_weights=True, working_dir='.', theta=0.2,
                 load_all_sequences=True, alphabet_type='protein', max_seq_len=-1):

        np.random.seed(42)
        self.dataset = dataset
        self.alignment_file = alignment_file
        self.focus_seq_name = focus_seq_name
        self.mutation_file = mutation_file
        self.working_dir = working_dir
        self.calc_weights = calc_weights
        self.alphabet_type = alphabet_type
        self.max_seq_len = max_seq_len
        # Alignment processing parameters
        self.theta = theta

        # If I am running tests with the model, I don't need all the
        #    sequences loaded
        self.load_all_sequences = load_all_sequences

        # Load necessary information for preloaded datasets
        if self.dataset != '':
            self.configure_datasets()

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

        # then generate the experimental data
        self.gen_basic_alignment()

        if self.load_all_sequences:
            self.gen_full_alignment()

    def configure_datasets(self):

        if self.dataset == 'amp_2500':
            self.alignment_file = self.working_dir + '/datasets/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m'
            self.focus_seq_name = '>BLAT_ECOLX/24-286'
            self.mutation_file = self.working_dir + '/datasets/BLAT_ECOLX_Ranganathan2015_BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105_amp_2500_seq.csv'
            self.pdb_id = '1axb'
            self.chain_color_list = ['A']
            self.chain_show_list = ['A']
            self.show_organic = True

        elif self.dataset == 'methyltransferase':
            self.alignment_file = self.working_dir + '/datasets/MTH3_HAEAESTABILIZED_hmmerbit_plmc_n5_m30_f50_t0.2_r1-330_id100_b165.a2m'
            self.focus_seq_name = '>MTH3_HAEAESTABILIZED/1-330'
            self.mutation_file = self.working_dir + '/datasets/MTH3_HAEAESTABILIZED_Tawfik2015_MTH3_HAEAESTABILIZED_hmmerbit_plmc_n5_m30_f50_t0.2_r1-330_id100_b165_filtered_w17_seq.csv'
            self.pdb_id = '1dct'
            self.chain_color_list = ['A']
            self.chain_show_list = ['A', 'F', 'M']

        elif self.dataset == 'PABP':
            self.alignment_file = self.working_dir + '/datasets/PABP_YEAST_hmmerbit_plmc_n5_m30_f50_t0.2_r115-210_id100_b48.a2m'
            self.focus_seq_name = '>PABP_YEAST/115-210'
            self.mutation_file = self.working_dir + '/datasets/PABP_YEAST_Fields2013-all_PABP_YEAST_hmmerbit_plmc_n5_m30_f50_t0.2_r115-210_id100_b48_seq.csv'
            self.pdb_id = '1cvj'
            self.chain_color_list = ['A']
            self.chain_show_list = ['A', 'M']

        elif self.dataset == 'DLG':
            self.alignment_file = self.working_dir + '/datasets/DLG4_RAT_hmmerbit_plmc_n5_m30_f50_t0.2_r300-400_id100_b50.a2m'
            self.focus_seq_name = '>DLG4_RAT/300-400'
            self.mutation_file = self.working_dir + '/datasets/DLG4_RAT_Ranganathan2012_DLG4_RAT_hmmerbit_plmc_n5_m30_f50_t0.2_r300-400_id100_b50_seq.csv'
            self.pdb_id = '1be9'
            self.chain_color_list = ['A']
            self.chain_show_list = ['A']

        elif self.dataset == 'DLG_small':
            self.alignment_file = self.working_dir + '/datasets/DLG4_RAT_small.a2m'
            self.focus_seq_name = '>DLG4_RAT/300-400'
            self.mutation_file = self.working_dir + '/datasets/DLG4_RAT_Ranganathan2012_DLG4_RAT_hmmerbit_plmc_n5_m30_f50_t0.2_r300-400_id100_b50_seq.csv'
            self.pdb_id = '1be9'
            self.chain_color_list = ['A']
            self.chain_show_list = ['A']

    def one_hot_3D(self, s):
        # One-hot encode as row vector
        x = np.zeros((len(s), len(self.alphabet)))
        for i, letter in enumerate(s):
            if letter in self.aa_dict:
                x[i, self.aa_dict[letter]] = 1
        return x

    def gen_basic_alignment(self):

        # Make a dictionary that goes from aa to a number for one-hot
        self.aa_dict = {}
        for i, aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        # Do the inverse as well
        self.num_to_aa = {i: aa for aa, i in self.aa_dict.items()}

        ix = np.array([self.alphabet.find(s) for s in self.reorder_alphabet])

        # Read alignment
        self.seq_name_to_sequence = defaultdict(str)
        self.seq_names = []

        name = ''
        INPUT = open(self.alignment_file, 'r')
        for i, line in enumerate(INPUT):
            line = line.rstrip()
            if line.startswith('>'):
                name = line
                self.seq_names.append(name)
            else:
                self.seq_name_to_sequence[name] += line
        INPUT.close()

        self.seq_name_to_sequence_aligned = {}
        self.seq_name_to_sequence_encoder = {}
        self.seq_name_to_sequence_decoder_input = {}
        self.seq_name_to_sequence_decoder_output = {}

        if self.focus_seq_name == '':
            self.focus_seq_name = self.seq_names[0]

        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
        self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper()]

        alphabet_set = set(list(self.alphabet))
        seq_names_to_remove = []

        # Add the end character
        for name, sequence in self.seq_name_to_sequence.items():

            ungapped_seq = sequence.replace('.', '')
            ungapped_seq = ungapped_seq.replace('-', '')
            ungapped_seq = ungapped_seq.upper()

            valid_sequence = True

            for letter in ungapped_seq:
                if letter not in alphabet_set:
                    valid_sequence = False
                    seq_names_to_remove.append(name)

            if valid_sequence:
                self.seq_name_to_sequence_aligned[name] = "".join([sequence[ix] for ix in self.focus_cols])
                self.seq_name_to_sequence_encoder[name] = ungapped_seq
                self.seq_name_to_sequence_decoder_input[name] = '*' + ungapped_seq
                self.seq_name_to_sequence_decoder_output[name] = ungapped_seq + '*'

        # Remove sequences that have bad characters
        seq_names_to_remove = list(set(seq_names_to_remove))
        for seq_name in seq_names_to_remove:
            del self.seq_name_to_sequence[seq_name]

        # If we don't have a focus sequence, pick the one that
        #   we used to generate the alignment
        if self.focus_seq_name == '':
            self.focus_seq_name = self.seq_names[0]

        # Select focus columns

        if self.max_seq_len == -1:
            self.seq_len = len(self.seq_name_to_sequence[self.focus_seq_name])
        else:
            self.seq_len = self.max_seq_len

        self.alphabet_size = len(self.alphabet)

    def gen_full_alignment(self):

        # Encode the sequences
        print('Encoding sequences')
        self.x_train_aligned = np.zeros(
            (len(list(self.seq_name_to_sequence.keys())), len(self.focus_cols), len(self.alphabet)))
        self.x_train_name_list = []
        for i, seq_name in enumerate(self.seq_name_to_sequence.keys()):
            sequence = self.seq_name_to_sequence_aligned[seq_name]
            self.x_train_name_list.append(seq_name)
            for j, letter in enumerate(sequence):
                if letter in self.aa_dict:
                    k = self.aa_dict[letter]
                    self.x_train_aligned[i, j, k] = 1.0

        # Make a non-gapped version
        self.x_train_encoder = np.zeros(
            (len(list(self.seq_name_to_sequence.keys())), 1, self.seq_len, len(self.alphabet)))
        self.x_train_lengths = np.zeros((len(list(self.seq_name_to_sequence.keys()))))
        self.x_train_mask_encoder = np.zeros((len(list(self.seq_name_to_sequence.keys())), 1, self.seq_len, 1))

        for i, seq_name in enumerate(self.x_train_name_list):
            sequence = self.seq_name_to_sequence_encoder[seq_name]
            for j, letter in enumerate(sequence):
                if j < self.seq_len:
                    if letter in self.aa_dict:
                        k = self.aa_dict[letter]
                        self.x_train_encoder[i, 0, j, k] = 1.0
                        self.x_train_mask_encoder[i, 0, j, 0] = 1.0
                self.x_train_lengths[i] = len(sequence)

        self.x_train_decoder_output = np.zeros(
            (len(list(self.seq_name_to_sequence.keys())), 1, self.seq_len + 1, len(self.alphabet)))
        self.x_train_decoder_input = np.zeros(
            (len(list(self.seq_name_to_sequence.keys())), 1, self.seq_len + 1, len(self.alphabet)))
        self.x_train_mask_decoder = np.zeros((len(list(self.seq_name_to_sequence.keys())), 1, self.seq_len + 1, 1))

        for i, seq_name in enumerate(self.x_train_name_list):
            sequence = self.seq_name_to_sequence_decoder_output[seq_name]
            for j, letter in enumerate(sequence):
                if j < self.seq_len:
                    if letter in self.aa_dict:
                        k = self.aa_dict[letter]
                        self.x_train_decoder_output[i, 0, j, k] = 1.0
                        self.x_train_mask_decoder[i, 0, j, 0] = 1.0

        for i, seq_name in enumerate(self.x_train_name_list):
            sequence = self.seq_name_to_sequence_decoder_input[seq_name]
            for j, letter in enumerate(sequence):
                if j < self.seq_len:
                    if letter in self.aa_dict:
                        k = self.aa_dict[letter]
                        self.x_train_decoder_input[i, 0, j, k] = 1.0

        # Fast sequence weights with Theano
        if self.calc_weights:
            print('Computing sequence weights')
            # Numpy version
            # self.weights = scale / np.sum(squareform(pdist(seq_index_array, metric='hamming')) < theta, axis=0)
            #
            # Theano weights
            X = T.tensor3("x")
            cutoff = T.scalar("theta")
            X_flat = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
            N_list, updates = theano.map(lambda x: 1.0 / T.sum(T.dot(X_flat, x) / T.dot(x, x) > 1 - cutoff), X_flat)
            weightfun = theano.function(inputs=[X, cutoff], outputs=[N_list], allow_input_downcast=True)
            #
            self.weights = weightfun(self.x_train_aligned, self.theta)[0]
        else:
            self.weights = np.ones(self.x_train_aligned.shape[0])

        self.Neff = np.sum(self.weights)

        print('Neff =', str(self.Neff))
        print("Data Shape =", self.x_train_aligned.shape)

        # Only set up the mutation data if we have it
        #   this will make it generalizable to all protein families
        """
        if self.mutation_file != '':
            INPUT = open(self.mutation_file, 'r')

            self.focus_seq_mut = ''
            self.mut_seq_to_measurement = {}
            self.hamming_distance_to_wt_list = []
            self.mut_seq_list = []

            one_hot_list = []
            measurement_list = []

            for i,line in enumerate(INPUT):
                line = line.rstrip()
                line_list = line.split(',')
                if i == 0:
                    self.focus_seq_mut = line_list[0]
                else:

                    mut_seq = line_list[0]

                    # If I just want to get a bunch of elbos, do not read any
                    #    measurements in here and fill it with dummy variables
                    if self.report_expr_accuracy:
                        measurement = float(line_list[1])
                    else:
                        measurement = 0.0

                    one_hot_mut_seq = self.one_hot_3D(mut_seq)

                    self.mut_seq_to_measurement[mut_seq] = measurement
                    self.mut_seq_list.append(mut_seq)

                    one_hot_list.append(one_hot_mut_seq)
                    measurement_list.append(measurement)

                    # Then calculate hamming distance to wt
                    dist = 0
                    for mut_aa,wt_aa in zip(mut_seq,self.focus_seq_mut):
                        if mut_aa != wt_aa:
                            dist += 1

                    self.hamming_distance_to_wt_list.append(dist)

            INPUT.close()

            self.measurement_array = np.asarray(measurement_list)
            self.one_hot_mut_array = np.asarray(one_hot_list)

            self.one_hot_mut_array_with_wt = np.asarray([self.one_hot_3D(self.focus_seq_mut)] + one_hot_list)
        """

    def gen_target_aa(self, x_input, x_input_lengths, dims):

        target_aa = []
        query_seq = np.copy(x_input)
        for i, input_len in enumerate(x_input_lengths):
            idx_rand = np.random.randint(input_len)
            target_aa.append(query_seq[i, 0, idx_rand].flatten().tolist())
            query_seq[i, 0, idx_rand] *= 0.
            query_seq[i, 0, idx_rand, dims["alphabet"] - 1] = 1.

        return query_seq, np.asarray(target_aa)

    def get_elbo_samples(self, model, N_pred_iterations=100, minibatch_size=2000):

        print(self.one_hot_mut_array_with_wt.shape)
        self.prediction_matrix = np.zeros((self.one_hot_mut_array_with_wt.shape[0], N_pred_iterations))

        batch_order = np.arange(self.one_hot_mut_array_with_wt.shape[0])

        for i in range(N_pred_iterations):
            np.random.shuffle(batch_order)

            for j in range(0, self.one_hot_mut_array_with_wt.shape[0], minibatch_size):

                batch_index = batch_order[j:j + minibatch_size]
                batch_preds, _, _ = model.all_likelihood_components(self.one_hot_mut_array_with_wt[batch_index])

                for k, idx_batch in enumerate(batch_index.tolist()):
                    self.prediction_matrix[idx_batch][i] = batch_preds[k]

    def get_experimental_error(self, model, plot_predictions=False, png_prefix='',
                               N_pred_iterations=100, minibatch_size=2000):

        self.prediction_matrix = np.zeros((self.one_hot_mut_array.shape[0], N_pred_iterations))

        for i in range(N_pred_iterations):
            batch_order = np.arange(self.one_hot_mut_array.shape[0])
            np.random.shuffle(batch_order)

            for j in range(0, self.one_hot_mut_array.shape[0], minibatch_size):

                batch_index = batch_order[j:j + minibatch_size]
                batch_preds, _, _ = model.all_likelihood_components(self.one_hot_mut_array[batch_index])

                for k, idx_batch in enumerate(batch_index.tolist()):
                    self.prediction_matrix[idx_batch][i] = batch_preds[k]

        self.prediction = np.mean(self.prediction_matrix, axis=1)

        if self.report_expr_accuracy:

            # Initialize a dictionary that will hold our predictions
            self.mutation_prediction_error_dict = {}
            self.mutation_prediction_error_dict['All data'] = {}
            self.mutation_prediction_error_dict['All data']['spearman'] = \
            spearmanr(self.prediction, self.measurement_array)[0]
            self.mutation_prediction_error_dict['All data']['pearson'] = \
            pearsonr(self.prediction, self.measurement_array)[0]
            self.mutation_prediction_error_dict['All data']['N'] = self.prediction.shape[0]

            for hamming_distance in list(set(self.hamming_distance_to_wt_list)):
                idx = []

                for i, dist in enumerate(self.hamming_distance_to_wt_list):
                    if dist == hamming_distance:
                        idx.append(i)

                dict_key = "Hamming distance " + str(hamming_distance)
                self.mutation_prediction_error_dict[dict_key] = {}
                self.mutation_prediction_error_dict[dict_key]['spearman'] = \
                spearmanr(self.prediction[idx], self.measurement_array[idx])[0]
                self.mutation_prediction_error_dict[dict_key]['pearson'] = \
                pearsonr(self.prediction[idx], self.measurement_array[idx])[0]
                self.mutation_prediction_error_dict[dict_key]['N'] = len(idx)

            # If we just have singles, only print out all measurement accuracy
            if len(list(self.mutation_prediction_error_dict.keys())) == 2:
                val_dict = self.mutation_prediction_error_dict['All data']
                error_string = "All data, Spearman: " + str(val_dict['spearman']) + ", Pearson: " + str(
                    val_dict['pearson']) + ", N: " + str(val_dict['N'])

            # Otherwise print out all of it
            else:
                error_string = ''
                for key, val_dict in sorted(self.mutation_prediction_error_dict.items()):
                    if error_string == '':
                        error_string += str(key) + ", Spearman: " + str(val_dict['spearman']) + ", Pearson: " + str(
                            val_dict['pearson']) + ", N: " + str(val_dict['N'])
                    else:
                        error_string += "\n" + str(key) + ", Spearman: " + str(
                            val_dict['spearman']) + ", Pearson: " + str(val_dict['pearson']) + ", N: " + str(
                            val_dict['N'])
        else:
            error_string = ''

        return error_string
