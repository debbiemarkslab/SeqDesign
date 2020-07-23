import tensorflow as tf
import numpy as np
from seqdesign import layers_auto_b as mpnn_layers


class AutoregressiveFR:
    def __init__(self, dims={}, hyperparams={}, channels=48, r_seed=42):

        self.hyperparams = {
            # For purely dilated conv network
            "encoder": {
                "channels": channels,
                "nonlinearity": "relu",
                "dilation_schedule": [1, 2, 4, 8, 16, 32, 64, 128, 256],
                "num_dilation_blocks": 6,
                "transformer": False,
                "inverse_temperature": False,
                "dropout_type": "inter"  # options = "final","inter","gaussian"
            },
            "sampler_hyperparams": {
                'warm_up': 1,
                'annealing_type': 'linear',
                'anneal_KL': True,
                'anneal_noise': True
            },
            "embedding_hyperparams": {
                'warm_up': 1,
                'annealing_type': 'linear',
                'anneal_KL': True,
                'anneal_noise': False
            },
            "random_seed": r_seed,
            "optimization": {
                "grad_clip": 100.0,
                "learning_rate": 0.001,
                "l2_regularization": True,
                "bayesian": False,
                "l2_lambda": 1.,
                "bayesian_logits": False,
                "mle_logits": False,
                "run_backward": True,
                "ema": False,
            }
        }

        self.dims = {
            "batch": 10,
            "alphabet": 21,
            "length": 256,
            "embedding_size": 1
        }

        # Merge with dictionary
        for key, value in dims.items():
            self.dims[key] = value

        # Model placeholders
        self.placeholders = {}
        self.model_type = 'autoregressive'

        print(self.dims["batch"], 1, self.dims["length"], self.dims["alphabet"])

        self.dims["batch"] = None
        self.dims["length"] = None

        self.placeholders["step"] = tf.placeholder(tf.float32, 1, name="update_num")

        self.placeholders["dropout"] = tf.placeholder_with_default(0.5, (), name="Dropout")

        # Forward sequences
        self.placeholders["sequences_start_f"] = tf.placeholder(
            tf.float32, [self.dims["batch"], 1, self.dims["length"], self.dims["alphabet"]],
            name="SequencesDecoderInF")

        self.placeholders["sequences_stop_f"] = tf.placeholder(
            tf.float32, [self.dims["batch"], 1, self.dims["length"], self.dims["alphabet"]],
            name="SequencesDecoderOutF")

        self.placeholders["mask_decoder_1D_f"] = tf.placeholder(
            tf.float32, [self.dims["batch"], 1, self.dims["length"], 1], name="SequenceDecoderMaskR")

        self.placeholders["Neff_f"] = tf.placeholder(tf.float32, 1, name="NeffF")

        # Reverse sequences
        self.placeholders["sequences_start_r"] = tf.placeholder(
            tf.float32, [self.dims["batch"], 1, self.dims["length"], self.dims["alphabet"]],
            name="SequencesDecoderInR")

        self.placeholders["sequences_stop_r"] = tf.placeholder(
            tf.float32, [self.dims["batch"], 1, self.dims["length"], self.dims["alphabet"]],
            name="SequencesDecoderOutR")

        self.placeholders["mask_decoder_1D_r"] = tf.placeholder(
            tf.float32, [self.dims["batch"], 1, self.dims["length"], 1], name="SequenceDecoderMaskR")

        self.placeholders["Neff_r"] = tf.placeholder(tf.float32, 1, name="NeffR")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        if self.hyperparams["encoder"]["transformer"]:
            mask_2D_f = self.placeholders["mask_decoder_1D_f"] * tf.transpose(self.placeholders["mask_decoder_1D_f"],
                                                                              [0, 2, 1, 3])
            mask_tri_f = tf.matrix_band_part(tf.transpose(mask_2D_f, [0, 3, 1, 2]), -1, 0)
            mask_tri_f = tf.transpose(mask_tri_f, [0, 2, 3, 1])

            mask_2D_r = self.placeholders["mask_decoder_1D_r"] * tf.transpose(self.placeholders["mask_decoder_1D_r"],
                                                                              [0, 2, 1, 3])
            mask_tri_r = tf.matrix_band_part(tf.transpose(mask_2D_r, [0, 3, 1, 2]), -1, 0)
            mask_tri_r = tf.transpose(mask_tri_r, [0, 2, 3, 1])

        else:
            mask_2D_f = None
            mask_tri_f = None

            mask_2D_r = None
            mask_tri_r = None

        # Model tensors
        self.tensors = {}

        with tf.variable_scope("Forward"):
            self.tensors["sequence_logits_f"], weight_cost_list, KL_logits \
                = self._build_encoder(self.placeholders["sequences_start_f"], self.placeholders["mask_decoder_1D_f"],
                                      self.placeholders["step"],
                                      mask_2D_f,
                                      mask_tri_f)

            self.tensors["reconstruction_f"], self.tensors["cross_entropy_loss_f"], \
            self.tensors["KL_embedding_loss_f"], \
            self.tensors["KL_loss_f"], self.tensors["loss_f"], \
            self.tensors["cross_entropy_per_seq_f"], self.tensors["loss_per_seq_f"] \
                = self.calculate_loss(
                self.placeholders["sequences_stop_f"], self.tensors["sequence_logits_f"], self.placeholders["step"],
                self.placeholders["mask_decoder_1D_f"], self.placeholders["Neff_f"], weight_cost_list, KL_logits)

        if self.hyperparams["optimization"]["run_backward"]:
            with tf.variable_scope("Reverse"):
                self.tensors["sequence_logits_r"], weight_cost_list, KL_logits \
                    = self._build_encoder(self.placeholders["sequences_start_r"],
                                          self.placeholders["mask_decoder_1D_r"], self.placeholders["step"],
                                          mask_2D_r,
                                          mask_tri_r)

                self.tensors["reconstruction_r"], self.tensors["cross_entropy_loss_r"], \
                self.tensors["KL_embedding_loss_r"], \
                self.tensors["KL_loss_r"], self.tensors["loss_r"], \
                self.tensors["cross_entropy_per_seq_r"], self.tensors["loss_per_seq_r"] \
                    = self.calculate_loss(
                    self.placeholders["sequences_stop_r"], self.tensors["sequence_logits_r"], self.placeholders["step"],
                    self.placeholders["mask_decoder_1D_r"], self.placeholders["Neff_r"], weight_cost_list, KL_logits)

            self.tensors["loss"] = self.tensors["loss_f"] + self.tensors["loss_r"]
            self.tensors["cross_entropy_loss"] = self.tensors["cross_entropy_loss_f"] + self.tensors[
                "cross_entropy_loss_r"]
            self.tensors["KL_embedding_loss"] = self.tensors["KL_embedding_loss_f"] + self.tensors[
                "KL_embedding_loss_r"]

        else:
            self.tensors["loss"] = self.tensors["loss_f"]
            self.tensors["cross_entropy_loss"] = self.tensors["cross_entropy_loss_f"]
            self.tensors["KL_embedding_loss"] = self.tensors["KL_embedding_loss_f"]

        with tf.variable_scope("SummariesDecoder"):
            tf.summary.scalar("ReconstructionLoss", self.tensors["cross_entropy_loss"])

            tf.summary.scalar("Loss", self.tensors["loss"])

            tf.summary.scalar("RegularizationLoss", self.tensors["KL_embedding_loss"])

            # from https://github.com/tensorflow/tensorflow/issues/675#issuecomment-299669705
            reconstr_f_T = tf.transpose(self.tensors["reconstruction_f"], [2, 0, 1, 3])
            reconstr_f_T_shape = tf.shape(reconstr_f_T)

            loop_vars = [
                tf.constant(0, tf.int32),
                tf.TensorArray(tf.float32, size=reconstr_f_T_shape[0]),
            ]

            _, jacobian = tf.while_loop(
                lambda j, _: j < reconstr_f_T_shape[0],
                lambda j, result: (
                j + 1, result.write(j, tf.gradients(reconstr_f_T[j], self.placeholders["sequences_start_f"]))),
                loop_vars)

            # Size = (L, 1, batch, 1 , L , alphabet)
            jacobian_arr = jacobian.stack()

            # Size = (L, batch, L , alphabet)
            jacobian_arr = tf.squeeze(jacobian_arr)

            # Size = (batch, L, L, 1)
            self.tensors["gradient_norm_input_output_f"] = tf.transpose(tf.norm(jacobian_arr, axis=-1, keepdims=True),
                                                                        perm=[1, 0, 2, 3])

        with tf.variable_scope("Backprop"):
            opt = tf.train.AdamOptimizer(learning_rate=self.hyperparams["optimization"]["learning_rate"])
            gvs = opt.compute_gradients(self.tensors["loss"])
            gradients, grad_norm = self.clip_gradients(gvs)

            if self.hyperparams["optimization"]["ema"]:
                opt_op = opt.apply_gradients(gradients, global_step=self.global_step)
                self.ema = tf.train.ExponentialMovingAverage(0.995, name='EMA')
                maintain_averages_op = self.ema.apply(tf.trainable_variables())
                with tf.control_dependencies([opt_op]):
                    self.opt_op = tf.group(maintain_averages_op)

            else:
                self.opt_op = opt.apply_gradients(gradients, global_step=self.global_step)

    def _nonlinearity(self, nonlin_type):
        if nonlin_type == "elu":
            return tf.nn.elu

    def _KLD_standard_normal(self, mu, log_sigma):
        """ KL divergence between two Diagonal Gaussians """
        return 0.5 * (1.0 + 2.0 * log_sigma - tf.square(mu) - tf.exp(2.0 * log_sigma))

    def _KLD_diag_gaussians(self, mu, log_sigma, prior_mu, prior_log_sigma):
        """ KL divergence between two Diagonal Gaussians """
        return prior_log_sigma - log_sigma + 0.5 * (tf.exp(2 * log_sigma) + tf.square(mu - prior_mu)) * tf.exp(-2. * prior_log_sigma) - 0.5

    def clip_gradients(self, gvs):
        """ Clip the gradients """
        gvs_zip = zip(*gvs)
        grads, gvars = list(next(gvs_zip)), list(next(gvs_zip))
        clipped_grads, global_norm = tf.clip_by_global_norm(grads, self.hyperparams["optimization"]["grad_clip"])
        tf.summary.scalar("gradient_norm", global_norm)
        clipped_gvs = list(zip(clipped_grads, gvars))
        return clipped_gvs, global_norm

    def _anneal(self, step):
        warm_up = self.hyperparams["sampler_hyperparams"]["warm_up"]
        annealing_type = self.hyperparams["sampler_hyperparams"]["annealing_type"]
        if annealing_type == "linear":
            return tf.squeeze(tf.minimum(step / warm_up, 1.))
        elif annealing_type == "piecewise_linear":
            return tf.squeeze(tf.minimum(tf.nn.sigmoid(step - warm_up) * ((step - warm_up) / warm_up), 1.))
        elif annealing_type == "sigmoid":
            slope = self.hyperparams["sampler_hyperparams"]["sigmoid_slope"]
            return tf.squeeze(tf.nn.sigmoid(slope * (step - warm_up)))

    def _anneal_embedding(self, step):
        warm_up = self.hyperparams["embedding_hyperparams"]["warm_up"]
        annealing_type = self.hyperparams["embedding_hyperparams"]["annealing_type"]
        if annealing_type == "linear":
            return tf.squeeze(tf.minimum(step / warm_up, 1.))
        elif annealing_type == "piecewise_linear":
            return tf.squeeze(tf.minimum(tf.nn.sigmoid(step - warm_up) * ((step - warm_up) / warm_up), 1.))
        elif annealing_type == "sigmoid":
            slope = self.hyperparams["embedding_hyperparams"]["sigmoid_slope"]
            return tf.squeeze(tf.nn.sigmoid(slope * (step - warm_up)))

    def sampler(self, mu, log_sigma, stddev=1.):
        if self.hyperparams["embedding_hyperparams"]["anneal_noise"]:
            stddev = self._anneal_embedding(self.placeholders["step"])
        with tf.variable_scope("Sampler"):
            # shape = tf.shape(mu)
            eps = tf.random_normal(tf.shape(mu), stddev=stddev)
            return mu + tf.exp(log_sigma) * eps

    def _log_gaussian(self, z, prior_mu, prior_sigma):
        prior_var = tf.square(prior_sigma)
        return -0.5 * tf.log(2. * np.pi * prior_var) \
               - tf.square(z - prior_mu) / (2. * prior_var)

    def _KL_mixture_gaussians(self, z, mu, log_sigma, p=0.1, mu_one=0., mu_two=0., sigma_one=1., sigma_two=1.):
        gauss_one = self._log_gaussian(z, mu_one, sigma_one)
        gauss_two = self._log_gaussian(z, mu_two, sigma_two)
        entropy = 0.5 * tf.log(2.0 * np.pi * np.e * tf.exp(2. * log_sigma))
        return (p * gauss_one) + ((1. - p) * gauss_two) + entropy

    def _MLE_mixture_gaussians(self, z, p=0.1, mu_one=0., mu_two=0., sigma_one=1., sigma_two=1.):
        gauss_one = self._log_gaussian(z, mu_one, sigma_one)
        gauss_two = self._log_gaussian(z, mu_two, sigma_two)
        return (p * gauss_one) + ((1. - p) * gauss_two)

    def _build_encoder(self, input_1D, mask_1D, step, mask_2D, mask_tri, reuse=None):

        hyperparams = self.hyperparams["encoder"]
        nonlin = self._nonlinearity(hyperparams["nonlinearity"])
        channels = self.hyperparams["encoder"]["channels"]
        dilation_schedule = self.hyperparams["encoder"]["dilation_schedule"]
        width_schedule = len(dilation_schedule) * [3]
        dilation_blocks = self.hyperparams["encoder"]["num_dilation_blocks"]
        dims = self.dims

        weight_cost_list = []

        if self.hyperparams["optimization"]["bayesian"]:
            sampler_hyperparams = self.hyperparams["sampler_hyperparams"]

        with tf.variable_scope("EncoderPrepareInput"):
            if self.hyperparams["optimization"]["bayesian"]:
                up_val_1D, weight_cost = mpnn_layers.conv2D_bayesian(input_1D, channels, step, sampler_hyperparams,
                                                                     mask=mask_1D, activation=nonlin)
            else:
                up_val_1D, weight_cost = mpnn_layers.conv2D(
                    input_1D, name="Features1D",
                    filters=channels, kernel_size=(1, 1),
                    activation=nonlin, padding="same", reuse=reuse)

            weight_cost_list += weight_cost

        with tf.variable_scope("Encoder"):
            for i in range(dilation_blocks):
                with tf.variable_scope("DilationBlock" + str(i + 1)):

                    if self.hyperparams["optimization"]["bayesian"]:

                        up_val_1D, weight_cost = mpnn_layers.convnet_1D_generative_bayesian_standard(up_val_1D, channels, mask_1D, width_schedule,
                                                                                                     dilation_schedule, step, sampler_hyperparams)

                    else:

                        up_val_1D, weight_cost = mpnn_layers.convnet_1D_generative(up_val_1D, channels, mask_1D, width_schedule, dilation_schedule,
                                                                                   dropout_p=self.placeholders["dropout"])

                    weight_cost_list += weight_cost

            tf.summary.image("LayerFeatures", tf.transpose(up_val_1D, perm=[0, 3, 2, 1]), 3)

        with tf.variable_scope("WriteSequence"):

            if hyperparams["dropout_type"] == "final":
                print("final dropout")
                final_dropout_p = self.placeholders["dropout"] + 0.3
                final_dropout_p = tf.cond(final_dropout_p > 1., lambda: 1., lambda: final_dropout_p)
                up_val_1D = tf.nn.dropout(up_val_1D, final_dropout_p)

            if self.hyperparams["optimization"]["bayesian"]:
                sequence_logits, weight_cost = mpnn_layers.conv2D_bayesian(up_val_1D, dims["alphabet"], step, sampler_hyperparams,
                                                                           mask=mask_1D, activation=None)
            else:
                sequence_logits, weight_cost = mpnn_layers.conv2D(up_val_1D, dims["alphabet"], mask=mask_1D, activation=None, g_init=0.1)

            weight_cost_list += weight_cost

            if self.hyperparams["optimization"]["mle_logits"]:

                KL_logits = - self._MLE_mixture_gaussians(sequence_logits, p=.6, mu_one=0., mu_two=0., sigma_one=1.25, sigma_two=3.)

            else:

                KL_logits = None

        return sequence_logits, weight_cost_list, KL_logits

    def calculate_loss(self, sequences, seq_logits, step, mask, Neff, weight_cost_list, KL_logits):

        hyperparams = self.hyperparams

        if hyperparams["optimization"]["l2_regularization"] or hyperparams["optimization"]["bayesian"]:

            with tf.variable_scope("CrossEntropyLoss"):
                L_total = tf.reduce_sum(mask)

                seq_reconstruct = tf.nn.softmax(seq_logits, dim=-1) * mask

                seq_logits_mask = seq_logits * mask

                cross_entropy = tf.reduce_sum(
                    tf.nn.softmax_cross_entropy_with_logits(labels=sequences, logits=seq_logits_mask), axis=[1, 2])

                reconstruction_per_seq = cross_entropy

                reconstruction_loss = tf.reduce_mean(cross_entropy)

            with tf.variable_scope("RegularizationCalculation"):

                weight_cost = tf.reduce_sum(tf.stack(weight_cost_list)) / tf.squeeze(Neff)

                weight_cost = weight_cost * hyperparams["optimization"]["l2_lambda"]

                KL_weight_loss = weight_cost

                KL_loss = weight_cost

            with tf.variable_scope("MergeLosses"):

                loss_per_seq = reconstruction_per_seq

                loss = reconstruction_loss + weight_cost

            if hyperparams["optimization"]["bayesian_logits"] or hyperparams["optimization"]["mle_logits"]:

                print("KL logits")
                KL_logits = KL_logits * mask

                KL_logits_per_seq = tf.reduce_sum(KL_logits, axis=[1, 2, 3])

                loss_per_seq = loss_per_seq + KL_logits_per_seq

                KL_logits_loss = tf.reduce_mean(KL_logits_per_seq)

                KL_loss += KL_logits_loss

                KL_embedding_loss = KL_logits_loss

                loss = loss + self._anneal_embedding(step) * KL_logits_loss

            else:
                KL_embedding_loss = KL_weight_loss


        else:
            with tf.variable_scope("CrossEntropyLoss"):
                L_total = tf.reduce_sum(mask)

                seq_reconstruct = tf.nn.softmax(seq_logits, dim=-1) * mask

                seq_logits_mask = seq_logits * mask

                cross_entropy = tf.reduce_sum(
                    tf.nn.softmax_cross_entropy_with_logits(labels=sequences, logits=seq_logits_mask), axis=[1, 2])

                reconstruction_per_seq = cross_entropy

                reconstruction_loss = tf.reduce_sum(cross_entropy) / L_total

            with tf.variable_scope("KLCalculation"):

                KL_embedding_loss = tf.constant(0, dtype=tf.float32)

                KL_loss = tf.constant(0, dtype=tf.float32)

            with tf.variable_scope("MergeLosses"):

                loss_per_seq = reconstruction_per_seq

                loss = reconstruction_loss

        with tf.variable_scope("SummariesDecoder"):

            tf.summary.scalar("ReconstructionLoss", reconstruction_loss)

            tf.summary.scalar("Loss", loss)

            if hyperparams["optimization"]["bayesian_logits"] or hyperparams["optimization"]["mle_logits"]:
                tf.summary.scalar("LogitLoss", KL_logits_loss)

            tf.summary.scalar("ParamLoss", KL_weight_loss)

            tf.summary.image(
                "SeqReconstruct", tf.transpose(seq_reconstruct, perm=[0, 3, 2, 1]), 3)

            tf.summary.image(
                "SeqTarget", tf.transpose(sequences, perm=[0, 3, 2, 1]), 3)

            tf.summary.image(
                "SeqDelta", tf.transpose(seq_reconstruct - sequences, perm=[0, 3, 2, 1]), 3)

        return seq_reconstruct, reconstruction_loss, KL_embedding_loss, \
               KL_loss, loss, reconstruction_per_seq, loss_per_seq
