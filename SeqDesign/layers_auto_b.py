import tensorflow as tf
import numpy as np


def nonlinearity(x):
    return tf.nn.elu(x)


"""
def normalize_layer(X, mask, per_channel=False, reduce_dims=[1,2]):
    with tf.variable_scope("Normalization"):
        N = tf.reduce_sum(mask, reduce_dims, keep_dims=True)
        if not per_channel:
            N *= X.get_shape().as_list()[3]
        X = mask * X
        av_X = tf.reduce_sum(X, reduce_dims, keep_dims=True) / N
        sqdev_X = tf.square(X - av_X)
        var_X = tf.reduce_sum(mask * sqdev_X, reduce_dims, keep_dims=True) / N
        sigma_X = tf.sqrt(var_X + 1E-5)
        X = (X - av_X) / sigma_X
    return X * mask
"""


def normalize_layer(X, mask, per_channel=False, reduce_dims=[3]):
    with tf.variable_scope("Normalization"):
        mean, var = tf.nn.moments(X, reduce_dims, keep_dims=True)
        X = (X - mean) / tf.sqrt(var + 1E-5)
    return X


def scale_shift(X, mask, channels, name="ScaleAndShift"):
    with tf.variable_scope(name):
        X_shape = tf.shape(X)
        param_init = tf.random_normal_initializer(1, 0.000001)

        # transform with a gain and bias
        g = tf.get_variable(
            "g", shape=[channels], dtype=tf.float32,
            initializer=param_init,
            trainable=True
        )
        b = tf.get_variable(
            "b", shape=[channels], dtype=tf.float32,
            initializer=param_init,
            trainable=True
        )

        shape = [1, 1, 1, channels]

        X = X * tf.reshape(g, shape) + tf.reshape(b, shape)

    return X * mask


def normalize_channels(X, mask):
    with tf.variable_scope("Normalization"):
        N = X.get_shape().as_list()[3]
        X = mask * X
        av_X = tf.reduce_sum(X, 3, keep_dims=True) / N
        sqdev_X = tf.square(X - av_X)
        var_X = tf.reduce_sum(sqdev_X, 3, keep_dims=True) / N
        sigma_X = tf.sqrt(var_X + 1E-5)
        X = mask * (X - av_X) / sigma_X
    return X


def scale_variables(X, scale_type="pos"):
    with tf.variable_scope("scale_variable-" + scale_type):
        bias_initializer = np.ones(1)

        with tf.device('/cpu:0'):

            scale = tf.get_variable("scale", 1, dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0001 * bias_initializer), trainable=True)

            if scale_type == "pos":
                loc = tf.nn.softplus(tf.get_variable("loc", 1, dtype=tf.float32,
                                                     initializer=tf.constant_initializer(0.0001 * bias_initializer),
                                                     trainable=True))

            elif scale_type == "neg":
                loc = tf.get_variable("loc", 1, dtype=tf.float32,
                                      initializer=tf.constant_initializer(-7. * bias_initializer), trainable=True)

        return tf.squeeze(scale) * X + tf.squeeze(loc)


def normclip(X, mask, per_channel=False):
    with tf.variable_scope("Normalization"):
        reduce_dims = [1, 2] if per_channel else [1, 2, 3]
        N = tf.reduce_sum(mask, reduce_dims, keep_dims=True)
        if not per_channel:
            N *= X.get_shape().as_list()[3]
        X = mask * X
        X_square = tf.square(X)
        norm_X = tf.reduce_sum(X_square, reduce_dims, keep_dims=True) / N
        norm_X = tf.sqrt(norm_X + 0.01)
        new_norm = tf.minimum(norm_X, 1.0)
        X = mask * X * new_norm / norm_X
    return X


def _anneal(step, sampler_hyperparams):
    warm_up = sampler_hyperparams["warm_up"]
    annealing_type = sampler_hyperparams["annealing_type"]
    if annealing_type == "linear":
        return tf.squeeze(tf.minimum(step / warm_up, 1.))
    elif annealing_type == "piecewise_linear":
        return tf.squeeze(tf.minimum(tf.nn.sigmoid(step - warm_up) * ((step - warm_up) / warm_up), 1.))
    elif annealing_type == "sigmoid":
        slope = sampler_hyperparams["sigmoid_slope"]
        return tf.squeeze(tf.nn.sigmoid(slope * (step - warm_up)))


def _sampler(mu, log_sigma, step, sampler_hyperparams, stddev=1.):
    stddev = _anneal(step, sampler_hyperparams)
    with tf.variable_scope("Sampler"):
        shape = tf.shape(mu)
        eps = tf.random_normal(shape, stddev=stddev)
        return mu + tf.exp(log_sigma) * eps


def _KLD_diag_gaussians(mu, log_sigma, prior_mu, prior_log_sigma):
    """ KL divergence between two Diagonal Gaussians """
    return prior_log_sigma - log_sigma + 0.5 * (tf.exp(2 * log_sigma)
                                                + tf.square(mu - prior_mu)) * tf.exp(-2. * prior_log_sigma) - 0.5


def _KLD_standard_normal(mu, log_sigma):
    """ KL divergence between two Diagonal Gaussians """
    return -0.5 * tf.reduce_sum(1.0 + 2.0 * log_sigma - tf.square(mu) - tf.exp(2.0 * log_sigma))


def conv2D_generative_bayesian(inputs, filters, step,
                               sampler_hyperparams, kernel_size=(1, 1),
                               padding="same", activation=None,
                               reuse=False, name="conv2D", strides=(1, 1),
                               dilation_rate=(1, 1), use_bias=True, mask=None):
    """ 2D convolution layer with weight normalization """

    with tf.variable_scope(name, reuse=reuse):
        # Ensure argument types
        strides = list(strides)
        dilation_rate = list(dilation_rate)
        kernel_size = list(kernel_size)
        padding = padding.upper()
        in_channels = int(inputs.get_shape()[-1])
        out_channels = filters

        # Initialize parameters
        # W_initializer = tf.orthogonal_initializer()
        W_shape = kernel_size + [in_channels, out_channels]

        dilated_conv_mask = tf.reshape(
            tf.constant(np.array([1., 1., 0.]), name="DilatedConvMask", dtype=tf.float32),
            [1, 3, 1, 1])

        with tf.device('/cpu:0'):
            g_init = np.ones(out_channels)
            b_init = np.zeros(out_channels)

            W_init = tf.random_normal_initializer(0, 0.05)
            W_bias_init = np.ones(W_shape)

            g_mu = tf.get_variable(
                "g_mu", shape=[out_channels], dtype=tf.float32,
                initializer=tf.constant_initializer(g_init),
                trainable=True
            )
            b_mu = tf.get_variable(
                "b_mu", shape=[out_channels], dtype=tf.float32,
                initializer=tf.constant_initializer(b_init),
                trainable=True
            )
            W_mu = tf.get_variable(
                "W_mu", shape=W_shape, dtype=tf.float32,
                initializer=W_init, trainable=True
            )
            g_log_sigma = tf.get_variable(
                "g_log_sigma", shape=[out_channels], dtype=tf.float32,
                initializer=tf.constant_initializer(-7 * g_init),
                trainable=True
            )
            b_log_sigma = tf.get_variable(
                "b_log_sigma", shape=[out_channels], dtype=tf.float32,
                initializer=tf.constant_initializer(-7 * np.ones(out_channels)),
                trainable=True
            )
            W_log_sigma = tf.get_variable(
                "W_log_sigma", shape=W_shape, dtype=tf.float32,
                initializer=tf.constant_initializer(-7 * W_bias_init), trainable=True
            )

        g = _sampler(g_mu, g_log_sigma, step, sampler_hyperparams)
        b = _sampler(b_mu, b_log_sigma, step, sampler_hyperparams)
        W_normed = _sampler(W_mu, W_log_sigma, step, sampler_hyperparams)

        KL_list = []
        KL_list.append(_KLD_standard_normal(g_mu, g_log_sigma))
        KL_list.append(_KLD_standard_normal(b_mu, b_log_sigma))
        KL_list.append(_KLD_standard_normal(W_mu, W_log_sigma))

        # Convolution operation
        W_normed = tf.nn.l2_normalize(W_normed, [0, 1, 2])
        W_normed = W_normed * dilated_conv_mask
        h = tf.nn.convolution(
            inputs, filter=W_normed, strides=strides,
            padding=padding, dilation_rate=dilation_rate
        )
        shape = [1, 1, 1, out_channels]
        h = h * tf.reshape(g, shape) + tf.reshape(b, shape)
        if activation is not None:
            h = activation(h)

    return h, KL_list


def conv2D_bayesian(inputs, filters, step,
                    sampler_hyperparams, kernel_size=(1, 1),
                    padding="same", activation=None,
                    reuse=False, name="conv2D", strides=(1, 1),
                    dilation_rate=(1, 1), use_bias=True, mask=None):
    """ 2D convolution layer with weight normalization """

    with tf.variable_scope(name, reuse=reuse):
        # Ensure argument types
        strides = list(strides)
        dilation_rate = list(dilation_rate)
        kernel_size = list(kernel_size)
        padding = padding.upper()
        in_channels = int(inputs.get_shape()[-1])
        out_channels = filters

        # Initialize parameters
        # W_initializer = tf.orthogonal_initializer()
        W_shape = kernel_size + [in_channels, out_channels]

        dilated_conv_mask = tf.expand_dims(
            tf.expand_dims(
                tf.expand_dims(
                    tf.constant(np.array([1., 1., 0.]), name="DilatedConvMask", dtype=tf.float32),
                    axis=0),
                axis=2),
            axis=3)

        with tf.device('/cpu:0'):
            g_init = np.ones(out_channels)
            b_init = np.zeros(out_channels)

            W_init = tf.random_normal_initializer(0, 0.05)
            W_bias_init = np.ones(W_shape)

            g_mu = tf.get_variable(
                "g_mu", shape=[out_channels], dtype=tf.float32,
                initializer=tf.constant_initializer(g_init),
                trainable=True
            )
            b_mu = tf.get_variable(
                "b_mu", shape=[out_channels], dtype=tf.float32,
                initializer=tf.constant_initializer(b_init),
                trainable=True
            )
            W_mu = tf.get_variable(
                "W_mu", shape=W_shape, dtype=tf.float32,
                initializer=W_init, trainable=True
            )
            g_log_sigma = tf.get_variable(
                "g_log_sigma", shape=[out_channels], dtype=tf.float32,
                initializer=tf.constant_initializer(-7 * g_init),
                trainable=True
            )
            b_log_sigma = tf.get_variable(
                "b_log_sigma", shape=[out_channels], dtype=tf.float32,
                initializer=tf.constant_initializer(-7 * np.ones(out_channels)),
                trainable=True
            )
            W_log_sigma = tf.get_variable(
                "W_log_sigma", shape=W_shape, dtype=tf.float32,
                initializer=tf.constant_initializer(-7 * W_bias_init), trainable=True
            )

        g = _sampler(g_mu, g_log_sigma, step, sampler_hyperparams)
        b = _sampler(b_mu, b_log_sigma, step, sampler_hyperparams)
        W_normed = _sampler(W_mu, W_log_sigma, step, sampler_hyperparams)

        KL_list = [
            _KLD_standard_normal(g_mu, g_log_sigma),
            _KLD_standard_normal(b_mu, b_log_sigma),
            _KLD_standard_normal(W_mu, W_log_sigma)
        ]

        # Convolution operation
        W_normed = tf.nn.l2_normalize(W_normed, [0, 1, 2])

        # W_normed = W_normed
        h = tf.nn.convolution(
            inputs, filter=W_normed, strides=strides,
            padding=padding, dilation_rate=dilation_rate
        )
        shape = [1, 1, 1, out_channels]

        h = h * tf.reshape(g, shape) + tf.reshape(b, shape)
        if activation is not None:
            h = activation(h)

    return h, KL_list


def conv2D_generative(inputs, filters, kernel_size=(1, 1),
                      padding="same", activation=None,
                      reuse=False, name="conv2D", strides=(1, 1),
                      dilation_rate=(1, 1), use_bias=True, mask=None,
                      bias_init=0.
                      ):
    """ 2D convolution layer with weight normalization """

    with tf.variable_scope(name, reuse=reuse):
        # Ensure argument types
        strides = list(strides)
        dilation_rate = list(dilation_rate)
        kernel_size = list(kernel_size)
        padding = padding.upper()
        in_channels = int(inputs.get_shape()[-1])
        out_channels = filters

        # Initialize parameters
        W_initializer = tf.random_normal_initializer(0, 0.05)
        # W_initializer = tf.orthogonal_initializer()
        W_shape = kernel_size + [in_channels, out_channels]

        # g_log_init = 0.5 * np.log(12. / float(in_channels + out_channels)) \
        #     * np.ones((out_channels))

        # should I be putting all these on cpu?
        # with tf.device('/cpu:0'):
        g_init = np.ones(out_channels)
        g = tf.get_variable(
            "g", shape=[out_channels], dtype=tf.float32,
            initializer=tf.constant_initializer(g_init),
            trainable=True
        )
        b = tf.get_variable(
            "b", shape=[out_channels], dtype=tf.float32,
            initializer=tf.constant_initializer(bias_init * np.ones(out_channels)),
            trainable=True
        )
        W = tf.get_variable(
            "W", shape=W_shape, dtype=tf.float32,
            initializer=W_initializer, trainable=True
        )

        weight_list = [tf.reduce_sum(tf.square(g)),
                       tf.reduce_sum(tf.square(b)),
                       tf.reduce_sum(tf.square(W)),
                       ]

        dilated_conv_mask = tf.reshape(
            tf.constant(np.array([1., 1., 0.]), name="DilatedConvMask", dtype=tf.float32),
            [1, 3, 1, 1])

        # Convolution operation
        W_normed = tf.nn.l2_normalize(W, [0, 1, 2])

        W_normed = W_normed * dilated_conv_mask

        h = tf.nn.convolution(
            inputs, filter=W_normed, strides=strides,
            padding=padding, dilation_rate=dilation_rate
        )
        shape = [1, 1, 1, out_channels]
        h = h * tf.reshape(g, shape) + tf.reshape(b, shape)
        if activation is not None:
            h = activation(h)

    return h, weight_list


def conv2D(inputs, filters, kernel_size=(1, 1),
           padding="same", activation=None,
           reuse=False, name="conv2D", strides=(1, 1),
           dilation_rate=(1, 1), use_bias=True, mask=None,
           bias_init=0.1, g_init=1.0
           ):
    """ 2D convolution layer with weight normalization """

    with tf.variable_scope(name, reuse=reuse):
        # Ensure argument types
        strides = list(strides)
        dilation_rate = list(dilation_rate)
        kernel_size = list(kernel_size)
        padding = padding.upper()
        in_channels = int(inputs.get_shape()[-1])
        out_channels = filters

        # Initialize parameters
        W_initializer = tf.random_normal_initializer(0, 0.05)
        # W_initializer = tf.orthogonal_initializer()
        W_shape = kernel_size + [in_channels, out_channels]

        # g_log_init = 0.5 * np.log(12. / float(in_channels + out_channels)) \
        #     * np.ones((out_channels))

        # should I be putting all these on cpu?
        # with tf.device('/cpu:0'):
        g = tf.get_variable(
            "g", shape=[out_channels], dtype=tf.float32,
            initializer=tf.constant_initializer(g_init * np.ones(out_channels)),
            trainable=True
        )
        b = tf.get_variable(
            "b", shape=[out_channels], dtype=tf.float32,
            initializer=tf.constant_initializer(bias_init * np.ones(out_channels)),
            trainable=True
        )
        W = tf.get_variable(
            "W", shape=W_shape, dtype=tf.float32,
            initializer=W_initializer, trainable=True
        )

        weight_list = [tf.reduce_sum(tf.square(g)),
                       tf.reduce_sum(tf.square(b)),
                       tf.reduce_sum(tf.square(W)),
                       ]

        # Convolution operation
        W_normed = tf.nn.l2_normalize(W, [0, 1, 2])
        h = tf.nn.convolution(
            inputs, filter=W_normed, strides=strides,
            padding=padding, dilation_rate=dilation_rate
        )
        shape = [1, 1, 1, out_channels]
        h = h * tf.reshape(g, shape) + tf.reshape(b, shape)
        if activation is not None:
            h = activation(h)

    return h, weight_list


def conv2D_transpose(inputs, filters, kernel_size=(1, 1),
                     padding="same", activation=None,
                     reuse=False, name="conv2D", strides=(1, 1),
                     dilation_rate=(1, 1), use_bias=True, mask=None
                     ):
    """ 2D convolution layer with weight normalization """

    with tf.variable_scope(name, reuse=reuse):
        # Ensure argument types
        strides = list(strides)
        dilation_rate = list(dilation_rate)
        kernel_size = list(kernel_size)
        padding = padding.upper()
        in_channels = int(inputs.get_shape()[-1])
        out_channels = filters

        # Initialize parameters
        W_initializer = tf.random_normal_initializer(0, 0.05)
        # W_initializer = tf.orthogonal_initializer()
        W_shape = kernel_size + [out_channels, in_channels]
        W = tf.get_variable(
            "W", shape=W_shape, dtype=tf.float32,
            initializer=W_initializer, trainable=True
        )
        # Independent initialization
        # g_log_init = 0.5 * np.log(12. / float(in_channels + out_channels)) \
        #     * np.ones((out_channels))
        with tf.device('/cpu:0'):
            g_init = np.ones(out_channels)
            g = tf.get_variable(
                "g", shape=[out_channels], dtype=tf.float32,
                initializer=tf.constant_initializer(g_init),
                trainable=True
            )
            b = tf.get_variable(
                "b", shape=[out_channels], dtype=tf.float32,
                initializer=tf.constant_initializer(np.zeros(out_channels)),
                trainable=True
            )

        # Data-dependent initialization
        # W_normed_init = tf.nn.l2_normalize(W.initialized_value(), [0, 1, 2])
        # h_init = tf.nn.conv2d(
        #     inputs, filter=W_normed_init, strides=[1]+strides+[1],
        #     padding=padding
        # )
        # h_means, h_vars = tf.nn.moments(h_init, [0, 1, 2])
        # g_init = 1. / tf.sqrt(h_vars + 1e-5)
        # b_init = -h_means * g_init
        # g = tf.get_variable(
        #     "g", dtype=tf.float32,
        #     initializer=g_init, trainable=True
        # )
        # b = tf.get_variable(
        #     "b", dtype=tf.float32,
        #     initializer=b_init, trainable=True
        # )

        # Convolution operation
        W_normed = tf.nn.l2_normalize(W, [0, 1, 2])
        in_shape = tf.shape(inputs)
        output_shape = tf.stack(
            [in_shape[0], in_shape[1], in_shape[2], out_channels])
        h = tf.nn.conv2d_transpose(
            inputs, output_shape=output_shape,
            filter=W_normed, strides=[1] + strides + [1],
            padding=padding
        )
        shape = [1, 1, 1, out_channels]
        h = h * tf.reshape(g, shape) + tf.reshape(b, shape)
        if activation is not None:
            h = activation(h)
    return h


def convnet_1D(inputs, inner_channels, mask, widths, dilations,
               dropout_p=0.5, reuse=None, transpose=False,
               additional_input=None, additional_layer_input=None):
    """ Residual dilated 1D conv stack. """

    with tf.variable_scope("ConvNet1D"):
        up_layer = inputs

        if additional_layer_input is not None:
            up_layer_normed = normalize_layer(
                tf.concat([up_layer, additional_layer_input], axis=3),
                mask, per_channel=False
            )

        for i, (width, dilation) in enumerate(zip(widths, dilations)):
            name = "Conv" + str(i) + "_" + str(width) + "x" + str(dilation)
            if transpose:
                name += "_Trans"
                f = conv2D_transpose
            else:
                f = conv2D
            with tf.variable_scope(name):

                if additional_input is not None:
                    up_layer_normed = normalize_layer(
                        tf.concat([up_layer, additional_input], axis=3),
                        mask, per_channel=False
                    )
                elif additional_layer_input is not None and i == 0:
                    up_layer_normed = up_layer_normed

                else:
                    up_layer_normed = normalize_layer(
                        up_layer, mask, per_channel=False
                    )

                up_layer_normed = scale_shift(up_layer_normed, mask, up_layer_normed.get_shape().as_list()[-1])

                delta_layer = conv2D(up_layer_normed,
                                     filters=inner_channels,
                                     kernel_size=(1, 1),
                                     padding="same",
                                     activation=nonlinearity,
                                     reuse=reuse,
                                     name="Mix1" + str(i))
                conv_dict = {
                    "inputs": delta_layer,
                    "filters": inner_channels,
                    "kernel_size": (1, width),
                    "padding": "same",
                    "activation": nonlinearity,
                    "reuse": reuse,
                    "name": "Conv2" + str(i)
                }
                if dilation is not 1:
                    conv_dict["dilation_rate"] = (1, dilation)
                delta_layer = f(**conv_dict)
                delta_layer = conv2D(delta_layer,
                                     filters=inner_channels,
                                     kernel_size=(1, 1),
                                     padding="same",
                                     activation=nonlinearity,
                                     reuse=reuse,
                                     name="Mix3" + str(i))
                delta_layer = tf.nn.dropout(delta_layer, dropout_p)
                delta_layer = normalize_layer(
                    delta_layer, mask, per_channel=False
                )
                with tf.variable_scope("ScaleShiftDeltaLayer"):
                    delta_layer = scale_shift(delta_layer, mask, inner_channels)

                up_layer = up_layer + delta_layer

    return up_layer


def convnet_1D_standard(inputs, inner_channels, mask, widths, dilations,
                        dropout_p=0.5, additional_input=None, additional_layer_input=None,
                        reuse=None, transpose=False, nonlinearity=tf.nn.relu):
    """ Residual dilated 1D conv stack. """

    weight_cost_list = []
    with tf.variable_scope("ConvNet1D"):
        up_layer = inputs

        if additional_layer_input is not None:
            up_layer_normed = normalize_layer(
                tf.concat([up_layer, additional_layer_input], axis=3),
                mask,
            )

        for i, (width, dilation) in enumerate(zip(widths, dilations)):
            name = "Conv" + str(i) + "_" + str(width) + "x" + str(dilation)
            if transpose:
                name += "_Trans"
                f = conv2D_transpose
            else:
                f = conv2D
            with tf.variable_scope(name):

                if additional_input is not None:
                    up_layer_normed = normalize_layer(
                        tf.concat([up_layer, additional_input], axis=3),
                        mask, per_channel=False
                    )
                elif additional_layer_input is not None and i == 0:
                    up_layer_normed = up_layer_normed

                else:
                    up_layer_normed = normalize_layer(
                        up_layer, mask, per_channel=False
                    )

                up_layer_normed = scale_shift(up_layer_normed,
                                              mask, up_layer_normed.get_shape().as_list()[-1], name="ScaleShiftOne")

                delta_layer = nonlinearity(up_layer_normed)

                delta_layer, weight_cost = conv2D(delta_layer, inner_channels,
                                                  kernel_size=(1, 1),
                                                  padding="same",
                                                  activation=None,
                                                  reuse=reuse,
                                                  name="Mix1" + str(i))

                weight_cost_list += weight_cost

                delta_layer = normalize_layer(
                    delta_layer, mask, per_channel=False
                )

                delta_layer = nonlinearity(scale_shift(delta_layer,
                                                       mask, inner_channels, name="ScaleShiftTwo"))

                delta_layer, weight_cost = conv2D(delta_layer, inner_channels,
                                                  kernel_size=(1, width),
                                                  padding="same",
                                                  activation=None,
                                                  reuse=reuse,
                                                  dilation_rate=(1, dilation),
                                                  name="DilatedConvGen" + str(i))

                weight_cost_list += weight_cost

                delta_layer = normalize_layer(
                    delta_layer, mask, per_channel=False
                )

                delta_layer = nonlinearity(scale_shift(delta_layer,
                                                       mask, inner_channels, name="ScaleShiftThree"))

                delta_layer, weight_cost = conv2D(delta_layer, inner_channels,
                                                  kernel_size=(1, 1),
                                                  padding="same",
                                                  activation=None,
                                                  reuse=reuse,
                                                  name="Mix3" + str(i))

                weight_cost_list += weight_cost

                delta_layer = tf.nn.dropout(delta_layer, dropout_p)

                up_layer = up_layer + delta_layer

    return up_layer, weight_cost_list


def convnet_1D_generative_standard(inputs, inner_channels, mask, widths, dilations,
                                   dropout_p=0.5, additional_input=None, additional_layer_input=None,
                                   reuse=None, transpose=False, nonlinearity=tf.nn.relu, dropout_type="inter"):
    """ Residual dilated 1D conv stack. """

    weight_cost_list = []
    with tf.variable_scope("ConvNet1D"):
        up_layer = inputs

        if additional_layer_input is not None:
            up_layer_normed = normalize_layer(
                tf.concat([up_layer, additional_layer_input], axis=3),
                mask,
            )

        for i, (width, dilation) in enumerate(zip(widths, dilations)):
            name = "Conv" + str(i) + "_" + str(width) + "x" + str(dilation)
            if transpose:
                name += "_Trans"
                f = conv2D_transpose
            else:
                f = conv2D
            with tf.variable_scope(name):

                if additional_input is not None:
                    up_layer_normed = normalize_layer(
                        tf.concat([up_layer, additional_input], axis=3),
                        mask, per_channel=False
                    )
                elif additional_layer_input is not None and i == 0:
                    up_layer_normed = up_layer_normed

                else:
                    up_layer_normed = normalize_layer(
                        up_layer, mask, per_channel=False
                    )

                up_layer_normed = scale_shift(up_layer_normed,
                                              mask, up_layer_normed.get_shape().as_list()[-1], name="ScaleShiftOne")

                delta_layer = nonlinearity(up_layer_normed)

                delta_layer, weight_cost = conv2D(delta_layer, inner_channels,
                                                  kernel_size=(1, 1),
                                                  padding="same",
                                                  activation=None,
                                                  reuse=reuse,
                                                  name="Mix1" + str(i))

                weight_cost_list += weight_cost

                delta_layer = normalize_layer(
                    delta_layer, mask, per_channel=False
                )

                delta_layer = nonlinearity(scale_shift(delta_layer,
                                                       mask, inner_channels, name="ScaleShiftTwo"))

                delta_layer, weight_cost = conv2D_generative(delta_layer, inner_channels,
                                                             kernel_size=(1, width),
                                                             padding="same",
                                                             activation=None,
                                                             reuse=reuse,
                                                             dilation_rate=(1, dilation),
                                                             name="DilatedConvGen" + str(i))

                weight_cost_list += weight_cost

                delta_layer = normalize_layer(
                    delta_layer, mask, per_channel=False
                )

                delta_layer = nonlinearity(scale_shift(delta_layer,
                                                       mask, inner_channels, name="ScaleShiftThree"))

                delta_layer, weight_cost = conv2D(delta_layer, inner_channels,
                                                  kernel_size=(1, 1),
                                                  padding="same",
                                                  activation=None,
                                                  reuse=reuse,
                                                  name="Mix3" + str(i))

                weight_cost_list += weight_cost

                if dropout_type == 'inter' or dropout_type == 'final':
                    delta_layer = tf.nn.dropout(delta_layer, dropout_p)

                elif dropout_type == 'gaussian':
                    shape = tf.shape(delta_layer)
                    eps = tf.random_normal(shape, stddev=dropout_p)
                    delta_layer = delta_layer + delta_layer * eps

                up_layer = up_layer + delta_layer

    return up_layer, weight_cost_list


def convnet_1D_generative_bayesian_standard(inputs, inner_channels, mask, widths, dilations,
                                            step, sampler_hyperparams, additional_input=None, reuse=None,
                                            transpose=False,
                                            nonlinearity=tf.nn.relu):
    """ Residual dilated 1D conv stack. """

    KL_list = []
    with tf.variable_scope("ConvNet1D"):
        up_layer = inputs
        for i, (width, dilation) in enumerate(zip(widths, dilations)):
            name = "Conv" + str(i) + "_" + str(width) + "x" + str(dilation)
            with tf.variable_scope(name):

                if additional_input is not None:
                    up_layer_normed = normalize_layer(
                        tf.concat([up_layer, additional_input], axis=3),
                        mask, per_channel=False
                    )
                else:
                    up_layer_normed = normalize_layer(
                        up_layer, mask, per_channel=False
                    )

                up_layer_normed = scale_shift(up_layer_normed, mask, inner_channels, name="ScaleShiftOne")

                delta_layer, layer_KL = conv2D_bayesian(up_layer_normed, inner_channels,
                                                        step, sampler_hyperparams,
                                                        kernel_size=(1, 1),
                                                        padding="same",
                                                        activation=None,
                                                        reuse=reuse,
                                                        name="Mix1" + str(i))
                KL_list += layer_KL

                delta_layer = normalize_layer(
                    delta_layer, mask, per_channel=False
                )

                delta_layer = nonlinearity(scale_shift(delta_layer,
                                                       mask, inner_channels, name="ScaleShiftTwo"))

                delta_layer, layer_KL = conv2D_generative_bayesian(delta_layer, inner_channels,
                                                                   step, sampler_hyperparams,
                                                                   kernel_size=(1, width),
                                                                   padding="same",
                                                                   activation=None,
                                                                   reuse=reuse,
                                                                   dilation_rate=(1, dilation),
                                                                   name="DilatedConvGen" + str(i))
                KL_list += layer_KL

                delta_layer = normalize_layer(
                    delta_layer, mask, per_channel=False
                )

                delta_layer = nonlinearity(scale_shift(delta_layer,
                                                       mask, inner_channels, name="ScaleShiftThree"))

                delta_layer, layer_KL = conv2D_bayesian(delta_layer, inner_channels,
                                                        step, sampler_hyperparams,
                                                        kernel_size=(1, 1),
                                                        padding="same",
                                                        activation=None,
                                                        reuse=reuse,
                                                        name="Mix3" + str(i))
                KL_list += layer_KL

                up_layer = up_layer + delta_layer

    return up_layer, KL_list


def convnet_1D_bayesian(inputs, inner_channels, mask, widths, dilations,
                        step, sampler_hyperparams, additional_input=None, reuse=None, transpose=False,
                        nonlinearity=tf.nn.elu):
    """ Residual dilated 1D conv stack. """

    KL_list = []
    with tf.variable_scope("ConvNet1D"):
        up_layer = inputs
        for i, (width, dilation) in enumerate(zip(widths, dilations)):
            name = "Conv" + str(i) + "_" + str(width) + "x" + str(dilation)
            with tf.variable_scope(name):

                if additional_input is not None:
                    up_layer_normed = normalize_layer(
                        tf.concat([up_layer, additional_input], axis=3),
                        mask, per_channel=False
                    )
                else:
                    up_layer_normed = normalize_layer(
                        up_layer, mask, per_channel=False
                    )
                delta_layer, layer_KL = conv2D_bayesian(up_layer_normed, inner_channels,
                                                        step, sampler_hyperparams,
                                                        kernel_size=(1, 1),
                                                        padding="same",
                                                        activation=nonlinearity,
                                                        reuse=reuse,
                                                        name="Mix1" + str(i))
                KL_list += layer_KL

                delta_layer, layer_KL = conv2D_bayesian(delta_layer, inner_channels,
                                                        step, sampler_hyperparams,
                                                        kernel_size=(1, width),
                                                        padding="same",
                                                        activation=nonlinearity,
                                                        reuse=reuse,
                                                        dilation_rate=(1, dilation),
                                                        name="DilatedConvGen" + str(i))
                KL_list += layer_KL

                delta_layer, layer_KL = conv2D_bayesian(delta_layer, inner_channels,
                                                        step, sampler_hyperparams,
                                                        kernel_size=(1, 1),
                                                        padding="same",
                                                        activation=nonlinearity,
                                                        reuse=reuse,
                                                        name="Mix3" + str(i))
                KL_list += layer_KL

                delta_layer = normalize_layer(
                    delta_layer, mask, per_channel=False
                )
                up_layer = up_layer + delta_layer

    return up_layer, KL_list


def convnet_1D_generative(inputs, inner_channels, mask, widths, dilations,
                          dropout_p=0.5, additional_input=None, additional_layer_input=None,
                          reuse=None, transpose=False, nonlinearity=tf.nn.elu):
    """ Residual dilated 1D conv stack. """

    weight_cost_list = []
    with tf.variable_scope("ConvNet1D"):
        up_layer = inputs

        if additional_layer_input is not None:
            up_layer_normed = normalize_layer(
                tf.concat([up_layer, additional_layer_input], axis=3),
                mask,
            )

        for i, (width, dilation) in enumerate(zip(widths, dilations)):
            name = "Conv" + str(i) + "_" + str(width) + "x" + str(dilation)
            if transpose:
                name += "_Trans"
                f = conv2D_transpose
            else:
                f = conv2D
            with tf.variable_scope(name):

                if additional_input is not None:
                    up_layer_normed = normalize_layer(
                        tf.concat([up_layer, additional_input], axis=3),
                        mask, per_channel=False
                    )
                elif additional_layer_input is not None and i == 0:
                    up_layer_normed = up_layer_normed

                else:
                    up_layer_normed = normalize_layer(
                        up_layer, mask, per_channel=False
                    )

                up_layer_normed = scale_shift(up_layer_normed, mask, up_layer_normed.get_shape().as_list()[-1])

                delta_layer, weight_cost = conv2D(up_layer_normed, inner_channels,
                                                  kernel_size=(1, 1),
                                                  padding="same",
                                                  activation=nonlinearity,
                                                  reuse=reuse,
                                                  name="Mix1" + str(i))

                weight_cost_list += weight_cost

                delta_layer, weight_cost = conv2D_generative(delta_layer, inner_channels,
                                                             kernel_size=(1, width),
                                                             padding="same",
                                                             activation=nonlinearity,
                                                             reuse=reuse,
                                                             dilation_rate=(1, dilation),
                                                             name="DilatedConvGen" + str(i))

                weight_cost_list += weight_cost

                delta_layer, weight_cost = conv2D(delta_layer, inner_channels,
                                                  kernel_size=(1, 1),
                                                  padding="same",
                                                  activation=nonlinearity,
                                                  reuse=reuse,
                                                  name="Mix3" + str(i))

                weight_cost_list += weight_cost

                delta_layer = tf.nn.dropout(delta_layer, dropout_p)

                delta_layer = normalize_layer(
                    delta_layer, mask,
                )

                with tf.variable_scope("ScaleShiftDeltaLayer"):

                    delta_layer = scale_shift(delta_layer, mask, inner_channels)

                up_layer = up_layer + delta_layer

    return up_layer, weight_cost_list


def convnet_2D(inputs, inner_channels, mask, widths, dilations,
               dropout_p=0.5, reuse=None, transpose=False):
    """ Residual dilated 2D conv stack. """

    with tf.variable_scope("ConvNet2D"):
        up_layer = inputs
        for i, (width, dilation) in enumerate(zip(widths, dilations)):
            name = "Conv" + str(i) + "_" + str(width) + "x" + str(dilation)
            if transpose:
                name += "_Trans"
                f = conv2D_transpose
            else:
                f = conv2D
            with tf.variable_scope(name):
                up_layer_normed = normalize_layer(
                    up_layer, mask, per_channel=False
                )
                delta_layer = conv2D(up_layer_normed,
                                     filters=inner_channels,
                                     kernel_size=(1, 1),
                                     padding="same",
                                     activation=nonlinearity,
                                     reuse=reuse,
                                     name="Mix1")
                conv_dict = {
                    "inputs": delta_layer,
                    "filters": inner_channels,
                    "kernel_size": (width, width),
                    "padding": "same",
                    "activation": nonlinearity,
                    "reuse": reuse,
                    "name": "Conv2"
                }
                if dilation is not 1:
                    conv_dict["dilation_rate"] = (dilation, dilation)
                delta_layer = f(**conv_dict)
                delta_layer = conv2D(delta_layer,
                                     filters=inner_channels,
                                     kernel_size=(1, 1),
                                     padding="same",
                                     activation=nonlinearity,
                                     reuse=reuse,
                                     name="Mix3")
                delta_layer = tf.nn.dropout(delta_layer, dropout_p)
                delta_layer = normalize_layer(
                    delta_layer, mask, per_channel=False
                )
                up_layer = up_layer + delta_layer
    return up_layer


def expand_1Dto2D(input_1D, channels_1D, channels_2D, mask_1D, mask_2D,
                  dropout_p=0.5, extra_input_2D=[], reuse=None):
    """ Gated expansion of 1D conv stack to a 2D conv stack"""

    with tf.variable_scope("1Dto2D"):
        input_1D = normalize_layer(
            input_1D, mask_1D, per_channel=False
        )
        # B,1,L,Channels_1D
        contribution_i = conv2D(input_1D,
                                filters=channels_2D,
                                kernel_size=(1, 1),
                                padding="same",
                                use_bias=False,
                                activation=None,
                                reuse=reuse,
                                name="iMix1")
        contribution_j = conv2D(input_1D,
                                filters=channels_2D,
                                kernel_size=(1, 1),
                                padding="same",
                                use_bias=False,
                                activation=None,
                                reuse=reuse,
                                name="jMix2")
        contribution_i *= mask_1D
        contribution_j *= mask_1D
        # 2x[B,1,L,K] => [B,K,1,L] + [B,K,L,1]  => [B,K,L,L]
        ij_sum = tf.transpose(contribution_i, [0, 3, 1, 2]) \
                 + tf.transpose(contribution_j, [0, 3, 2, 1])
        # [B,K,L,L] => [B,L,L,K]
        ij_input = tf.transpose(ij_sum, [0, 2, 3, 1])

        # Concatenate extra input if available
        if len(extra_input_2D) > 0:
            input_set = extra_input_2D + [ij_input]
            extra_input_2D = tf.concat(axis=3, values=extra_input_2D)
            extra_input_2D = normalize_layer(
                extra_input_2D, mask_2D, per_channel=False
            )
            ij_input = tf.concat(
                axis=3, values=[extra_input_2D, ij_input]
            )

        ij_input = normalize_layer(
            ij_input, mask_2D, per_channel=False
        )

        activation1 = conv2D(ij_input,
                             filters=channels_2D,
                             kernel_size=(1, 1),
                             padding="same",
                             activation=nonlinearity,
                             reuse=reuse,
                             name="ConcatMix3")
        activation2 = conv2D(activation1,
                             filters=channels_2D,
                             kernel_size=(1, 1),
                             padding="same",
                             activation=nonlinearity,
                             reuse=reuse,
                             name="Mix4")
        activation2 = normalize_layer(
            activation2, mask_2D, per_channel=False
        )
        activation2 = tf.nn.dropout(activation2, dropout_p)
        gate_layer = conv2D(ij_input,
                            filters=channels_2D,
                            kernel_size=(1, 1),
                            padding="same",
                            activation=tf.nn.sigmoid,
                            reuse=reuse,
                            name="Gate5")
        out_layer = gate_layer * activation2
    return out_layer


def reduce_2Dto1D(input_1D, input_2D, channels_2D, channels_1D, mask_1D, mask_2D,
                  reuse=None):
    """ Gated reduction of 2D conv stack. to a 1D conv stack """

    with tf.variable_scope("2Dto1D"):
        input_1D = normalize_layer(
            input_1D, mask_1D, per_channel=False
        )
        input_2D = normalize_layer(
            input_2D, mask_2D, per_channel=False
        )
        # Channel mixing
        delta_layer = conv2D(input_2D,
                             filters=channels_2D,
                             kernel_size=(1, 1),
                             padding="same",
                             activation=nonlinearity,
                             reuse=reuse,
                             name="Mix1")
        # Attention query
        query = conv2D(input_1D,
                       filters=channels_2D + 1,
                       kernel_size=(1, 1),
                       padding="same",
                       activation=None,
                       reuse=reuse,
                       name="Query2")
        # Produce both temperature and address
        beta = tf.expand_dims(tf.exp(query[:, :, :, -1]), 3)
        query = query[:, :, :, :channels_2D]
        # Masked softmax over the edges
        attention = tf.reduce_sum(
            beta * tf.nn.l2_normalize(input_2D, 3) * tf.nn.l2_normalize(query, 3),
            axis=3, keep_dims=True
        )
        attention_max = tf.reduce_max(attention, axis=1, keep_dims=True)
        attention_weights = mask_2D * tf.exp(attention - attention_max)
        Z = tf.reduce_sum(attention_weights, axis=1, keep_dims=True)
        attention_weights = attention_weights / (Z + 1E-3)
        # Attention-weighted average of the edges
        delta_layer = attention_weights * delta_layer
        delta_layer = tf.reduce_sum(delta_layer, axis=1, keep_dims=True)
        # Channel mixing
        in_layer = tf.concat(axis=3, values=[input_1D, delta_layer])
        delta_layer = normalize_layer(
            delta_layer, mask_1D, per_channel=False
        )
        out_layer = conv2D(delta_layer,
                           filters=channels_1D,
                           kernel_size=(1, 1),
                           padding="same",
                           activation=nonlinearity,
                           reuse=reuse,
                           name="Mix3")
    return out_layer


def expand_1Dto2D_generative(input_1D, channels_1D, channels_2D, mask_1D, mask_2D,
                             mask_tri, dropout_p=0.5, extra_input_2D=[], reuse=None):
    """ Gated expansion of 1D conv stack to a 2D conv stack"""

    with tf.variable_scope("1Dto2D"):
        input_1D = normalize_layer(
            input_1D, mask_1D, per_channel=False
        )
        # B,1,L,Channels_1D
        contribution_i = conv2D(input_1D,
                                filters=channels_2D,
                                kernel_size=(1, 1),
                                padding="same",
                                use_bias=False,
                                activation=None,
                                reuse=reuse,
                                name="iMix1")
        contribution_j = conv2D(input_1D,
                                filters=channels_2D,
                                kernel_size=(1, 1),
                                padding="same",
                                use_bias=False,
                                activation=None,
                                reuse=reuse,
                                name="jMix2")
        contribution_i *= mask_1D
        contribution_j *= mask_1D
        # 2x[B,1,L,K] => [B,K,1,L] + [B,K,L,1]  => [B,K,L,L]
        ij_sum = tf.transpose(contribution_i, [0, 3, 1, 2]) \
                 + tf.transpose(contribution_j, [0, 3, 2, 1])
        # [B,K,L,L] => [B,L,L,K]
        ij_input = tf.transpose(ij_sum, [0, 2, 3, 1])

        ij_input = ij_input * mask_tri

        # Concatenate extra input if available
        if len(extra_input_2D) > 0:
            input_set = extra_input_2D + [ij_input]
            extra_input_2D = tf.concat(axis=3, values=extra_input_2D)
            extra_input_2D = normalize_layer(
                extra_input_2D, mask_tri, per_channel=False
            )
            ij_input = tf.concat(
                axis=3, values=[extra_input_2D, ij_input]
            )

        ij_input = normalize_layer(
            ij_input, mask_tri
        )

        activation1 = conv2D(ij_input,
                             filters=channels_2D,
                             kernel_size=(1, 1),
                             padding="same",
                             activation=nonlinearity,
                             reuse=reuse,
                             name="ConcatMix3")
        activation2 = conv2D(activation1,
                             filters=channels_2D,
                             kernel_size=(1, 1),
                             padding="same",
                             activation=nonlinearity,
                             reuse=reuse,
                             name="Mix4")

        activation2 = normalize_layer(
            activation2, mask_tri,
        )
        activation2 = tf.nn.dropout(activation2, dropout_p)
        gate_layer = conv2D(ij_input,
                            filters=channels_2D,
                            kernel_size=(1, 1),
                            padding="same",
                            activation=tf.nn.sigmoid,
                            reuse=reuse,
                            name="Gate5")
        out_layer = gate_layer * activation2 * mask_tri
    return out_layer


def reduce_2Dto1D_generative(input_1D, input_2D, channels_2D, channels_1D,
                             mask_1D, mask_2D, mask_tri, reuse=None):
    """ Gated reduction of 2D conv stack. to a 1D conv stack """

    with tf.variable_scope("2Dto1D"):
        input_1D = normalize_layer(
            input_1D, mask_1D, per_channel=False
        )
        input_2D = normalize_layer(
            input_2D, mask_tri
        )

        input_2D = input_2D * mask_tri

        # Channel mixing
        delta_layer = conv2D(input_2D,
                             filters=channels_2D,
                             kernel_size=(1, 1),
                             padding="same",
                             activation=nonlinearity,
                             reuse=reuse,
                             name="Mix1")

        # Attention query
        query = conv2D(input_1D,
                       filters=channels_2D + 1,
                       kernel_size=(1, 1),
                       padding="same",
                       activation=None,
                       reuse=reuse,
                       name="Query2")
        # Produce both temperature and address
        beta = tf.expand_dims(tf.exp(query[:, :, :, -1]), 3)
        query = query[:, :, :, :channels_2D]
        # Masked softmax over the edges
        attention = tf.reduce_sum(
            beta * tf.nn.l2_normalize(input_2D, 3) * tf.nn.l2_normalize(query, 3),
            axis=3, keep_dims=True
        )
        attention_max = tf.reduce_max(attention, axis=1, keep_dims=True)
        attention_weights = mask_tri * tf.exp(attention - attention_max)
        Z = tf.reduce_sum(attention_weights, axis=1, keep_dims=True)
        attention_weights = attention_weights / (Z + 1E-3)
        # Attention-weighted average of the edges
        delta_layer = attention_weights * delta_layer
        delta_layer = tf.reduce_sum(delta_layer, axis=1, keep_dims=True)
        # Channel mixing
        in_layer = tf.concat(axis=3, values=[input_1D, delta_layer])
        delta_layer = normalize_layer(
            delta_layer, mask_1D, per_channel=False
        )
        out_layer = conv2D(delta_layer,
                           filters=channels_1D,
                           kernel_size=(1, 1),
                           padding="same",
                           activation=nonlinearity,
                           reuse=reuse,
                           name="Mix3")
    return out_layer


def attention_1D_generative(feature_input_1D, input_1D, channels_1D, mask_1D, mask_tri, attention_type='softmax',
                            reuse=None, nonlinearity=tf.nn.elu, num_heads=1):
    """ Gated reduction of 2D conv stack. to a 1D conv stack """

    with tf.variable_scope("Attention"):
        input_1D_normed = normalize_layer(
            input_1D, mask_1D, per_channel=False
        )

        source_one = conv2D(feature_input_1D,
                            filters=channels_1D,
                            kernel_size=(1, 1),
                            padding="same",
                            activation=nonlinearity,
                            reuse=reuse,
                            name="Source1")

        source = conv2D(source_one,
                        filters=channels_1D,
                        kernel_size=(1, 1),
                        padding="same",
                        activation=None,
                        reuse=reuse,
                        name="Source")

        source = tf.transpose(source, perm=[0, 2, 1, 3])

        # Attention query
        query_one = conv2D(input_1D_normed,
                           filters=channels_1D,
                           kernel_size=(1, 1),
                           padding="same",
                           activation=nonlinearity,
                           reuse=reuse,
                           name="Query1")

        query = conv2D(query_one,
                       filters=channels_1D,
                       kernel_size=(1, 1),
                       padding="same",
                       activation=None,
                       reuse=reuse,
                       name="Query2")

        # Masked softmax over the edges
        attention = tf.reduce_sum(
            tf.nn.l2_normalize(source, 3) * tf.nn.l2_normalize(query, 3),
            axis=3, keep_dims=True
        )

        if attention_type == 'sigmoid':
            attention = tf.nn.sigmoid(attention) * mask_tri

            with tf.variable_scope("1_Attention"):
                tf.summary.image("1_Attention", attention, 3)

            attended_ij_input = source * attention

            delta_layer = tf.reduce_sum(attended_ij_input, axis=1, keep_dims=True)

            delta_layer = normalize_layer(
                delta_layer, mask_1D, per_channel=False
            )
            with tf.variable_scope("2_NormalizedSummedVals"):
                tf.summary.image("2_NormalizedSummedVals",
                                 tf.transpose(delta_layer, perm=[0, 3, 2, 1]), 3)

            # delta_layer = tf.concat(axis=3, values=[input_1D_normed, delta_layer])

        else:

            attention_max = tf.reduce_max(attention, axis=1, keep_dims=True)
            attention_weights = mask_tri * tf.exp(attention - attention_max)
            Z = tf.reduce_sum(attention_weights, axis=1, keep_dims=True)
            attention = attention_weights / (Z + 1E-3)

            with tf.variable_scope("1_Attention"):
                tf.summary.image("1_Attention", attention, 3)
            # Attention-weighted average of the edges
            attended_ij_input = attention * source
            delta_layer = tf.reduce_sum(attended_ij_input, axis=1, keep_dims=True)
            delta_layer = normalize_layer(
                delta_layer, mask_1D, per_channel=False
            )

            with tf.variable_scope("2_NormalizedSummedVals"):
                tf.summary.image("2_NormalizedSummedVals",
                                 tf.transpose(delta_layer, perm=[0, 3, 2, 1]), 3)

            # delta_layer = tf.concat(axis=3, values=[input_1D_normed, delta_layer])

        out_layer = conv2D(delta_layer,
                           filters=channels_1D,
                           kernel_size=(1, 1),
                           padding="same",
                           activation=None,
                           reuse=reuse,
                           name="Mix3")
        out_layer = out_layer * mask_1D

        with tf.variable_scope("3_AfterAttnState"):
            tf.summary.image("3_AfterAttnState",
                             tf.transpose(out_layer, perm=[0, 3, 2, 1]), 3)

    return input_1D + out_layer


def multihead_attention_1D_generative(input_1D, channels_1D,
                                      mask_1D, mask_2D, mask_tri, attention_type='softmax', reuse=None,
                                      nonlinearity=tf.nn.elu, num_heads=4, dropout_p=0.5):
    with tf.variable_scope("Attention"):
        input_1D_normed = normalize_layer(
            input_1D, mask_1D, per_channel=False
        )

        input_1D_normed = scale_shift(input_1D_normed, mask_1D, channels_1D,
                                      name="InputScaleShift")

        source_one = conv2D(input_1D_normed, channels_1D,
                            kernel_size=(1, 1),
                            padding="same",
                            activation=nonlinearity,
                            reuse=reuse,
                            name="Source1")

        source = conv2D(source_one, channels_1D,
                        kernel_size=(1, 1),
                        padding="same",
                        activation=None,
                        reuse=reuse,
                        name="Source")

        source = tf.transpose(source, perm=[0, 2, 1, 3])

        # Attention query
        query_one = conv2D(input_1D_normed, channels_1D,
                           kernel_size=(1, 1),
                           padding="same",
                           activation=nonlinearity,
                           reuse=reuse,
                           name="Query1")

        query = conv2D(query_one, channels_1D,
                       kernel_size=(1, 1),
                       padding="same",
                       activation=None,
                       reuse=reuse,
                       name="Query2")

        batch_size = tf.shape(input_1D)[0]
        length = tf.shape(input_1D)[2]
        source = tf.reshape(source, (batch_size, length, 1, num_heads, channels_1D / num_heads))

        query = tf.reshape(query, (batch_size, 1, length, num_heads, channels_1D / num_heads))

        # Masked softmax over the edges
        attention = tf.reduce_sum(
            tf.nn.l2_normalize(source, 4) * tf.nn.l2_normalize(query, 4),
            axis=4, keep_dims=True
        )

        ####attention_max = tf.reduce_max(attention, axis=1, keep_dims=True)
        ###attention_weights = tf.expand_dims(mask_tri,axis=-1) * tf.exp(attention - attention_max)
        # attention_max = tf.reduce_max(attention, axis=1, keep_dims=True)
        # attention_weights = tf.exp(-1e9*(1.-tf.expand_dims(mask_tri,axis=-1)) + attention - attention_max)
        # Z = tf.reduce_sum(attention_weights, axis=1, keep_dims=True)
        # attention = attention_weights / (Z + 1E-3)

        # attention_max = tf.reduce_max(attention, axis=1, keep_dims=True)
        # attention_weights = tf.expand_dims(mask_tri,axis=-1) * tf.exp(attention - attention_max)
        attention_max = tf.reduce_max(attention, axis=1, keep_dims=True)
        attention_weights = tf.exp(
            -1e9 * tf.expand_dims(mask_tri + (1. - mask_2D), axis=-1) + attention - attention_max)
        Z = tf.reduce_sum(attention_weights, axis=1, keep_dims=True)
        attention = attention_weights / (Z + 1E-3)

        with tf.variable_scope("3_Attention"):
            tf.summary.image("3_Attention", tf.squeeze(attention), 3)
            # tf.summary.image("3_AttentionSum",tf.transpose(tf.reduce_sum(tf.squeeze(attention),axis=1,keep_dims=True))

        attention = tf.nn.dropout(attention, dropout_p)

        # Attention-weighted average of the edges
        attended_vals = attention * source
        delta_layer = tf.reduce_sum(attended_vals, axis=1, keep_dims=True)

        delta_layer = tf.reshape(delta_layer, (batch_size, 1, length, channels_1D))

        delta_layer = normalize_layer(
            delta_layer, mask_1D, per_channel=False
        )

        input_1D_normed = scale_shift(input_1D_normed, mask_1D, channels_1D,
                                      name="PostAttentionScaleShift")

        with tf.variable_scope("1_NormalizedSummedVals"):
            tf.summary.image("1_NormalizedSummedVals",
                             tf.transpose(delta_layer, perm=[0, 3, 2, 1]), 3)

        out_layer = conv2D(delta_layer, channels_1D,
                           kernel_size=(1, 1),
                           padding="same",
                           activation=None,
                           reuse=reuse,
                           name="Mix3")

        out_layer = out_layer * mask_1D

        with tf.variable_scope("2_AfterAttnState"):
            tf.summary.image("2_AfterAttnState",
                             tf.transpose(out_layer, perm=[0, 3, 2, 1]), 3)

    return input_1D + out_layer


def multihead_attention_1D_generative_bayesian(input_1D, channels_1D,
                                               mask_1D, mask_2D, mask_tri, step, sampler_hyperparams,
                                               attention_type='softmax', reuse=None,
                                               nonlinearity=tf.nn.elu, num_heads=1):
    """ Gated reduction of 2D conv stack. to a 1D conv stack """

    with tf.variable_scope("Attention"):
        KL_list = []

        input_1D_normed = normalize_layer(
            input_1D, mask_1D, per_channel=False
        )

        source_one, layer_KL = conv2D_bayesian(input_1D_normed, channels_1D,
                                               step, sampler_hyperparams,
                                               kernel_size=(1, 1),
                                               padding="same",
                                               activation=nonlinearity,
                                               reuse=reuse,
                                               name="Source1")

        KL_list += layer_KL

        source, layer_KL = conv2D_bayesian(source_one, channels_1D,
                                           step, sampler_hyperparams,
                                           kernel_size=(1, 1),
                                           padding="same",
                                           activation=None,
                                           reuse=reuse,
                                           name="Source")

        KL_list += layer_KL

        source = tf.transpose(source, perm=[0, 2, 1, 3])

        # Attention query
        query_one, layer_KL = conv2D_bayesian(input_1D_normed, channels_1D,
                                              step, sampler_hyperparams,
                                              kernel_size=(1, 1),
                                              padding="same",
                                              activation=nonlinearity,
                                              reuse=reuse,
                                              name="Query1")

        KL_list += layer_KL

        query, layer_KL = conv2D_bayesian(query_one, channels_1D,
                                          step, sampler_hyperparams,
                                          kernel_size=(1, 1),
                                          padding="same",
                                          activation=None,
                                          reuse=reuse,
                                          name="Query2")

        KL_list += layer_KL

        batch_size = tf.shape(input_1D)[0]
        length = tf.shape(input_1D)[2]
        source = tf.reshape(source, (batch_size, length, 1, num_heads, channels_1D / num_heads))

        query = tf.reshape(query, (batch_size, 1, length, num_heads, channels_1D / num_heads))

        # Masked softmax over the edges
        attention = tf.reduce_sum(
            tf.nn.l2_normalize(source, 4) * tf.nn.l2_normalize(query, 4),
            axis=4, keep_dims=True
        )

        ####attention_max = tf.reduce_max(attention, axis=1, keep_dims=True)
        ###attention_weights = tf.expand_dims(mask_tri,axis=-1) * tf.exp(attention - attention_max)
        # attention_max = tf.reduce_max(attention, axis=1, keep_dims=True)
        # attention_weights = tf.exp(-1e9*(1.-tf.expand_dims(mask_tri,axis=-1)) + attention - attention_max)
        # Z = tf.reduce_sum(attention_weights, axis=1, keep_dims=True)
        # attention = attention_weights / (Z + 1E-3)

        # attention_max = tf.reduce_max(attention, axis=1, keep_dims=True)
        # attention_weights = tf.expand_dims(mask_tri,axis=-1) * tf.exp(attention - attention_max)
        attention_max = tf.reduce_max(attention, axis=1, keep_dims=True)
        attention_weights = tf.exp(
            -1e9 * tf.expand_dims(mask_tri + (1. - mask_2D), axis=-1) + attention - attention_max)
        Z = tf.reduce_sum(attention_weights, axis=1, keep_dims=True)
        attention = attention_weights / (Z + 1E-3)

        with tf.variable_scope("3_Attention"):
            tf.summary.image("3_Attention", tf.squeeze(attention), 3)
            # tf.summary.image("3_AttentionSum",tf.transpose(tf.reduce_sum(tf.squeeze(attention),axis=1,keep_dims=True))

        # Attention-weighted average of the edges
        attended_vals = attention * source
        delta_layer = tf.reduce_sum(attended_vals, axis=1, keep_dims=True)

        delta_layer = tf.reshape(delta_layer, (batch_size, 1, length, channels_1D))

        delta_layer = normalize_layer(
            delta_layer, mask_1D, per_channel=False
        )

        with tf.variable_scope("1_NormalizedSummedVals"):
            tf.summary.image("1_NormalizedSummedVals",
                             tf.transpose(delta_layer, perm=[0, 3, 2, 1]), 3)

        out_layer, layer_KL = conv2D_bayesian(delta_layer, channels_1D,
                                              step, sampler_hyperparams,
                                              kernel_size=(1, 1),
                                              padding="same",
                                              activation=None,
                                              reuse=reuse,
                                              name="Mix3")

        KL_list += layer_KL

        out_layer = out_layer * mask_1D

        with tf.variable_scope("2_AfterAttnState"):
            tf.summary.image("2_AfterAttnState",
                             tf.transpose(out_layer, perm=[0, 3, 2, 1]), 3)

    return input_1D + out_layer, KL_list


def conv_graph_generative(state_1D, state_2D, channels_1D, channels_2D, widths_1D,
                          dilations_1D, widths_2D, dilations_2D, mask_1D, mask_2D, mask_tri,
                          reuse=None, dropout_p=0.5, aux_1D=None, aux_2D=None):
    with tf.variable_scope("NodeUpdate"):
        # Update nodes given edges
        update_1D = reduce_2Dto1D_generative(
            state_1D, state_2D, channels_2D, channels_1D,
            mask_1D, mask_2D, mask_tri, reuse=reuse
        )
        # update_1D = reduce_2Dto1D(
        #     state_1D, state_2D, channels_2D, channels_1D,
        #     mask_1D, mask_2D, reuse=reuse
        # )
        with tf.variable_scope("1_reduce2Dto1D"):
            tf.summary.image("1_reduce2Dto1D",
                             tf.transpose(update_1D, perm=[0, 3, 2, 1]), 3)
        # Input-aware convolution on nodes
        input_1D = [state_1D, update_1D]
        if aux_1D is not None:
            input_1D += [aux_1D]
        update_1D = tf.concat(axis=3, values=input_1D)
        conv_channels = int(update_1D.get_shape()[-1])
        # Conv, LayerNorm, TransConv
        update_1D = conv2D(
            update_1D, filters=channels_1D,
            kernel_size=(1, 1), padding="same",
            activation=nonlinearity, reuse=reuse,
            name="DownMix"
        )
        update_1D = normalize_layer(update_1D, mask_1D)
        update_1D = convnet_1D_generative(
            update_1D, channels_1D, mask_1D,
            widths_1D,
            dilations_1D,
            dropout_p,
            reuse=reuse
        )
        # update_1D = convnet_1D(
        #     update_1D, channels_1D, mask_1D,
        #     widths_1D,
        #     dilations_1D,
        #     dropout_p,
        #     reuse=reuse
        # )
        with tf.variable_scope("2_AfterConvnet1D"):
            tf.summary.image("2_AfterConvnet1D",
                             tf.transpose(update_1D, perm=[0, 3, 2, 1]), 3)
        # update_1D = convnet_1D(
        #     update_1D, channels_1D, mask_1D,
        #     reversed(widths_1D),
        #     reversed(dilations_1D),
        #     dropout_p,
        #     reuse=reuse,
        #     transpose=True
        # )
        state_1D = state_1D + update_1D
        # state_1D = normalize_layer(state_1D, mask_1D, per_channel=False)
        with tf.variable_scope("3_AfterStateUpdate"):
            tf.summary.image("3_AfterStateUpdate",
                             tf.transpose(state_1D, perm=[0, 3, 2, 1]), 3)

    with tf.variable_scope("EdgeUpdate"):
        # Update edges given nodes
        update_2D = expand_1Dto2D_generative(
            state_1D,
            channels_1D, channels_2D,
            mask_1D, mask_2D, mask_tri,
            dropout_p,
            reuse=reuse
        )
        # update_2D = expand_1Dto2D(
        #     state_1D,
        #     channels_1D, channels_2D,
        #     mask_1D, mask_2D,
        #     dropout_p,
        #     reuse=reuse
        # )
        # Optional aux input for convolutionss
        input_2D = [state_2D, update_2D]
        if aux_2D is not None:
            input_2D += [aux_2D]
        update_2D = tf.concat(axis=3, values=input_2D)
        conv_channels = int(update_2D.get_shape()[-1])
        # 2D convolution is expensive, so just pre-mix
        update_2D = conv2D(
            update_2D, filters=channels_2D,
            kernel_size=(1, 1), padding="same",
            activation=nonlinearity, reuse=reuse,
            name="DownMix"
        )
        # update_2D *= mask_tri

        # update_2D = convnet_2D(
        #     update_2D, channels_2D, mask_2D,
        #     reversed(widths_2D),
        #     reversed(dilations_2D),
        #     dropout_p,
        #     reuse=reuse,
        #     transpose=True
        # )
        state_2D = state_2D + update_2D

    return state_1D, state_2D


def conv_graph(state_1D, state_2D, channels_1D, channels_2D, widths_1D,
               dilations_1D, widths_2D, dilations_2D, mask_1D, mask_2D,
               reuse=None, dropout_p=0.5, aux_1D=None, aux_2D=None):
    with tf.variable_scope("NodeUpdate"):
        # Update nodes given edges
        update_1D = reduce_2Dto1D(
            state_1D, state_2D, channels_2D, channels_1D,
            mask_1D, mask_2D, reuse=reuse
        )
        # Input-aware convolution on nodes
        input_1D = [state_1D, update_1D]
        if aux_1D is not None:
            input_1D += [aux_1D]
        update_1D = tf.concat(axis=3, values=input_1D)
        conv_channels = int(update_1D.get_shape()[-1])
        # Conv, LayerNorm, TransConv
        update_1D = conv2D(
            update_1D, filters=channels_1D,
            kernel_size=(1, 1), padding="same",
            activation=nonlinearity, reuse=reuse,
            name="DownMix"
        )
        update_1D = normalize_layer(update_1D, mask_1D)
        update_1D = convnet_1D(
            update_1D, channels_1D, mask_1D,
            widths_1D,
            dilations_1D,
            dropout_p,
            reuse=reuse
        )
        # update_1D = convnet_1D(
        #     update_1D, channels_1D, mask_1D,
        #     reversed(widths_1D),
        #     reversed(dilations_1D),
        #     dropout_p,
        #     reuse=reuse,
        #     transpose=True
        # )
        state_1D = state_1D + update_1D
        state_1D = normalize_layer(state_1D, mask_1D, per_channel=False)
    with tf.variable_scope("EdgeUpdate"):
        # Update edges given nodes
        update_2D = expand_1Dto2D(
            state_1D + update_1D,
            channels_1D, channels_2D,
            mask_1D, mask_2D,
            dropout_p,
            reuse=reuse
        )
        # Optional aux input for convolutionss
        input_2D = [state_2D, update_2D]
        if aux_2D is not None:
            input_2D += [aux_2D]
        update_2D = tf.concat(axis=3, values=input_2D)
        conv_channels = int(update_2D.get_shape()[-1])
        # 2D convolution is expensive, so just pre-mix
        update_2D = conv2D(
            update_2D, filters=channels_2D,
            kernel_size=(1, 1), padding="same",
            activation=nonlinearity, reuse=reuse,
            name="DownMix"
        )
        # Convolution on edges
        update_2D = normalize_layer(update_2D, mask_2D)
        update_2D = convnet_2D(
            update_2D, channels_2D, mask_2D,
            widths_2D,
            dilations_2D,
            dropout_p,
            reuse=reuse
        )
        # update_2D = convnet_2D(
        #     update_2D, channels_2D, mask_2D,
        #     reversed(widths_2D),
        #     reversed(dilations_2D),
        #     dropout_p,
        #     reuse=reuse,
        #     transpose=True
        # )
        state_2D = state_2D + update_2D
        state_2D = normalize_layer(state_2D, mask_2D, per_channel=False)
    return state_1D, state_2D


def reduce_2DtoVec(U, mask, num_hidden, num_steps, reuse=None):
    """ Unrolled attentive GRU for reducing a batch of 2D multichannel
        images to single vector (a la set2set)
    """
    with tf.variable_scope("Reduction"):
        batch_size = tf.shape(U)[0]
        num_in_channels = U.get_shape().as_list()[3]
        h = tf.zeros([tf.shape(U)[0], num_hidden])
        for i in range(num_steps):
            with tf.name_scope("Step" + str(i + 1)):
                reuse_layer = None if reuse is None and i is 0 else True
                # Emit query
                q = tf.layers.dense(
                    h, num_in_channels, activation=tf.nn.tanh,
                    name="Query1", reuse=reuse_layer
                )
                q = tf.layers.dense(
                    q, num_in_channels, activation=None,
                    name="Query2", reuse=reuse_layer
                )
                q = tf.reshape(q, [batch_size, 1, 1, num_in_channels])

                # Compute attention weights
                a = tf.reduce_sum(U * q, axis=3, keep_dims=True)
                a_max = tf.reduce_max(a, axis=[1, 2], keep_dims=True)
                a_weights = mask * tf.exp(a - a_max)
                Z = tf.reduce_sum(a_weights, axis=[1, 2], keep_dims=True)
                a_weights = a_weights / (Z + 1E-3)

                # Compute attention-weighted result
                U_avg = tf.reduce_sum(a_weights * U, axis=[1, 2])

                # GRU update
                hU = tf.concat(axis=1, values=[h, U_avg])
                gate1 = tf.layers.dense(
                    hU, num_hidden, activation=tf.nn.sigmoid,
                    name="Gate1", reuse=reuse_layer
                )
                gate2 = tf.layers.dense(
                    hU, num_hidden, activation=tf.nn.sigmoid,
                    name="Gate2", reuse=reuse_layer
                )
                update = tf.nn.tanh(tf.layers.dense(
                    gate2 * h, num_hidden, activation=None,
                    name="h", reuse=reuse_layer
                ) + tf.layers.dense(
                    U_avg, num_hidden, activation=None,
                    name="U", reuse=reuse_layer
                ))
                h = gate1 * h + (1. - gate1) * update
    return h
