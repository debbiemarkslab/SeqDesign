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
