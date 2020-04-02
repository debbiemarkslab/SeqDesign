import tensorflow as tf
import numpy as np

def nonlinearity(x):
  return tf.nn.elu(x)

def normalize_layer(X, mask, per_channel=False, reduce_dims=[3]):
    with tf.variable_scope("Normalization"):
        mean, var = tf.nn.moments(X, reduce_dims, keep_dims=True)
        X = (X - mean) / tf.sqrt(var+1E-5)
    return X

def scale_shift(X, mask, channels,name="ScaleAndShift"):
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

def scale_variables(X,scale_type="pos"):
    with tf.variable_scope("scale_variable-"+scale_type):
        bias_initializer = np.ones((1))

        with tf.device('/cpu:0'):

            scale = tf.get_variable("scale",1, dtype=tf.float32,\
                initializer=tf.constant_initializer(0.0001*bias_initializer), trainable=True)

            if scale_type == "pos":
                loc = tf.nn.softplus(tf.get_variable("loc", 1, dtype=tf.float32,
                    initializer=tf.constant_initializer(0.0001*bias_initializer), trainable=True))

            elif scale_type == "neg":
                loc = tf.get_variable("loc", 1, dtype=tf.float32,
                    initializer=tf.constant_initializer(-7.*bias_initializer), trainable=True)

        return tf.squeeze(scale) * X + tf.squeeze(loc)



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
        #with tf.device('/cpu:0'):
        g_init = np.ones((out_channels))
        g = tf.get_variable(
            "g", shape=[out_channels], dtype=tf.float32,
            initializer=tf.constant_initializer(g_init),
            trainable=True
        )
        b = tf.get_variable(
            "b", shape=[out_channels], dtype=tf.float32,
            initializer=tf.constant_initializer(bias_init*np.ones((out_channels))),
            trainable=True
        )
        W = tf.get_variable(
            "W", shape=W_shape, dtype=tf.float32,
            initializer=W_initializer, trainable=True
        )

        weight_list = [tf.reduce_sum(tf.square(g)),\
            tf.reduce_sum(tf.square(b)),\
            tf.reduce_sum(tf.square(W)),\
            ]

        dilated_conv_mask = tf.reshape(
            tf.constant(np.array([1.,1.,0.]),name="DilatedConvMask",dtype=tf.float32),
            [1,3,1,1])

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

        #with tf.device('/cpu:0'):
        g = tf.get_variable(
            "g", shape=[out_channels], dtype=tf.float32,
            initializer=tf.constant_initializer(g_init*np.ones((out_channels))),
            trainable=True
        )
        b = tf.get_variable(
            "b", shape=[out_channels], dtype=tf.float32,
            initializer=tf.constant_initializer(bias_init*np.ones((out_channels))),
            trainable=True
        )
        W = tf.get_variable(
            "W", shape=W_shape, dtype=tf.float32,
            initializer=W_initializer, trainable=True
        )

        weight_list = [tf.reduce_sum(tf.square(g)),\
            tf.reduce_sum(tf.square(b)),\
            tf.reduce_sum(tf.square(W)),\
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



def convnet_1D_generative(inputs, inner_channels, mask, widths, dilations,
    dropout_p=0.5, additional_input=None, additional_layer_input=None,
    reuse=None, transpose=False, nonlinearity=tf.nn.elu):
    """ Residual dilated 1D conv stack. """

    weight_cost_list = []
    with tf.variable_scope("ConvNet1D"):
        up_layer = inputs

        if additional_layer_input != None:
            up_layer_normed = normalize_layer(
                tf.concat([up_layer,additional_layer_input],axis=3),
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

                if additional_input != None:
                    up_layer_normed = normalize_layer(
                        tf.concat([up_layer,additional_input],axis=3),
                        mask, per_channel=False
                    )
                elif additional_layer_input != None and i == 0:
                    up_layer_normed = up_layer_normed

                else:
                    up_layer_normed = normalize_layer(
                        up_layer, mask, per_channel=False
                    )

                up_layer_normed = scale_shift(up_layer_normed, mask, up_layer_normed.get_shape().as_list()[-1])

                delta_layer, weight_cost = conv2D(up_layer_normed,inner_channels,
                                     kernel_size=(1, 1),
                                     padding="same",
                                     activation=nonlinearity,
                                     reuse=reuse,
                                     name="Mix1" + str(i))

                weight_cost_list += weight_cost

                delta_layer, weight_cost = conv2D_generative(delta_layer,inner_channels,
                                     kernel_size=(1, width),
                                     padding="same",
                                     activation=nonlinearity,
                                     reuse=reuse,
                                     dilation_rate=(1, dilation),
                                     name="DilatedConvGen" + str(i))

                weight_cost_list += weight_cost

                delta_layer, weight_cost = conv2D(delta_layer,inner_channels,
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
