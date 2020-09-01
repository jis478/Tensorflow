import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np

class SpectralNormalization(tf.keras.layers.Wrapper):
    ''' # https://github.com/thisisiron/spectral_normalization-tf2/blob/master/sn.py
        # https://groups.google.com/a/tensorflow.org/g/discuss/c/PRjyj6tiQvU?pli=1 '''
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        # print('w_shape', self.w_shape)

        # Conv layer: k x k x previous_c x next_c
        if len(self.w_shape) >= 3:  # Conv2D
            self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                     initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                     trainable=False,
                                     name='sn_v',
                                     dtype=tf.float32)

        # Dense layer: prev_unit X next_unit
        else:
            self.v = self.add_weight(shape=(1, self.w_shape[0]),  # 1 x 2048
                                     initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                     trainable=False,
                                     name='sn_v',
                                     dtype=tf.float32)

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_u',
                                 dtype=tf.float32)

        super(SpectralNormalization, self).build()

    def call(self, inputs):
        self.update_weights()
        output = self.layer(inputs)
        self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = self.v

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))  #
                v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        # print(v_hat.shape)
        self.v.assign(v_hat)

        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        self.layer.kernel.assign(self.w)




class Conv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, pad, normal=None, act=None, use_bias=False, pad_type='REFLECT', **kwargs):
        super(Conv, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pad = pad
        self.normal = normal
        self.act = act
        self.use_bias = use_bias
        self.pad_type = pad_type
        if 'ILN_factor' in kwargs:
            self.ILN_factor = kwargs['ILN_factor']

        # normalization
        if self.normal == 'IN':
            self.normal = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,
                                                           beta_initializer="random_uniform",
                                                           gamma_initializer="random_uniform")
        elif self.normal == 'ILN':
            self.normal = ILN(self.ILN_factor)

        # activation
        if self.act == 'relu':
            self.act = tf.keras.layers.ReLU()
        elif self.act == 'tanh':
            self.act = tf.keras.activations.tanh
        elif self.act == 'lrelu':
            self.act = tf.keras.layers.LeakyReLU(alpha=0.2)

        # padding
        if self.pad > 0:
            if (self.kernel_size - self.strides) % 2 == 0:
                self.pad_top = self.pad
                self.pad_bottom = self.pad
                self.pad_left = self.pad
                self.pad_right = self.pad

            else:
                self.pad_top = self.pad
                self.pad_bottom = self.kernel_size - self.strides - self.pad_top
                self.pad_left = self.pad
                self.pad_right = self.kernel_size - self.strides - self.pad_left

        # conv2d
        if self.normal == 'SN':
            self.conv = SpectralNormalization(
                tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(self.kernel_size, self.kernel_size),
                                       kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                       strides=(self.strides, self.strides), use_bias=self.use_bias))
        else:
            self.conv = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(self.kernel_size, self.kernel_size),
                                               kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                               strides=(self.strides, self.strides), use_bias=self.use_bias)

    def __call__(self, x):
        if self.pad > 0:
            x = tf.pad(x, [[0, 0], [self.pad_top, self.pad_bottom], [self.pad_left, self.pad_right], [0, 0]],
                       mode=self.pad_type)
        x = self.conv(x)
        if (self.normal is not None) & (self.normal != 'SN'):
            x = self.normal(x)
        if self.act is not None:
            x = self.act(x)
        return x




class ResnetBlock(tf.keras.layers.Layer):
  def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        self.use_bias = use_bias
        self.dim = dim
        self.conv1 = Conv(filters = self.dim, kernel_size = 3, strides = 1, pad = 1, normal = 'IN',
                          act = 'relu', use_bias=self.use_bias, pad_type='REFLECT')
        self.conv2 = Conv(filters = self.dim, kernel_size = 3, strides = 1, pad = 1, normal = 'IN',
                          act = None, use_bias=self.use_bias, pad_type='REFLECT')

  def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x += inputs
        return x




class ResnetAdaILNBlock(tf.keras.layers.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.dim = dim
        self.use_bias = use_bias
        self.conv1 = tf.keras.layers.Conv2D(filters=self.dim, kernel_size=(3, 3),
                                            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                            strides=(1, 1), use_bias=self.use_bias)
        self.norm1 = adaILN(self.dim)
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters=self.dim, kernel_size=(3, 3),
                                            kernel_regularizer=tf.keras.regularizers.l2(0.0001), strides=(1, 1),
                                            use_bias=self.use_bias)
        self.norm2 = adaILN(self.dim)

    def call(self, inputs, gamma, beta):
        x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.conv1(x)
        x = self.norm1(x, gamma, beta)
        x = self.relu1(x)
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.conv2(x)
        x = self.norm2(x, gamma, beta)
        x += inputs
        return x




class adaILN(tf.keras.layers.Layer):
    def __init__(self, num_features):
        super(adaILN, self).__init__()
        self.num_features = num_features
        self.eps = 1e-12

    def build(self, input_shape):  # 반드시 input_shape를 써야하는지? input shape에 상관 없으므로, 그냥 __init__에 넣어도 될 듯하다.
        self.rho = tf.Variable(initial_value=tf.fill([1, 1, 1, self.num_features], 0.9), dtype=tf.float32,
                               trainable=True)

    def call(self, inputs, gamma, beta):
        in_mean, in_var = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True), tf.math.reduce_variance(inputs,
                                                                                                      axis=[1, 2],
                                                                                                      keepdims=True)
        out_in = (inputs - in_mean) / tf.math.sqrt(in_var + self.eps)
        ln_mean, ln_var = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True), tf.math.reduce_variance(inputs,
                                                                                                         axis=[1, 2, 3],
                                                                                                         keepdims=True)
        out_ln = (inputs - ln_mean) / tf.math.sqrt(ln_var + self.eps)
        out = tf.broadcast_to(self.rho,
                              [inputs.shape[0], self.rho.shape[1], self.rho.shape[2], self.rho.shape[3]]) * out_in + (
                          1 - tf.broadcast_to(self.rho, [inputs.shape[0], self.rho.shape[1], self.rho.shape[2],
                                                         self.rho.shape[3]])) * out_ln
        out = out * tf.expand_dims(tf.expand_dims(gamma, axis=1), axis=1) + tf.expand_dims(tf.expand_dims(beta, axis=1),
                                                                                           axis=1)  # (batch,channel -> batch, height, width, channel)
        self.rho.assign(tf.clip_by_value(self.rho, clip_value_min=0, clip_value_max=1))
        return out






class ILN(tf.keras.layers.Layer):
    def __init__(self, num_features):
        super(ILN, self).__init__()
        self.num_features = num_features
        self.eps = 1e-12

    def build(self, input_shape):
        self.rho = tf.Variable(initial_value=tf.fill([1, 1, 1, self.num_features], 0.0), dtype=tf.float32,
                               trainable=True)
        self.gamma = tf.Variable(initial_value=tf.fill([1, 1, 1, self.num_features], 1.0), dtype=tf.float32,
                                 trainable=True)
        self.beta = tf.Variable(initial_value=tf.fill([1, 1, 1, self.num_features], 0.0), dtype=tf.float32,
                                trainable=True)

    def call(self, inputs):
        in_mean, in_var = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True), tf.math.reduce_variance(inputs,
                                                                                                      axis=[1, 2],
                                                                                                      keepdims=True)
        out_in = (inputs - in_mean) / tf.math.sqrt(in_var + self.eps)
        ln_mean, ln_var = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True), tf.math.reduce_variance(inputs,
                                                                                                         axis=[1, 2, 3],
                                                                                                         keepdims=True)
        out_ln = (inputs - ln_mean) / tf.math.sqrt(ln_var + self.eps)
        out = tf.broadcast_to(self.rho,
                              [inputs.shape[0], self.rho.shape[1], self.rho.shape[2], self.rho.shape[3]]) * out_in + (
                          1 - tf.broadcast_to(self.rho, [inputs.shape[0], self.rho.shape[1], self.rho.shape[2],
                                                         self.rho.shape[3]])) * out_ln
        out = out * tf.broadcast_to(self.gamma, [inputs.shape[0], self.rho.shape[1], self.rho.shape[2],
                                                 self.rho.shape[3]]) + tf.broadcast_to(self.beta, [inputs.shape[0],
                                                                                                   self.rho.shape[1],
                                                                                                   self.rho.shape[2],
                                                                                                   self.rho.shape[3]])
        self.rho.assign(tf.clip_by_value(self.rho, clip_value_min=0, clip_value_max=1))
        return out
