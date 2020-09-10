import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
from utils import *


# from https://github.com/thisisiron/spectral_normalization-tf2/blob/master/sn.py
# 1)the restore function was deleted
# 2)dense layer is supported '''
class SpectralNormalization(tf.keras.layers.Wrapper):
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

        # 4 x 4 x 3 x 32 (h w previous_c next_c)   # 324 X 1 ( prev_unit, next_unit)
        if len(self.w_shape) >= 3:  # Conv2D
            self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                     initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                     trainable=False,
                                     name='sn_v',
                                     dtype=tf.float32)
        else:
            self.v = self.add_weight(shape=(1, self.w_shape[0]),
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
        self.v.assign(v_hat)
        self.layer.kernel.assign(self.w / sigma)


class Conv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, pad, normal=None, act=None, use_bias=False, pad_type='REFLECT',
                 weight_decay=0.0001, **kwargs):
        super(Conv, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pad = pad
        self.normal = normal
        self.act = act
        self.use_bias = use_bias
        self.pad_type = pad_type
        self.weight_decay = weight_decay
        if 'ILN_factor' in kwargs:
            self.ILN_factor = kwargs['ILN_factor']

        # normalization
        if self.normal == 'IN':
            self.normal = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,
                                                           beta_initializer="random_uniform",
                                                           gamma_initializer="random_uniform")
        elif self.normal == 'ILN':
            self.normal = ILN(self.ILN_factor)

        elif self.normal == 'SN':
            self.conv_sn = SpectralNormalization(
                tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(self.kernel_size, self.kernel_size),
                                       kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),
                                       kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                       strides=(self.strides, self.strides), use_bias=self.use_bias))

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
        if self.normal != 'SN':
            self.conv = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(self.kernel_size, self.kernel_size),
                                               kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),
                                               kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                               strides=(self.strides, self.strides), use_bias=self.use_bias)

    def __call__(self, x):
        if self.pad > 0:
            x = tf.pad(x, [[0, 0], [self.pad_top, self.pad_bottom], [self.pad_left, self.pad_right], [0, 0]],
                       mode=self.pad_type)
        if self.normal not in ('SN', None):
            x = self.normal(self.conv(x))
        elif self.normal == 'SN':
            x = self.conv_sn(x)
        else:
            x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ResnetGenerator(tf.keras.Model):
    def __init__(self, output_nc, ngf=64, n_blocks=6, weight_decay=0.0001, light=False):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.weight_decay = weight_decay
        self.light = light

        self.DownBlock = []
        self.DownBlock.append(Conv(filters=self.ngf, kernel_size=7, strides=1, pad=3, normal='IN',
                                   act='relu', use_bias=True, pad_type='REFLECT', weight_decay=self.weight_decay))

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            self.DownBlock.append(Conv(filters=self.ngf * mult * 2, kernel_size=3, strides=2, pad=1, normal='IN',
                                       act='relu', use_bias=False, pad_type='REFLECT', weight_decay=self.weight_decay))

        # Down-Sampling Bottleneck
        mult = 2 ** n_downsampling
        for i in range(self.n_blocks):
            self.DownBlock.append(ResnetBlock(self.ngf * mult, use_bias=False, weight_decay=self.weight_decay))

        # CAM
        self.cam_fc = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(self.weight_decay),
                                            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),
                                            use_bias=True)
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.gmp = tf.keras.layers.GlobalMaxPool2D()
        self.conv1x1 = tf.keras.layers.Conv2D(filters=self.ngf * mult, kernel_size=(1, 1),
                                              kernel_regularizer=tf.keras.regularizers.L2(self.weight_decay),
                                              strides=(1, 1), use_bias=True)
        self.relu = tf.keras.layers.ReLU()

        # Gamma, Beta
        self.FC = []
        for i in range(2):
            self.FC.append(
                tf.keras.layers.Dense(self.ngf * self.n_blocks,
                                      kernel_regularizer=tf.keras.regularizers.L2(self.weight_decay),
                                      use_bias=True))
            self.FC.append(tf.keras.layers.ReLU())
        self.gamma = tf.keras.layers.Dense(self.ngf * self.n_blocks,
                                           kernel_regularizer=tf.keras.regularizers.L2(self.weight_decay),
                                           use_bias=True)
        self.beta = tf.keras.layers.Dense(self.ngf * self.n_blocks,
                                          kernel_regularizer=tf.keras.regularizers.L2(self.weight_decay),
                                          use_bias=True)

        # Up-Sampling Bottleneck
        self.UpBlock1 = []
        for _ in range(self.n_blocks):
            self.UpBlock1.append(ResnetAdaILNBlock(self.ngf * mult, use_bias=True, smoothing=True))

        # Up-Sampling
        self.UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            self.UpBlock2.append(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
            self.UpBlock2.append(Conv(filters=self.ngf * mult // 2, kernel_size=3, strides=1, pad=1,
                                      normal='ILN', act='relu', use_bias=False, pad_type='REFLECT',
                                      weight_decay=self.weight_decay, ILN_factor=self.ngf * mult // 2))
        self.UpBlock2.append(Conv(filters=self.output_nc, kernel_size=7, strides=1, pad=3, normal=None,
                                  act='tanh', use_bias=False, pad_type='REFLECT', weight_decay=self.weight_decay))

    def call(self, x):
        for layer in self.DownBlock:
            x = layer(x)
        gap = self.gap(x)
        gap_logit = self.cam_fc(gap)
        gap_weight = tf.squeeze(self.cam_fc.trainable_variables[0] + self.cam_fc.trainable_variables[1])
        gap = x * gap_weight
        gmp = self.gmp(x)
        gmp_logit = self.cam_fc(gmp)
        gmp_weight = tf.squeeze(self.cam_fc.trainable_variables[0] + self.cam_fc.trainable_variables[1])
        gmp = x * gmp_weight

        cam_logit = tf.keras.layers.concatenate([gap_logit, gmp_logit], axis=-1)
        x = tf.keras.layers.concatenate([gap, gmp], axis=-1)
        x = self.relu(self.conv1x1(x))
        heatmap = tf.reduce_sum(x, axis=-1, keepdims=True)

        if self.light:
            x_ = self.gap(x)
            x_ = tf.keras.layers.Flatten()(x_)
            for layer in self.FC:
                x_ = layer(x_)
        else:
            x_ = tf.keras.layers.Flatten()(x)
            for layer in self.FC:
                x_ = layer(x_)

        gamma, beta = self.gamma(x_), self.beta(x_)

        for layer in self.UpBlock1:
            x = layer(x, gamma, beta)

        for layer in self.UpBlock2:
            x = layer(x)

        return x, cam_logit, heatmap


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, dim, use_bias, weight_decay=0.0001):
        super(ResnetBlock, self).__init__()
        self.use_bias = use_bias
        self.dim = dim
        self.weight_decay = weight_decay
        self.conv1 = Conv(filters=self.dim, kernel_size=3, strides=1, pad=1, normal='IN',
                          act='relu', use_bias=self.use_bias, pad_type='REFLECT', weight_decay=self.weight_decay)
        self.conv2 = Conv(filters=self.dim, kernel_size=3, strides=1, pad=1, normal='IN',
                          act=None, use_bias=self.use_bias, pad_type='REFLECT', weight_decay=self.weight_decay)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x += inputs
        return x


class ResnetAdaILNBlock(tf.keras.layers.Layer):
    def __init__(self, dim, use_bias, smoothing=True, weight_decay=0.0001):
        super(ResnetAdaILNBlock, self).__init__()
        self.dim = dim
        self.use_bias = use_bias
        self.smoothing = smoothing
        self.weight_decay = weight_decay
        self.conv1 = tf.keras.layers.Conv2D(filters=self.dim, kernel_size=(3, 3),
                                            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                            strides=(1, 1), use_bias=self.use_bias)
        self.norm1 = AdaILN(self.dim, self.smoothing)
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters=self.dim, kernel_size=(3, 3),
                                            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                            strides=(1, 1), use_bias=self.use_bias)
        self.norm2 = AdaILN(self.dim, self.smoothing)

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


class AdaILN(tf.keras.layers.Layer):
    def __init__(self, num_features, smoothing=True):
        super(AdaILN, self).__init__()
        self.num_features = num_features
        self.eps = 1e-12
        self.smoothing = smoothing
        self.rho = tf.Variable(name='rho1', initial_value=tf.fill([self.num_features], 1.0), dtype=tf.float32, trainable=True,
                               constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))

    def call(self, inputs, gamma, beta):
        in_mean, in_var = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True), tf.math.reduce_variance(inputs,
                                                                                                      axis=[1, 2],
                                                                                                      keepdims=True)
        out_in = (inputs - in_mean) / tf.math.sqrt(in_var + self.eps)
        ln_mean, ln_var = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True), tf.math.reduce_variance(inputs,
                                                                                                         axis=[1, 2, 3],
                                                                                                         keepdims=True)
        out_ln = (inputs - ln_mean) / tf.math.sqrt(ln_var + self.eps)
        if self.smoothing:
            self.rho.assign(tf.clip_by_value(self.rho - tf.constant(0.1), 0.0, 1.0))
        out = self.rho * out_in + (1 - self.rho) * out_ln
        out = out * tf.expand_dims(tf.expand_dims(gamma, axis=1), axis=1) + tf.expand_dims(tf.expand_dims(beta, axis=1),
                                                                                           axis=1)
        return out


class ILN(tf.keras.layers.Layer):
    def __init__(self, num_features):
        super(ILN, self).__init__()
        self.num_features = num_features
        self.eps = 1e-12
        self.rho = tf.Variable(initial_value=tf.fill([self.num_features], 0.0), dtype=tf.float32, trainable=True,
                               constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        self.gamma = tf.Variable(initial_value=tf.fill([self.num_features], 1.0), dtype=tf.float32, trainable=True)
        self.beta = tf.Variable(initial_value=tf.fill([self.num_features], 0.0), dtype=tf.float32, trainable=True)

    def call(self, inputs):
        in_mean, in_var = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True), tf.math.reduce_variance(inputs,
                                                                                                      axis=[1, 2],
                                                                                                      keepdims=True)
        out_in = (inputs - in_mean) / tf.math.sqrt(in_var + self.eps)
        ln_mean, ln_var = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True), tf.math.reduce_variance(inputs,
                                                                                                         axis=[1, 2, 3],
                                                                                                         keepdims=True)
        out_ln = (inputs - ln_mean) / tf.math.sqrt(ln_var + self.eps)
        out = self.rho * out_in + (1 - self.rho) * out_ln
        out = out * self.gamma + self.beta
        return out


class GlobalDiscriminator(tf.keras.Model):
    def __init__(self, ndf=64, n_layers=6, weight_decay=0.0001):
        assert (n_layers >= 0)
        super(GlobalDiscriminator, self).__init__()
        self.ndf = ndf
        self.n_layers = n_layers
        self.weight_decay = weight_decay

        self.model = tf.keras.Sequential()
        self.model.add(Conv(filters=self.ndf, kernel_size=4, strides=2, pad=1, normal='SN', act='lrelu',
                            use_bias=True, pad_type='REFLECT', weight_decay=self.weight_decay))

        for i in range(1, self.n_layers - 1):
            mult = 2 ** (i - 1)
            self.model.add(Conv(filters=self.ndf * mult * 2, kernel_size=4, strides=2, pad=1, normal='SN', act='lrelu',
                                use_bias=True, pad_type='REFLECT', weight_decay=self.weight_decay))

        mult = 2 ** (self.n_layers - 1 - 1)
        self.model.add(Conv(filters=self.ndf * mult * 2, kernel_size=4, strides=1, pad=1, normal='SN', act='lrelu',
                            use_bias=True, pad_type='REFLECT', weight_decay=self.weight_decay))

        # Class Activation Map
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.gmp = tf.keras.layers.GlobalMaxPool2D()
        self.cam_fc = SpectralNormalization(
            tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(self.weight_decay),
                                  kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),
                                  use_bias=True))
        self.conv1x1 = tf.keras.layers.Conv2D(filters=self.ndf * mult * 2,
                                              kernel_size=(1, 1),
                                              kernel_regularizer=tf.keras.regularizers.L2(self.weight_decay),
                                              strides=(1, 1),
                                              use_bias=True)
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.conv = Conv(filters=1, kernel_size=4, strides=1, pad=1, normal='SN', act=None,
                         use_bias=True, pad_type='REFLECT', weight_decay=self.weight_decay)

    def call(self, inputs):
        x = self.model(inputs)
        gap = self.gap(x)
        gap_logit = self.cam_fc(gap)
        gap_weight = tf.squeeze(self.cam_fc.trainable_variables[0] + self.cam_fc.trainable_variables[1])
        gap = x * gap_weight
        gmp = self.gmp(x)
        gmp_logit = self.cam_fc(gmp)  # 4 x 1
        gmp_weight = tf.squeeze(self.cam_fc.trainable_variables[0] + self.cam_fc.trainable_variables[1])
        gmp = x * gmp_weight
        cam_logit = tf.keras.layers.concatenate([gap_logit, gmp_logit], axis=-1)
        x = tf.keras.layers.concatenate([gap, gmp], axis=-1)
        x = self.lrelu(self.conv1x1(x))
        heatmap = tf.reduce_sum(x, axis=3, keepdims=True)
        out = self.conv(x)
        return out, cam_logit, heatmap


class LocalDiscriminator(tf.keras.Model):
    def __init__(self, ndf=64, n_layers=6, weight_decay=0.0001):
        assert (n_layers >= 0)
        super(LocalDiscriminator, self).__init__()
        self.ndf = ndf
        self.n_layers = n_layers
        self.weight_decay = weight_decay
        self.model = tf.keras.Sequential()
        self.model.add(Conv(filters=self.ndf, kernel_size=4, strides=2, pad=1, normal='SN', act='lrelu',
                            use_bias=True, pad_type='REFLECT', weight_decay=self.weight_decay))

        for i in range(1, self.n_layers - 2 - 1):
            mult = 2 ** (i - 1)
            self.model.add(Conv(filters=self.ndf * mult * 2, kernel_size=4, strides=2, pad=1, normal='SN', act='lrelu',
                                use_bias=True, pad_type='REFLECT', weight_decay=self.weight_decay))

        mult = 2 ** (self.n_layers - 1 - 1)
        self.model.add(Conv(filters=self.ndf * mult * 2, kernel_size=4, strides=1, pad=1, normal='SN', act='lrelu',
                            use_bias=True, pad_type='REFLECT', weight_decay=self.weight_decay))

        # Class Activation Map
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.gmp = tf.keras.layers.GlobalMaxPool2D()
        self.cam_fc = SpectralNormalization(
            tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(self.weight_decay),
                                  kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),
                                  use_bias=True))
        self.conv1x1 = tf.keras.layers.Conv2D(filters=self.ndf * mult * 2, kernel_size=(1, 1),
                                              kernel_regularizer=tf.keras.regularizers.L2(self.weight_decay),
                                              strides=(1, 1),
                                              use_bias=True)
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.conv = Conv(filters=1, kernel_size=4, strides=1, pad=1, normal='SN', act=None,
                         use_bias=True, pad_type='REFLECT', weight_decay=self.weight_decay)

    def call(self, inputs):
        x = self.model(inputs)
        gap = self.gap(x)
        gap_logit = self.cam_fc(gap)
        gap_weight = tf.squeeze(self.cam_fc.trainable_variables[0] + self.cam_fc.trainable_variables[1])
        gap = x * gap_weight
        gmp = self.gmp(x)
        gmp_logit = self.cam_fc(gmp)
        gmp_weight = tf.squeeze(self.cam_fc.trainable_variables[0] + self.cam_fc.trainable_variables[1])
        gmp = x * gmp_weight
        cam_logit = tf.keras.layers.concatenate([gap_logit, gmp_logit], axis=-1)
        x = tf.keras.layers.concatenate([gap, gmp], axis=-1)
        x = self.lrelu(self.conv1x1(x))
        heatmap = tf.reduce_sum(x, axis=3, keepdims=True)
        out = self.conv(x)
        return out, cam_logit, heatmap
