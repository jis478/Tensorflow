import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np




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
        if self.normal in ('IN', 'ILN'):
            x = self.normal(self.conv(x))
        elif self.normal == 'SN':
            x = self.conv_sn(x)
        else:
            x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x

