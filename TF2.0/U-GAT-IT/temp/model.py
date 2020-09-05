import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
from ops import *
from utils import *

class ResnetGenerator(tf.keras.Model):
    def __init__(self, output_nc, ngf=64, n_blocks=6, light=False):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.light = light

        self.DownBlock = []
        self.DownBlock.append(Conv(filters=self.ngf, kernel_size=7, strides=1, pad=3, normal='IN',
                                   act='relu', use_bias=False, pad_type='REFLECT'))

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            self.DownBlock.append(Conv(filters=self.ngf * mult * 2, kernel_size=3, strides=2, pad=1, normal='IN',
                                       act='relu', use_bias=False, pad_type='REFLECT'))

        # Down-Sampling Bottleneck
        mult = 2 ** n_downsampling
        for i in range(self.n_blocks):
            self.DownBlock.append(ResnetBlock(self.ngf * mult, use_bias=False))

        # CAM
        self.gap_fc = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.0001), use_bias=False)
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.gmp_fc = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.0001), use_bias=False)
        self.gmp = tf.keras.layers.GlobalMaxPool2D()
        self.conv1x1 = tf.keras.layers.Conv2D(filters=self.ngf * mult, kernel_size=(1, 1),
                                              kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                              strides=(1, 1), use_bias=True)
        self.relu = tf.keras.layers.ReLU()

        # Gamma, Beta block (the argument 'self.light' is not used in Tensorflow as the input dims are automatically detected)
        self.FC = []
        self.FC.append(
            tf.keras.layers.Dense(self.ngf * mult, kernel_regularizer=tf.keras.regularizers.l2(0.0001), use_bias=False))
        self.FC.append(tf.keras.layers.ReLU())
        self.FC.append(
            tf.keras.layers.Dense(self.ngf * mult, kernel_regularizer=tf.keras.regularizers.l2(0.0001), use_bias=False))
        self.FC.append(tf.keras.layers.ReLU())
        self.gamma = tf.keras.layers.Dense(self.ngf * mult, kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                           use_bias=False)
        self.beta = tf.keras.layers.Dense(self.ngf * mult, kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                          use_bias=False)

        self.UpBlock1 = []
        for _ in range(self.n_blocks):
            self.UpBlock1.append(ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        self.UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            self.UpBlock2.append(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
            self.UpBlock2.append(
                Conv(filters=int(self.ngf * mult / 2), kernel_size=3, strides=1, pad=1, normal='ILN', act='relu',
                     use_bias=False, pad_type='REFLECT', ILN_factor=int(self.ngf * mult / 2)))
        self.UpBlock2.append(Conv(filters=self.output_nc, kernel_size=7, strides=1, pad=3, normal=None,
                                  act='tanh', use_bias=False, pad_type='REFLECT'))

    def call(self, x):

        for layer in self.DownBlock:
            x = layer(x)

        gap = self.gap(x)  # Global Average Pooling  # eg. 1 x 32 x 32 x 256 -> 1 x 256
        gap_logit = self.gap_fc(gap)  # 1 x 1
        gap_weight = tf.squeeze(self.gap_fc.trainable_variables[0])
        gap = x * gap_weight  # feature map에 gap_weight를 element-wise로 multiply. broadcasting 가능. 즉, 4 x 32 x 32 x 256 이니까.. 4 x 1 x 1 x 256 로 multiply하면 broadcasting (test필요)

        gmp = self.gmp(x)  # Global Average Pooling  # 4 x 32 x 32 x 256 -> 4 x 256 or 4 x 1 x 1 x 256?
        gmp_logit = self.gmp_fc(gmp)  # 4 x 1
        gmp_weight = tf.squeeze(self.gmp_fc.trainable_variables[0])
        gmp = x * gmp_weight  # feature map에 gap_weight를 element-wise로 multiply. broadcasting 가능. 즉, 4 x 32 x 32 x 256 이니까.. 4 x 1 x 1 x 256 로 multiply하면 broadcasting (test필요)

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


class Discriminator(tf.keras.Model):
    def __init__(self, ndf=64, n_layers=5):
        assert (n_layers >= 0)
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.n_layers = n_layers

        self.model = tf.keras.Sequential()
        self.model.add(Conv(filters=self.ndf, kernel_size=4, strides=2, pad=1, normal='SN', act='lrelu',
                            use_bias=True, pad_type='REFLECT'))

        for i in range(1, self.n_layers - 2):
            mult = 2 ** (i - 1)
            self.model.add(Conv(filters=self.ndf * mult * 2, kernel_size=4, strides=2, pad=1, normal='SN', act='lrelu',
                                use_bias=True, pad_type='REFLECT'))

        mult = 2 ** (self.n_layers - 2 - 1)
        self.model.add(Conv(filters=self.ndf * mult * 2, kernel_size=4, strides=1, pad=1, normal='SN', act='lrelu',
                            use_bias=True, pad_type='REFLECT'))

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.gap_fc = SpectralNormalization(
            tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.0001), use_bias=False))
        # self.gap_fc = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.0001), use_bias=False)
        self.gmp = tf.keras.layers.GlobalMaxPool2D()
        # self.gmp_fc = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.0001), use_bias=False)
        self.gmp_fc = SpectralNormalization(
            tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.0001), use_bias=False))
        # self.gmp_fc = SpectralNormalization(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.0001), use_bias=False))
        self.conv1x1 = tf.keras.layers.Conv2D(filters=self.ndf * mult, kernel_size=(1, 1),
                                              kernel_regularizer=tf.keras.regularizers.l2(0.0001), strides=(1, 1),
                                              use_bias=True)
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.conv = Conv(filters=1, kernel_size=4, strides=1, pad=1, normal='SN', act=None,
                         use_bias=False, pad_type='REFLECT')

    def call(self, inputs):
        x = self.model(inputs)

        gap = self.gap(x)
        gap_logit = self.gap_fc(gap)  # 4 x 1
        gap_weight = tf.squeeze(self.gap_fc.trainable_variables[0])
        gap = x * gap_weight  # feature map에 gap_weight를 element-wise로 multiply. broadcasting 가능. 즉, 4 x 32 x 32 x 256 이니까.. 4 x 1 x 1 x 256 로 multiply하면 broadcasting (test필요)

        gmp = self.gmp(x)  # Global Average Pooling  # 4 x 32 x 32 x 256 -> 4 x 256 or 4 x 1 x 1 x 256?
        gmp_logit = self.gmp_fc(gmp)  # 4 x 1
        gmp_weight = tf.squeeze(self.gmp_fc.trainable_variables[0])
        gmp = x * gmp_weight  # feature map에 gap_weight를 element-wise로 multiply. broadcasting 가능. 즉, 4 x 32 x 32 x 256 이니까.. 4 x 1 x 1 x 256 로 multiply하면 broadcasting (test필요)

        cam_logit = tf.keras.layers.concatenate([gap_logit, gmp_logit], axis=-1)
        x = tf.keras.layers.concatenate([gap, gmp], axis=-1)
        x = self.lrelu(self.conv1x1(x))

        heatmap = tf.reduce_sum(x, axis=3, keepdims=True)

        out = self.conv(x)

        return out, cam_logit, heatmap