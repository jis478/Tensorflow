from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, ReLU, Conv2DTranspose, LeakyReLU, Layer, ZeroPadding2D, BatchNormalization, LayerNormalization
from tensorflow.keras.activations import tanh

class ConvBlock_G(tf.keras.Model):
    def __init__(self,out_channel, ker_size, padd, stride): 
        super(ConvBlock_G, self).__init__()
        self.out_channel = out_channel
        self.ker_size = ker_size
        self.padd = padd
        self.stride = stride
        self.Conv = Conv2D(self.out_channel, self.ker_size, strides=(self.stride, self.stride), padding='VALID', use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02) , bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))
        self.BN = BatchNormalization()        
        self.Act = LeakyReLU(alpha=0.2)
        self.padding = ZeroPadding2D(self.padd)
        
    def __call__(self,x,train):
        if self.padd > 0:
            x = self.padding(x)
        x = self.Conv(x)
        x = self.BN(x, training=train) 
        x = self.Act(x) 
        return x
    
class ConvBlock_D(tf.keras.Model):
    def __init__(self,out_channel, ker_size, padd, stride): 
        super(ConvBlock_D, self).__init__()
        self.out_channel = out_channel
        self.ker_size = ker_size
        self.padd = padd
        self.stride = stride
        self.Conv = Conv2D(self.out_channel, self.ker_size, strides=(self.stride, self.stride), padding='VALID', use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02) , bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))
        self.LN = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.Act = LeakyReLU(alpha=0.2)
        self.padding = ZeroPadding2D(self.padd)
        
    def __call__(self,x):
        if self.padd > 0:
            x = self.padding(x)
        x = self.Conv(x)
        x = self.LN(x) 
        x = self.Act(x) 
        return x
        
class GeneratorConcatSkip2CleanAdd(tf.keras.Model):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        N = opt.nfc
        
        # head
        self.head = ConvBlock_G(N,opt.ker_size,opt.padd_size,1) 
        
        # body
        self.body = []
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            self.body.append((ConvBlock_G(max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)))

        # tail
        self.tail = []
        self.tail.append(ZeroPadding2D(opt.padd_size)) # zero_padding은 class
        self.tail.append(Conv2D(opt.nc_im, opt.ker_size, strides=(1,1), padding='VALID', use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02) , bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
        self.tail.append(tf.keras.activations.tanh) 
            
    def __call__(self,x,y,train): 
        print(x.shape)
        x = self.head(x, train=train)
        for block in self.body:
            x = block(x, train=train)
        for block in self.tail: # tail은 bn 없음
            x = block(x)                    
        ind = int((int(y.shape[2])-int(x.shape[2]))/2)
        y = y[:,ind:(int(y.shape[1])-ind),ind:(int(y.shape[2])-ind), :]
        return x+y
    
    
class WDiscriminator(tf.keras.Model):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        N = int(opt.nfc)
        
        # head
        self.head = ConvBlock_D(N,opt.ker_size,opt.padd_size,1) 
        
        # body
        self.body = []
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            self.body.append((ConvBlock_D(max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)))
            
        # tail
        self.tail = []
        self.tail.append(ZeroPadding2D(opt.padd_size))
        self.tail.append(Conv2D(1, opt.ker_size, strides=(1,1), padding='VALID', use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02) , bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    
    def __call__(self,x):
        x = self.head(x) 
        for block in self.body:
            x = block(x)
        for block in self.tail:
            x = block(x)
        return x
    
