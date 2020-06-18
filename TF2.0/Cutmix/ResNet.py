# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

class BasicBlock(tf.keras.layers.Layer): 
    expansion = 1   
      
    def __init__(self, out_planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.pad = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
        self.conv1 = tf.keras.layers.Conv2D(filters=out_planes,kernel_size=(3, 3), strides=(stride, stride), use_bias=False, kernel_initializer=tf.keras.initializers.he_normal())      
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=out_planes,kernel_size=(3, 3), strides=(1, 1), use_bias=False, kernel_initializer=tf.keras.initializers.he_normal())      
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.downsample = downsample

    def call(self, inputs):
        residual = inputs

        x = self.pad(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.pad(X)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
           residual = self.downsample(inputs)

        x = x + residual
        x = tf.nn.relu(x)

        return output


class BottleNeck(tf.keras.layers.Layer): 
    expansion = 4 

    def __init__(self, planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.downsample = downsample
        self.conv1 = tf.keras.layers.Conv2D(filters=planes, kernel_size=(1, 1), use_bias=False)  # tf와 py defaut conv2d 같은지?
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
        self.conv2 = tf.keras.layers.Conv2D(filters=planes, kernel_size=(3, 3), strides=(stride, stride), use_bias=False) 
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=planes * BottleNeck.expansion, kernel_size=(1, 1), use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        residual = inputs

        out = self.conv1(inputs)
        out = self.bn1(out,training=training)
        out = tf.nn.relu(out)

        out = self.pad1(out)
        out = self.conv2(out)
        out = self.bn2(out,training=training)
        out = tf.nn.relu(out)

        out = self.conv3(out)
        out = self.bn3(out,training=training)

        if self.downsample is not None:
            residual = self.downsample(inputs)

        out += residual
        out = tf.nn.relu(out)

        return out


class ResNet(tf.keras.Model):
    def __init__(self, dataset, depth, num_classes, bottleneck=False):
        super(ResNet, self).__init__()        
        self.dataset = dataset
        if self.dataset.startswith('cifar'):
            self.inplanes = 16  
            
            if bottleneck == True:
                n = int((depth - 2) / 9)
                block = BottleNeck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock

            self.pad1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
            self.conv1 = tf.keras.layers.Conv2D(filters=self.inplanes, kernel_size=(3, 3), strides=(1, 1), use_bias=False)  # tf와 py defaut conv2d 같은지?
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.relu = tf.nn.relu

            self.layer1 = self._make_layer(block, 16, n)
            self.layer2 = self._make_layer(block, 32, n, stride=2)
            self.layer3 = self._make_layer(block, 64, n, stride=2) 
            self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(8, 8))
            self.fc = tf.keras.layers.Dense(num_classes)

        elif self.dataset.startswith('imagenet'):
          ''' to be implemented soon '''
          pass 


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
          downsample = tf.keras.Sequential()
          downsample.add(tf.keras.layers.Conv2D(filters=planes * block.expansion, kernel_size=(1, 1), strides=(stride, stride), use_bias=False))
          downsample.add(tf.keras.layers.BatchNormalization())
  
        self.inplanes = planes * block.expansion
        layers = tf.keras.Sequential()
        layers.add(block(planes, stride, downsample))
        for i in range(1, blocks):
            layers.add(block(planes))

        return layers

    def call(self, x):
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':

            x = self.pad1(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = self.fc(x)

        elif self.dataset == 'imagenet':
          ''' to be implemented soon '''
          pass 
    
        return x
