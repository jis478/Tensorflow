# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def model_save(model):
  exp_directory = f'runs/{args.expname}/'
  if not os.path.exists(exp_directory):
      os.makedirs(exp_directory)
  # filename=f'Resnet50_cutmix_{epoch}_test_err1_{test_err1:.3f}'        
  filename='Resnet50_cutmix_best_model'        
  filename = directory + filename
  model.save(filename)  


def normalize(image, mean=[125.3, 123.0, 113.9], std=[63.0, 62.1, 66.7]):
  R = tf.divide(tf.subtract(image[..., 0], mean[0]), std[0])
  G = tf.divide(tf.subtract(image[..., 1], mean[1]), std[1])
  B = tf.divide(tf.subtract(image[..., 2], mean[2]), std[2])
  return tf.stack([R,G,B], axis=-1)

def train_augment(image,label):
  image = tf.image.resize_with_crop_or_pad(image, 36, 36) 
  image = tf.image.random_crop(image, size=[32, 32, 3]) 
  image = tf.image.random_flip_left_right(image)
  image = tf.cast(image, dtype=tf.float32)
  label = tf.cast(label, dtype=tf.float32)
  return image,label

def test_augment(image,label):
  image = tf.cast(image, dtype=tf.float32)
  image = normalize(image)
  label = tf.cast(label, dtype=tf.float32)
  return image,label

def rand_bbox(size, lam):

    W = size[1] 
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, init_lr):
    self.init_lr = init_lr
    self.steps_per_epoch = len(train_labels) // args.batch_size 

  def __call__(self, step): 
      epoch = step // self.steps_per_epoch 
      self.lr = self.init_lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs* 0.75)))
      return self.lr 

  def _get_lr(self):
      return self.lr.numpy()
