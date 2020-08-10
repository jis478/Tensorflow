# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# def model_save(model):
#   exp_directory = f'runs/{args.expname}/'
#   if not os.path.exists(exp_directory):
#       os.makedirs(exp_directory)
#   # filename=f'Resnet50_cutmix_{epoch}_test_err1_{test_err1:.3f}'        
#   filename='Resnet50_cutmix_best_model'        
#   filename = directory + filename
#   model.save(filename)  


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

  
# def learning_rate_schedule(boundaries, values, train_labels):
#     lr_bound = []
#     for bound in boundaries:
#       lr_bound.append(bound * len(train_labels))      
#     return tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_bound, values)

  
def learning_rate_schedule(bound_epoch, lr, train_labels, batch_size, num_epochs):
    stpes_per_epoch = len(train_labels)//batch_size
    bound_step = [a*b for a,b in zip(bound_epoch,[stpes_per_epoch]*len(bound_epoch))]
    print("********** Learning rate schedule ***************")
    print(f'Epoch 0 ~ {bound_epoch[0]}: {lr[0]}')
    for idx in range(len(bound_epoch)-1):
        print(f'Epoch {bound_epoch[idx]} ~ {bound_epoch[idx+1]} : {lr[idx+1]}')
    print(f'Epoch {bound_epoch[-1]} ~ {num_epochs} : {lr[-1]}')
    print("*************************************************")
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(bound_step, lr)
