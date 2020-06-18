# -*- coding: utf-8 -*-


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from ResNet import ResNet
from Functions import normalize, train_augment, test_augment, rand_bbox, CustomSchedule, model_save
import numpy as np
import argparse
import os
import time

parser = argparse.ArgumentParser(description='Tensorflow implementation of Cutmix on CIFAR-10 / CIFAR-100 datasets')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--beta', default=0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')

parser.set_defaults(bottleneck=True)
best_err1 = 100
best_err5 = 100
AUTO = tf.data.experimental.AUTOTUNE
template = 'Epoch: {:03d} / {:03d}:, LR: {:.10f}, Train loss: {:.3f}, Train top1 err: {:.3f}, Train top5 err: {:.3f}, Test loss: {:.3f}, Test top1 err: {:.3f}, Test top5 err: {:.3f}, Time: {:.3f}'

def main():

  global args, best_err1, best_err5
  args = parser.parse_args()

  if arg.dataset == 'cifar10':
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    num_classes = 10
  elif arg.dataset == 'cifar100':
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()
    num_classes = 100

  train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
  train_dataset = train_dataset.map(train_augment, num_parallel_calls=AUTO)
  train_dataset = train_dataset.batch(args.batch_size)
  train_dataset = train_dataset.shuffle(512)
  train_dataset = train_dataset.prefetch(AUTO)

  test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
  test_dataset = test_dataset.map(test_augment, num_parallel_calls=AUTO)
  test_dataset = test_dataset.batch(args.batch_size)

  model = ResNet(dataset=args.dataset, depth=args.depth, num_classes=num_classes, bottleneck=args.bottleneck) 

  criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  learning_rate = CustomSchedule(init_lr = args.lr)
  optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=args.momentum, nesterov=True)

  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy_top1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='train_accuracy_top1')
  train_accuracy_top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='train_accuracy_top5')

  test_loss = tf.keras.metrics.Mean(name='test_loss')
  test_accuracy_top1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='test_accuracy_top1')
  test_accuracy_top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='test_accuracy_top5')

  for epoch in range(args.epochs):

    s_time = time.time()
    train_loss.reset_states()
    train_accuracy_top1.reset_states()
    train_accuracy_top5.reset_states()

    test_loss.reset_states()
    test_accuracy_top1.reset_states()
    test_accuracy_top5.reset_states()
  
    for image, target in train_dataset: 
      
      r = np.random.rand(1)
      
      if args.beta > 0 and r < args.cutmix_prob:
          lam = np.random.beta(args.beta, args.beta)
          rand_index = tf.random.shuffle(tf.range(len(target)))
          target_a = target
          target_b = tf.gather(target, rand_index)
          bbx1, bby1, bbx2, bby2 = rand_bbox(image.shape, lam)       
          image_a = image
          image_b = tf.gather(image, rand_index)
          mask = np.ones_like(image)
          mask[:, bbx1:bbx2, bby1:bby2, :] = 0            

          image_cutmix = normalize(tf.math.multiply(image_a,mask) + tf.math.multiply(image_b, (abs(1.-mask)))) 
          image = normalize(image)
          lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image_cutmix.shape[1] * image_cutmix.shape[2]))   
          
          with tf.GradientTape() as tape:
            output = model(image_cutmix, training=True) 
            loss = criterion(target_a, output) * lam + criterion(target_b, output) * (1. - lam)
          gradients = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(gradients, model.trainable_variables)) 
      
      else:
          with tf.GradientTape() as tape:
            output = model(image, training=True)
            loss = criterion(target, output)
          gradients = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(gradients, model.trainable_variables)) 

      train_loss(loss)
      train_accuracy_top1(target, output)
      train_accuracy_top5(target, output)

    for image, target in test_dataset:
      output = model(image, training=False)
      loss = criterion(target, output)
      test_loss(loss)
      test_accuracy_top1(target, output)
      test_accuracy_top5(target, output)

    train_err1 = (1-train_accuracy_top1.result())*100
    train_err5 = (1-train_accuracy_top5.result())*100
    test_err1 = (1-test_accuracy_top1.result())*100
    test_err5 = (1-test_accuracy_top5.result())*100
  
    is_best = test_err1 <= best_err1
    best_err1 = min(test_err1, best_err1)
    if is_best:
        best_err5 = test_err5
        model_save(model, epoch, test_err1)
  
    if (epoch+1) % args.print_freq_epoch == 0:
      print(template.format(epoch+1, 
                            args.num_epochs,
                            learning_rate.return_lr(),
                            train_loss.result(),
                            train_err1,
                            train_err5,
                            test_loss.result(),
                            test_err1,
                            test_err5,
                            time.time()-s_time))
  
  print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)

if __name__ == '__main__':
    main()
