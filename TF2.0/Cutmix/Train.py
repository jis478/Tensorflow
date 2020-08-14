# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from ResNet import ResNet
from Functions import normalize, train_augment, test_augment, rand_bbox, learning_rate_schedule
import numpy as np
import argparse
import os
import time
import datetime

parser = argparse.ArgumentParser(description='Tensorflow implementation of Cutmix on CIFAR-10 / CIFAR-100 datasets')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print_freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--layers', default=[3, 4, 6, 3], type=list,
                    help='resnet structure (eg. resnet50: [3,4,6,3]) (default: [3,4,6,3])')
parser.add_argument('--dataset', dest='dataset', default='cifar100', type=str,
                    help='dataset (options: cifar10 or cifar100)')
parser.add_argument('--beta', default=0.0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0.5, type=float,
                    help='cutmix probability')
parser.add_argument('--lr_boundaries', default=[60, 120, 160], type=list, 
                    help='epochs for piecewise learning rate decay boundaries')
parser.add_argument('--lr_values', default=[0.1, 0.02, 0.004, 0.0008], type=list, 
                    help='learning rates for corresponding boundaries')
parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='./ckpt_dir/', type=str,
                    help='checkpoint saving folder')
parser.add_argument('--tensorboard_dir', dest='tensorboard_dir', default='./tensorboard_dir/', type=str,
                    help='tensorboard summary folder')
parser.add_argument('--verbose', default=1, type=int,
                    help='print training process to screen')

parser.set_defaults(bottleneck=True)
AUTO = tf.data.experimental.AUTOTUNE
template = 'Epoch: {:03d} / {:03d}:, LR: {:.10f}, Train loss: {:.3f}, Train top1 err: {:.3f}, Train top5 err: {:.3f}, Test loss: {:.3f}, Test top1 err: {:.3f}, Test top5 err: {:.3f}, Time: {:.3f}'


# main
def main():
        
  @tf.function
  def train_cutmix_image(image_cutmix, target_a, target_b, lam):
    with tf.GradientTape() as tape:
      output = model(image_cutmix, training=True) 
      loss = criterion(target_a, output) * lam + criterion(target_b, output) * (1. - lam)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) 
    return loss, output


  @tf.function
  def train_original_image(image, target):
    with tf.GradientTape() as tape:
      output = model(image, training=True)
      loss = criterion(target, output)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) 
    return loss, output


  args = parser.parse_args()
  assert len(args.lr_boundaries)+1 == len(args.lr_values), "Epochs and Learning rates are not matched"

  if args.dataset == 'cifar10':
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    num_classes = 10
    
  elif args.dataset == 'cifar100':
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()
    num_classes = 100

  # dataset 
  train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
  train_dataset = train_dataset.shuffle(int(len(train_labels)/4))
  train_dataset = train_dataset.map(train_augment, num_parallel_calls=AUTO)
  train_dataset = train_dataset.batch(args.batch_size)
  train_dataset = train_dataset.prefetch(AUTO)

  test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
  test_dataset = test_dataset.map(test_augment, num_parallel_calls=AUTO)
  test_dataset = test_dataset.batch(args.batch_size)

  # model building (ResNet50)
  model = ResNet(num_blocks=args.layers, num_classes=num_classes, bottleneck=args.bottleneck) 
  
  # loss & optimizer
  criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  lr_schedule = learning_rate_schedule(args.lr_boundaries, args.lr_values, train_labels, args.batch_size, args.epochs)    
  optimizer = tf.keras.optimizers.SGD(lr_schedule, momentum=args.momentum, nesterov=True)
    
  # ckpt manager
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  ckpt_dir = os.path.join(args.ckpt_dir, current_time)
  ckpt = tf.train.Checkpoint(model=model)
  ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)

  # metrics 
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy_top1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='train_accuracy_top1')
  train_accuracy_top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='train_accuracy_top5')

  test_loss = tf.keras.metrics.Mean(name='test_loss')
  test_accuracy_top1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='test_accuracy_top1')
  test_accuracy_top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='test_accuracy_top5')

  # tensorboard writer 
  train_log_dir = os.path.join(args.tensorboard_dir, current_time, 'train')
  train_summary_writer = tf.summary.create_file_writer(train_log_dir)
  print(f'******* Train result log written to {train_log_dir} ******')
    
  test_log_dir = os.path.join(args.tensorboard_dir, current_time, 'test')
  test_summary_writer = tf.summary.create_file_writer(test_log_dir)
  print(f'******* Test result log written to {test_log_dir} ******')
 
  template = '******* Epoch: {:03d} / {:03d}:, LR: {:.3f}, Train loss: {:.3f}, Train top1 err: {:.3f}, Train top5 err: {:.3f}, Test loss: {:.3f}, Test top1 err: {:.3f}, Test top5 err: {:.3f}, Time: {:.3f}'
    
  # custom training
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
      
      # training on cutmix image 
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
          lam = tf.convert_to_tensor(1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image_cutmix.shape[1] * image_cutmix.shape[2])), dtype=tf.float32)
          loss, output = train_cutmix_image(image_cutmix, target_a, target_b, lam)      
         
      # training on original image
      else:
          image = normalize(image)
          loss, output = train_original_image(image, target)

      train_loss(loss)
      train_accuracy_top1(target, output)
      train_accuracy_top5(target, output)

    for image, target in test_dataset:
      image = normalize(image)
      output = model(image, training=False)
      loss = criterion(target, output)
      test_loss(loss)
      test_accuracy_top1(target, output)
      test_accuracy_top5(target, output)


    train_err1 = (1-train_accuracy_top1.result())*100
    train_err5 = (1-train_accuracy_top5.result())*100
    test_err1 = (1-test_accuracy_top1.result())*100
    test_err5 = (1-test_accuracy_top5.result())*100
      
    # writing to tensorboard
    with train_summary_writer.as_default():
      tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
      tf.summary.scalar('train_err_top1', train_err1, step=epoch)
      tf.summary.scalar('train_err_top5', train_err5, step=epoch)

    with test_summary_writer.as_default():
      tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
      tf.summary.scalar('test_err_top1', test_err1, step=epoch)
      tf.summary.scalar('test_err_top5', test_err5, step=epoch)

    # print metrics to screen
    if ((epoch+1) % args.print_freq == 0) & (args.verbose==1):
      print(template.format(epoch+1, 
                            args.epochs,
                            optimizer.learning_rate(optimizer.iterations).numpy(),
                            train_loss.result(),
                            train_err1,
                            train_err5,
                            test_loss.result(),
                            test_err1,  
                            test_err5,
                            time.time()-s_time))
 
#     # saving only the best model 
#     is_best = test_err1 <= best_err1
#     best_err1 = min(test_err1, best_err1)
#     if is_best:
#       print('Model being saved...')  
#       manager.save(checkpoint_number=optimizer.iterations)
#       print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)

  # save only the last epoch model
  ckpt_manager.save(checkpoint_number=optimizer.iterations)
  print(f'******* Model saved to {ckpt_dir} ******') 
  print(f'******* Test top-1 err: {test_err1}  top-5 err: {test_err5}')


if __name__ == '__main__':
    main()

    
    
