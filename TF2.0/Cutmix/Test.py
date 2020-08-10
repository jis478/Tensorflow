from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from ResNet import ResNet
from Functions import normalize, test_augment
import numpy as np
import argparse
import os
import time
import datetime


parser = argparse.ArgumentParser(description='Tensorflow implementation of Cutmix on CIFAR-10 / CIFAR-100 datasets')

parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='./ckpt_dir/20200810-071421/', type=str,
                    help='checkpoint to be loaded')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--dataset', dest='dataset', default='cifar100', type=str,
                    help='dataset (options: cifar10 or cifar100')
parser.add_argument('--verbose', default=1, type=int,
                    help='print training process to screen')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--tensorboard_dir', dest='tensorboard_dir', default='./tensorboard_dir/', type=str,
                    help='tensorboard summary folder')


AUTO = tf.data.experimental.AUTOTUNE
template = '******* Test top1 err: {:.3f}, Test top5 err: {:.3f}, Time: {:.3f} *******'
parser.set_defaults(bottleneck=True)


def main():
        
    args = parser.parse_args()
        
    if args.dataset == 'cifar10':
        _ , (test_images, test_labels) = datasets.cifar10.load_data()
        num_classes = 10
        height, width, channels = 32, 32, 3
    
    elif args.dataset == 'cifar100':
        _ , (test_images, test_labels) = datasets.cifar100.load_data()
        num_classes = 100
        height, width, channels = 32, 32, 3

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.map(test_augment, num_parallel_calls=AUTO)
    test_dataset = test_dataset.batch(args.batch_size)

    test_accuracy_top1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='test_accuracy_top1')
    test_accuracy_top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='test_accuracy_top5')

    # model initialization
    model = ResNet(dataset=args.dataset, depth=args.depth, num_classes=num_classes, bottleneck=args.bottleneck)     
 
    # load the latest checkpoint from the given ckpt folder
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, args.ckpt_dir, max_to_keep=1)

    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial() # optimizer not loaded
        print('checkpoint restored')
    else:
        raise Except
        print('checkpoint not restored')
   
    # test_summary_writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    test_log_dir = os.path.join(args.tensorboard_dir, current_time, 'test')
    print(f'******* Test result log written to {test_log_dir} ******')
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # training
    s_time = time.time()
              
    for image, target in test_dataset:

        image = normalize(image)
        output = model(image, training=False)
        
        test_accuracy_top1(target, output)
        test_accuracy_top5(target, output)

    test_err1 = (1-test_accuracy_top1.result())*100
    test_err5 = (1-test_accuracy_top5.result())*100

    with test_summary_writer.as_default():
        tf.summary.scalar('test_err_top1', test_err1, step=0)
        tf.summary.scalar('test_err_top5', test_err5, step=0)
                          
    if args.verbose==1:
        print(template.format(test_err1, test_err5, time.time()-s_time))          
                               
if __name__ == '__main__':
    main()
