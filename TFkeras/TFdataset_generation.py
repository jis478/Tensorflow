import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
import aug
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input

def augmentation(image,
                 flip_up_down = False,
                 flip_left_right = False,
                 random_crop = False,
                 random_noise_prob=0.00,
                 random_brightness_prob=0.5,         
                 random_brightness_max_delta=0.9,    # [0, 1) 
                 random_hue_prob=0.5,
                 random_hue_max_delta=0.5,           # [0, 0.5)]
                 random_saturation_prob=0.5,
                 random_saturation_lower=0.2,
                 random_saturation_upper=5.0,
                 ):

    if random_noise_prob != 0:
        image = aug.random_noise(image, prob_noise=random_noise_prob)

    if random_brightness_prob != 0:
        image = aug.random_brightness(
            image,
            prob_brightness=random_brightness_prob,
            max_delta=random_brightness_max_delta
        )

    if random_hue_prob != 0:
        image = aug.random_hue(
            image,
            prob_hue=random_hue_prob,
            max_delta=random_hue_max_delta
        )

    if random_saturation_prob != 0:
        image = aug.random_saturation(
            image,
            prob_saturation=random_saturation_prob,
            lower=random_saturation_lower,
            upper=random_saturation_upper
        )

    return image

def _read_from_tfrecord(example_proto, x_dim, y_dim, channels, is_training):
        tfrecord_features = tf.parse_single_example(example_proto,
                                                    features={
                                                        'image': tf.FixedLenFeature([], tf.string),
                                                        'label': tf.FixedLenFeature([], tf.int64),
                                                    }, name='features')

        image = tf.decode_raw(tfrecord_features['image'], tf.float32) 
        image = tf.reshape(image, [x_dim, y_dim, channels])
        if is_training:
            image = augmentation(image)
        label = tfrecord_features['label']
        label = tf.one_hot(label, 2)

        return image, label
    
def tfdata_generator(tfrecord_path, buffer_size, batch_size, x_dim, y_dim, channels, is_training):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(lambda x: _read_from_tfrecord(x, x_dim , y_dim, channels, is_training)).repeat()
    dataset = dataset.shuffle(buffer_size=(int(buffer_size) * 2 + 3 * batch_size))
    dataset = dataset.batch(batch_size)
    if is_training:
        dataset = dataset.shuffle(buffer_size) 
    dataset = dataset.prefetch(buffer_size=batch_size)
    return dataset