#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math

def _get_image_depth(x): 
    return x.get_shape()[2].value

def _get_image_size(x):
    return x.get_shape()[1].value

def random_noise_fn(image):
    noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=1.0, dtype=tf.float32)
    image = tf.add(image, noise)
     
    return image

def random_rot(image,lower, upper):
    angle = tf.random_uniform([1], np.deg2rad(lower), np.deg2rad(upper))
    image = tf.contrib.image.rotate(image, angle)
     
    return image

def random_shift(left_right, up_down):
    shift_x = tf.random_uniform([1], -1 * left_right, left_right)
    shift_y = tf.random_uniform([1], -1 * up_down, up_down)
 
    shift_row1 = tf.concat([tf.ones([1]), tf.zeros([1]), shift_x], axis=0)
    shift_row2 = tf.concat([tf.zeros([1]), tf.ones([1]), shift_y], axis=0)
    shift_row3 = tf.concat([tf.zeros([2]), tf.ones([1])], axis=0)
    shift_matrix = tf.stack([shift_row1, shift_row2, shift_row3])
 
    return shift_matrix

def random_shear(lower=-5.0, upper=5.0):
    shear_angle = tf.random_uniform([1], np.deg2rad(lower), np.deg2rad(upper))
 
    shear_row1 = tf.concat([tf.ones([1]), tf.negative(tf.sin(shear_angle)), tf.zeros([1])], axis=0)
    shear_row2 = tf.concat([tf.zeros([1]), tf.cos(shear_angle), tf.zeros([1])], axis=0)
    shear_row3 = tf.concat([tf.zeros([2]), tf.ones([1])], axis=0)
    shear_matrix = tf.stack([shear_row1, shear_row2, shear_row3])
 
    return shear_matrix

def random_zoom(lower=0.95,upper=1.05):
    zoom = tf.random_uniform([1], 1/upper, 1/lower)
 
    zoom_row1 = tf.concat([zoom, tf.zeros([2])], axis=0)
    zoom_row2 = tf.concat([tf.zeros([1]), zoom, tf.zeros([1])], axis=0)
    zoom_row3 = tf.concat([tf.zeros([2]), tf.ones([1])], axis=0)
    zoom_matrix = tf.stack([zoom_row1, zoom_row2, zoom_row3])
 
    return zoom_matrix

def random_gamma_fn(image, lower=0.8, upper=1.0):
    gamma = tf.random_uniform([1], lower, upper)
    
    image = tf.image.adjust_gamma(image, gamma[0], 1.0)
    return image

def flip_up_down(image):
    image = tf.image.random_flip_up_down(image)
    return image

def flip_left_right(image):
    image = tf.image.random_flip_left_right(image)
     
    return image

def random_brightness(
    image,
    prob_brightness=0.5,
    max_delta=1,
    ):
    random_value = tf.random_uniform([1], 0, 1)
    image = tf.cond(random_value[0] < prob_brightness,
                    lambda: tf.image.random_brightness(image, max_delta),
                    lambda: image)
     
    return image

def random_contrast(
    image,
    prob_contrast=0.5,
    lower=0.9,
    upper=1.1
    ):
    random_value = tf.random_uniform([1], 0, 1)
    image = tf.cond(random_value[0] < prob_contrast,
                    lambda: tf.image.random_contrast(image, lower, upper),
                    lambda: image)
     
    return image


def random_hue(
    image,
    prob_hue=0.5,
    max_delta=0.5
    ):
    random_value = tf.random_uniform([1], 0, 1)
    image = tf.cond(random_value[0] < prob_hue,
                    lambda: tf.image.random_hue(image, max_delta),
                    lambda: image)
     
    return image

def random_gamma(
    image,
    prob_gamma=0.5,
    lower=0.8,
    upper=1.0
    ): 
    random_value = tf.random_uniform([1], 0, 1)
    image = tf.cond(random_value[0] < prob_gamma,
                    lambda: random_gamma_fn(image,lower,upper),
                    lambda: image)
     
    return image

def random_saturation(
    image,
    prob_saturation=0.5,
    lower=0.0,
    upper=10.0
    ):
    random_value = tf.random_uniform([1], 0, 1)
    image = tf.cond(random_value[0] < prob_saturation,
                    lambda: tf.image.random_saturation(image, lower, upper),
                    lambda: image)
     
    return image

def random_noise(
    image,
    prob_noise=0.5
    ):
    random_value = tf.random_uniform([1], 0, 1)
    image = tf.cond(random_value[0] < prob_noise,
                    lambda: random_noise_fn(image),
                    lambda: image)
     
    return image

def random_affine(
    image,
    prob_shift=0.5,
    prob_shear=0.5,
    prob_zoom=0.5,
    prob_rot=0.5,
    shift_left_right=10.0,
    shift_up_down=10.0,
    shear_lower=-5.0,
    shear_upper=5.0,
    zoom_lower=0.95,
    zoom_upper=1.05,
    rot_lower=-3,
    rot_upper=3,
    padding=False
     ):
    
    img_size = _get_image_size(image)
    pad_size = int(img_size/4)

    if (padding == True):
        image = tf.pad(image, [[pad_size, pad_size], [pad_size, pad_size], [0, 0]], "REFLECT")

    random_value = tf.random_uniform([4], 0, 1) 
    ## transform_matrix끼리 먼저 곱한 후 transform
    transform_matrix = tf.cond(random_value[0] < prob_shift,
                               lambda: random_shift(shift_left_right, shift_up_down),
                               lambda: tf.eye(3))
    transform_matrix = tf.cond(random_value[1] < prob_shear,
                               lambda: tf.matmul(transform_matrix, random_shear(shear_lower, shear_upper)),
                               lambda: transform_matrix)
    transform_matrix = tf.cond(random_value[2] < prob_zoom,
                               lambda: tf.matmul(transform_matrix, random_zoom(zoom_lower, zoom_upper)),
                               lambda: transform_matrix)
     
    ## tf.contrib.image.transform은 3x3 matrix중 마지막 고정값 1 제외 8개를 input으로 받음
    image = tf.cond((random_value[0] < prob_shift) | (random_value[1] < prob_shear) | (random_value[2] < prob_zoom),
                    lambda: tf.contrib.image.transform(
                            image,
                            tf.gather_nd(transform_matrix, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1]])),
                    lambda: image)
     
    image = tf.cond(random_value[3] < prob_rot,
                    lambda: random_rot(image,rot_lower,rot_upper),
                    lambda: image)
     
    if (padding == True):
        image = tf.slice(image, [pad_size, pad_size, 0], [img_size, img_size, _get_image_depth(image)])   
     
    return image


def random_crop(image, crop_size):
    image = tf.random_crop(image,[crop_size,crop_size,_get_image_depth(image)])
    return image

def central_crop(image, crop_size):
    image = tf.image.central_crop(image,float(crop_size)/_get_image_size(image))
    image = tf.image.resize_images(image,[crop_size,crop_size])
    return image

def patch(image) :
    size = image.get_shape()[1].value
    crop = tf.image.central_crop(image,0.5)
    resize = tf.image.resize_images(crop,[size,size])
    aug = tf.concat([image,resize],axis=1)
    return aug
