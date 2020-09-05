import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL
import PIL.Image

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def ad_loss(y_pred, y_true):
    return tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))


def bce_loss(y_pred, y_true):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)


def id_loss(y_pred, y_true):
    return tf.reduce_mean(tf.keras.losses.mean_absolute_error(y_true, y_pred))


def recon_loss(y_pred, y_true):
    return tf.reduce_mean(tf.keras.losses.mean_absolute_error(y_true, y_pred))


def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1 # [-1,1]
  return image

# def train_augment(image,label,img_size):
#   image = tf.image.resize(image, [img_size+30, img_size+30], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#   image = tf.image.random_crop(image, size=[img_size, img_size, 3])
#   image = tf.image.random_flip_left_right(image)
#   image = tf.cast(image, dtype=tf.float32)
#   image = normalize(image)
#   return image

#  def test_augment(image,label,img_size):
#   image = tf.image.resize(image, [img_size, img_size])
#   image = tf.cast(image, dtype=tf.float32)
#   image = normalize(image)
#   return image

def train_augment(image_file,img_size):
  image = tf.io.read_file(image_file) 
  image = tf.image.decode_jpeg(image)
  image = tf.image.resize(image, [img_size+30, img_size+30], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  image = tf.image.random_crop(image, size=[img_size, img_size, 3])
  image = tf.image.random_flip_left_right(image)
  image = tf.cast(image, dtype=tf.float32)
  image = normalize(image)
  return image


def test_augment(image_file,img_size):
  image = tf.io.read_file(image_file) 
  image = tf.image.decode_jpeg(image)
  image = tf.image.resize(image, [img_size, img_size])
  image = tf.cast(image, dtype=tf.float32)
  image = normalize(image)
  return image


def image_save(real_images, fake_images, path, iter=None, test=True):
  assert real_images.shape == fake_images.shape
  real_images = real_images * 0.5 + 0.5  # [-1,1 -> 0,1]
  fake_images = fake_images * 0.5 + 0.5
  batch_size, h, w, c = real_images.shape
  figure = np.zeros((batch_size * h ,w *2, c))
  idx = 0
  for real_img, fake_img in zip(real_images, fake_images):
    figure[h*idx:h*(idx+1), 0:w, ...] = real_img # loc [0-255, 0-255, ...]
    figure[h*idx:h*(idx+1), w: , ...] = fake_img # loc [0-255, 256-, ...]
    idx += 1
  
  suffix = '.jpg'
  
  if test == False:
      path = os.path.join(path, 'train')
      if not os.path.exists(path):
        os.makedirs(path)
      path = os.path.join(path, 'iter_' + str(iter) + suffix)
    
  else:
      path = os.path.join(path, 'test')
      if not os.path.exists(path):
        os.makedirs(path)
      path = os.path.join(path, str(iter) + suffix)
  
  plt.imsave(path ,figure)
