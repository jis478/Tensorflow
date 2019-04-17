
# 라이브러리 로딩
import tensorflow as tf
import numpy as np 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, ReLU, Conv2DTranspose
from tensorflow.keras.activations import tanh

######## Class, 보조 함수 정의 ##################

# InstanceNormalization function (https://arxiv.org/abs/1607.08022) 정의
def IN(images):
  HW = images.shape[1]*images.shape[2] 

  u_ti = 1/HW*np.sum(images, axis=(1,2))   # 2x3  
  for _ in range(2):
    u_ti = np.stack((u_ti, )*images.shape[1], axis=-1) # 2x3x128x128
  u_ti = np.swapaxes(u_ti,1,3) # 2x128x128x3  

  var_ti = 1/HW*np.sum((images - u_ti), axis=(1,2))**2 # 2x3  
  for _ in range(2):
    var_ti = np.stack((var_ti, )*images.shape[1], axis=-1) # 2x3x128x128
  var_ti = np.swapaxes(var_ti,1,3) # 2x128x128x3                      

  y_tijk = (images - u_ti) / np.sqrt(var_ti + np.finfo(np.float32).eps)  # 2x128x128x3
  return y_tijk   

# Input merge function (이미지와 label 정보 합침) 정의
def input_merge(images, labels):
  batch_size = images.shape[0]
  image_size = images.shape[1]
  channels = labels.shape[1]
  merged = np.zeros([batch_size,image_size,image_size,channels])
  for batch in range(batch_size):
    temp = tf.broadcast_to(labels[batch], [image_size,image_size,channels])
    merged[batch] = temp
  merged = tf.concat([images, merged], axis=-1)
  return merged  
  
