
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
  
# Downsampling class 정의
class Downsampling_Part(tf.keras.Model):
  def __init__(self):
    super(Downsampling_Part, self).__init__()
    self.conv1 = Conv2D(64, kernel_size = 7, strides = 1, padding = 'SAME')
    self.conv2 = Conv2D(128, kernel_size = 4, strides = 2, padding = 'SAME')
    self.conv3 = Conv2D(256, kernel_size = 4, strides = 2, padding = 'SAME')
    self.activation = ReLU()
    
  def call(self, images, labels):
    x = input_merge(images,labels)
    x = self.conv1(x)
    x = IN(x)
    x = self.activation(x)
    assert x.shape == (2, 128, 128, 64) 

    x = self.conv2(x)
    x = IN(x)
    x = self.activation(x)
    assert x.shape == (2, 64, 64, 128) 
    
    x = self.conv3(x)
    x = IN(x)
    x = self.activation(x)
    assert x.shape == (2, 32, 32, 256) 
    return x
 
# ResidualBlock class 정의
class ResidualBlock_Part(tf.keras.Model):
  def __init__(self, kernel_size=3, filters=256, strides=1, padding='SAME'):
    super(ResidualBlock_Part, self).__init__()
    self.conv = Conv2D(filters, kernel_size, strides, padding)
    self.activation_ReLU = ReLU()

  def call(self, input_tensor):
    x = input_tensor
    x_list = []
    for i in range(6):
      x_list.append(x)
      x_list[i] = self.conv(x_list[i])
      x_list[i] = IN(x_list[i])
      if i == 0:
        x_list[i] += input_tensor
      else:
        x_list[i] += x_list[i-1]
      x_list[i] = self.activation_ReLU(x_list[i])
    return x_list[-1] 
  
  
# Upsampling class 정의
class Upsampling_Part(tf.keras.Model):
  def __init__(self):
    super(Upsampling_Part, self).__init__()
    self.deconv1 = Conv2DTranspose(128, kernel_size = 4, strides = 2, padding = 'SAME')
    self.deconv2 = Conv2DTranspose(64, kernel_size = 4, strides = 2, padding = 'SAME')
    self.conv1 = Conv2D(3, kernel_size = 7, strides = 1, padding = 'SAME')
    self.activation_ReLU = ReLU()
    
  def call(self, x):
    x = self.deconv1(x)
    x = IN(x)
    x = self.activation_ReLU(x)
    assert x.shape == (2, 64, 64, 128) 

    x = self.deconv2(x)
    x = IN(x)
    x = self.activation_ReLU(x)
    assert x.shape == (2, 128, 128, 64) 
    
    x = self.conv1(x)
    x = tanh(x)
    assert x.shape == (2, 128, 128, 3) 
    return x
  
# 최종 Generator class 정의
class Build_Generator(tf.keras.Model):
  def __init__(self):
    super(Build_Generator, self).__init__()
    self.Downsampling = Downsampling_Part()
    self.ResidualBlock = ResidualBlock_Part()
    self.Upsampling = Upsampling_Part()
    
  def call(self, images, labels):
    x = self.Downsampling(images, labels)
    print("The shape of the input tensor after downsampling :", x.shape)
    x = self.ResidualBlock(x)
    print("The shape of the input tensor after Bottleneck :", x.shape)
    x = self.Upsampling(x)
    print("The shape of the input tensor after upsampling :", x.shape)
    return x
 
######## Generator 수행 ##################
# 데이터 생성
N = 1000
image_size = 128
image_data = np.random.normal(size=[N, image_size, image_size, 3])

label_1 = np.random.uniform(low=0., high=3., size=N).astype(np.int32) # facial expression attributes
label_1 = tf.one_hot(label_1, depth=3)
label_2 = np.random.randint(2, size=(N,2)) # male, young attributes
label =  np.concatenate([label_1, label_2], axis=-1)

train_dataset = tf.data.Dataset.from_tensor_slices((image_data,label))
train_dataset = train_dataset.batch(2) 

# Generator 수행
Generator = Build_Generator()
result = Generator(images, labels)
print(result.shape)
