#!/usr/bin/env python
# coding: utf-8

# ## Library loading
# 

# In[15]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np 
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.layers import Dense, Flatten, Conv2D, ReLU, Conv2DTranspose, LeakyReLU, Layer, ZeroPadding2D
from tensorflow.keras.activations import tanh
import time
from IPython import display
import matplotlib.pyplot as plt

print(tf.__version__)
tf.enable_eager_execution()


# ## Auxiliary functions

# In[16]:


def data_preprocessing(folder,attr_csv, num, img_size): 
    ''' data preprocessing'''
    domain_list = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    list_attr_celeba = pd.read_csv(attr_csv)
    list_attr_celeba = list_attr_celeba.loc[(list_attr_celeba['Black_Hair'] == 1) | (list_attr_celeba['Blond_Hair'] == 1) | (list_attr_celeba['Brown_Hair'] == 1), domain_list]
    list_attr_celeba = list_attr_celeba.replace({-1:0})
    list_attr_celeba = list_attr_celeba.loc[list_attr_celeba.apply(lambda x: x['Black_Hair'] + x['Blond_Hair'] + x['Brown_Hair'], axis=1) == 1]
    list_attr_celeba = list_attr_celeba.sample(n=num,replace=False)
    idx = list(list_attr_celeba.index)
    list_attr_celeba = np.array(list_attr_celeba)

    img_list = glob.glob(os.path.join(folder, '*'))
    X_train = np.zeros((num, img_size, img_size, 3))
    for i, img_idx in enumerate(idx):
        X_train[i] = (np.array(Image.open(img_list[img_idx]).resize((size,size))) - 127.5) / 127.5

    print("{} training image loaded ".format(len(X_train)))
    print("{} x {} original domain dataset loaded".format(list_attr_celeba.shape[0], list_attr_celeba.shape[1]))
    
    return X_train, list_attr_celeba

def dataset_split(X_train, y_train, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size)
    print("Number of train images: {}".format(len(X_train)))
    print("Number of train original domain labels: {}".format(len(y_train)))
    print("Number of test images: {}".format(len(X_test)))
    print("Number of test original domain labels: {}".format(len(y_test)))
    return X_train, X_test, y_train, y_test

def _preprocessing(image, domain, train=True):
    ''' image flip & domain reshape'''
    domain = tf.reshape(domain, shape=(-1,1, 1, tf.shape(domain)[-1])) 
    if train:
        image = tf.image.random_flip_left_right(image)     # 50% flip logic is already embedded in tf.image.random_flip_left_right
    return image,domain

def random_target_domain_generation(batch_size):
  ''' target domain generation '''
  target_domain_1 = np.random.uniform(low=0., high=3., size=batch_size).astype(np.int32) # facial expression attributes
  target_domain_1 = tf.one_hot(target_domain_1, depth=3)
  target_domain_2 = np.random.randint(2, size=(batch_size,2)) # male, young attributes
  target_domain =  np.concatenate([target_domain_1, target_domain_2], axis=-1)
  target_domain = target_domain.reshape((target_domain.shape[0],1,1,target_domain.shape[1]))
  return target_domain
            
def generate_and_save_images(model, step, test_dataset):
  ''' generated image saving'''

  test_image, _ = next(iter(test_dataset))
  target_domain = random_target_domain_generation(tf.shape(test_image)[0])
  predictions = model(test_image, target_domain)
  fig = plt.figure(figsize=(20,20))
  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(predictions[i, :, :, :] * 127.5 + 127.5)
    plt.axis('off')        
  plt.savefig('image_at_step_{:04d}.png'.format(step))
  plt.show()
  
def input_merge(images, domain):
  ''' input image and domain merge'''
  batch_size = images.shape[0]
  image_size = images.shape[1] 
  channels = domain.shape[-1] 
  domain = np.squeeze(domain, axis=(1,2))  
  merged = np.zeros([batch_size,image_size,image_size,channels])
  for batch in range(batch_size):
    temp = tf.broadcast_to(domain[batch], [image_size,image_size,channels])
    merged[batch] = temp
  merged = tf.concat([images, merged], axis=-1)
  return merged  


class InstanceNormalization(tf.keras.layers.Layer):
  '''InstanceNormalization for only 4-rank Tensor (image data)'''

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    shape = tf.TensorShape(input_shape)
    param_shape = shape[-1]
    self.gamma = self.add_weight(name='gamma',
                                 shape=param_shape,
                                 initializer='ones',
                                 trainable=True)
    self.beta = self.add_weight(name='beta',
                                shape=param_shape,
                                initializer='zeros',
                                trainable=True)
    super(InstanceNormalization, self).build(input_shape)

  def call(self, inputs):
    input_shape = inputs.get_shape()
    HW = int(input_shape[1])*int(input_shape[2]) 
    u_ti = 1./HW*np.sum(inputs, axis=(1,2))   
    for _ in range(2):
      u_ti = np.stack((u_ti, )*input_shape[1], axis=-1) 
    u_ti = np.swapaxes(u_ti,1,3) 
    var_ti = 1/HW*np.sum((inputs - u_ti), axis=(1,2))**2
    for _ in range(2):
      var_ti = np.stack((var_ti, )*input_shape[1], axis=-1) 
    var_ti = np.swapaxes(var_ti,1,3)                    
    y_tijk = (inputs - u_ti) / np.sqrt(var_ti +  self.epsilon)  
    return self.gamma * y_tijk + self.beta


# ## Network structures

# In[17]:


class Downsampling_Part(tf.keras.Model):
  ''' Downsampling part of Generator'''
  
  def __init__(self):
    super(Downsampling_Part, self).__init__()
    self.conv1 = Conv2D(64, kernel_size = 7, strides = 1, padding = 'valid')
    self.conv2 = Conv2D(128, kernel_size = 4, strides = 2, padding = 'valid')
    self.conv3 = Conv2D(256, kernel_size = 4, strides = 2, padding = 'valid')
    self.zeropadding1 = ZeroPadding2D(3)
    self.zeropadding2 = ZeroPadding2D(1)
    self.zeropadding3 = ZeroPadding2D(1)
    self.in1 = InstanceNormalization()
    self.in2 = InstanceNormalization()
    self.in3 = InstanceNormalization()
    self.activation_ReLU = ReLU()
  
  def call(self, images, labels):
    x = input_merge(images,labels)
    x = self.zeropadding1(x)
    x = self.conv1(x)
    x = self.in1(x)
    x = self.activation_ReLU(x)

    x = self.zeropadding2(x)
    x = self.conv2(x)
    x = self.in2(x)
    x = self.activation_ReLU(x)

    x = self.zeropadding3(x)
    x = self.conv3(x)
    x = self.in3(x)
    x = self.activation_ReLU(x)
    return x
  

class ResnetIdentityBlock(tf.keras.Model):
  ''' ResentIdentityBlock for Residual part of Generator'''
  
  def __init__(self):
    super(ResnetIdentityBlock, self).__init__()
    self.conv1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='valid')
    self.conv2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='valid')
    self.zeropadding = ZeroPadding2D(1)
    self.in1 = InstanceNormalization()
    self.activation_ReLU = ReLU()

  def call(self, input_tensor):
    x = self.zeropadding(input_tensor)
    x = self.conv1(x)
    x = self.in(x)
    x = self.activation_ReLU(x)
    x = self.zeropadding(x)
    x = self.conv2(x)
    x += input_tensor
    x = self.activation_ReLU(x)
    return x

  
class Bottleneck_Part(tf.keras.Model):
  ''' Bottleneck part of Generator'''
  def __init__(self):
    super(Bottleneck_Part, self).__init__()
    self.ResnetIdentityBlock = ResnetIdentityBlock()
    
  def call(self, input_tensor):
    x  = self.ResnetIdentityBlock(input_tensor)
    x  = self.ResnetIdentityBlock(x)
    x  = self.ResnetIdentityBlock(x)
    x  = self.ResnetIdentityBlock(x)
    x  = self.ResnetIdentityBlock(x)
    x  = self.ResnetIdentityBlock(x)       
    return x
  

class Upsampling_Part(tf.keras.Model):
  ''' Upsampling part of Generator'''
  
  def __init__(self):
    super(Upsampling_Part, self).__init__()
    self.deconv1 = Conv2DTranspose(128, kernel_size = 4, strides = 2, padding = 'same')
    self.deconv2 = Conv2DTranspose(64, kernel_size = 4, strides = 2, padding = 'same')
    self.conv1 = Conv2D(3, kernel_size = 7, strides = 1, padding = 'same')
    self.zeropadding1 = ZeroPadding2D(1)
    self.zeropadding2 = ZeroPadding2D(1)
    self.zeropadding3 = ZeroPadding2D(3)
    self.activation_ReLU = ReLU()
    self.in1 = InstanceNormalization()
    self.in2 = InstanceNormalization()

    
  def call(self, x):
    x = self.deconv1(x)
    x = self.in1(x)
    x = self.activation_ReLU(x)

    x = self.deconv2(x)
    x = self.in2(x)
    x = self.activation_ReLU(x)

    x = self.conv1(x)
    x = tanh(x)
    return x
  
  
class Build_generator(tf.keras.Model):
  ''' Building a generator'''
  def __init__(self):
    super(Build_generator, self).__init__()
    self.Downsampling = Downsampling_Part()
    self.ResidualBlock = Bottleneck_Part()
    self.Upsampling = Upsampling_Part()
    
  def call(self, images, labels):
    x = self.Downsampling(images, labels)
    x = self.ResidualBlock(x)
    x = self.Upsampling(x)
    return x
 

class Build_discriminator(tf.keras.Model):
  ''' Building a discriminator'''
  
  def __init__(self, image_size, nd):
    super(Build_discriminator, self).__init__()
    self.conv1 = Conv2D(64, kernel_size = 4, strides = 2, padding = 'valid')
    self.conv2 = Conv2D(128, kernel_size = 4, strides = 2, padding = 'valid')
    self.conv3 = Conv2D(256, kernel_size = 4, strides = 2, padding = 'valid')
    self.conv4 = Conv2D(512, kernel_size = 4, strides = 2, padding = 'valid')
    self.conv5 = Conv2D(1024, kernel_size = 4, strides = 2, padding = 'valid')
    self.conv6 = Conv2D(2048, kernel_size = 4, strides = 2, padding = 'valid')    
    self.conv7_1 = Conv2D(1, kernel_size = 3, strides = 1)    
    self.conv7_2 = Conv2D(nd, kernel_size = int(image_size/64), strides = 1)    
    self.zeropadding0 = ZeroPadding2D(0) 
    self.zeropadding = ZeroPadding2D(1)
    self.activation_LeakyReLU = LeakyReLU(alpha=0.01)
    
  def call(self, x):
    x = self.zeropadding(x)
    x = self.conv1(x)
    x = self.activation_LeakyReLU(x)

    x = self.zeropadding(x)
    x = self.conv2(x)
    x = self.activation_LeakyReLU(x)
    
    x = self.zeropadding(x)
    x = self.conv3(x)
    x = self.activation_LeakyReLU(x)
    
    x = self.zeropadding(x)
    x = self.conv4(x)
    x = self.activation_LeakyReLU(x)

    x = self.zeropadding(x)
    x = self.conv5(x)
    x = self.activation_LeakyReLU(x)

    x = self.zeropadding(x)
    x = self.conv6(x)
    x = self.activation_LeakyReLU(x)
    
    x_src = self.zeropadding(x)
    D_src = self.conv7_1(x_src)
    D_cls = self.conv7_2(x)

    return  D_src, D_cls


# ## Loss

# In[18]:


### Loss
def adverserial_loss(logits, real=True):
  if real == True:
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(logits),logits = logits)
  else:
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.zeros_like(logits),logits = logits)
  return loss

def reconstruction_loss(image,rec_image):
  return lambda_rec * tf.reduce_mean(np.abs(image  - rec_image))
 
def domain_cls_loss(domain, logits):
  return lambda_cls * tf.losses.sigmoid_cross_entropy(multi_class_labels = domain, logits = logits)

def w_adverserial_loss(logits, real=True):
  loss = tf.reduce_mean(logits)
  if real == True:
    loss = -loss
  return loss

## G_loss, D_loss
def G_loss(fake_D_src, target_D_cls, target_domain,input_image, reconstructed_image, lambda_cls,lambda_rec):
  loss = w_adverserial_loss(fake_D_src, real=True) + lambda_cls * domain_cls_loss(target_domain, target_D_cls) + lambda_rec * reconstruction_loss(input_image, reconstructed_image)
  return loss
  
def D_loss(real_D_src, fake_D_src, original_domain, original_D_cls, lambda_cls):
  loss = w_adverserial_loss(real_D_src, real=True) + w_adverserial_loss(fake_D_src, real=False) + lambda_cls* domain_cls_loss(original_domain,original_D_cls)
  return loss


# ## Optimizer

# In[19]:


generator_optimizer = tf.train.AdamOptimizer(0.0001)
discriminator_optimizer = tf.train.AdamOptimizer(0.0001)


# ## Train loop (training happens here)

# In[4]:


def train_step(input_image, original_domain, target_domain, step, i):

# fake_image (image) # generator: generated fake images
# fake_D_src (logits) # discriminator: real / fake image classification for fake image
# target_D_cls (logits) # discriminator: original / target label classification for fake image
# real_D_src (logits) # discriminator: real / fake image classificiaion for real image
# original_D_cls (logits) # discriminator: original / target label classificiaion for real image
# fake_D_src (logits) # discriminator: original/target classification for fake image
# reconstructed_image (image) # generator: generated real (reconstructed) images 


  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
    fake_image = generator(input_image, target_domain)  # step(b)
    fake_D_src, target_D_cls = discriminator(fake_image)  # step(d) 
    reconstructed_image = generator(fake_image, original_domain) # step(c)        
    real_D_src, original_D_cls = discriminator(input_image) #step(a) 
 
    discriminator_loss = D_loss(real_D_src, fake_D_src, original_domain, original_D_cls, lambda_cls)
    generator_loss = G_loss(fake_D_src, target_D_cls, target_domain,input_image, reconstructed_image, lambda_cls,lambda_rec)
 
    # discriminator 5번 우선 업데이트  
  gradients_of_discriminator = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
  print ("Step %d-%d [D loss: %f]" % (step+1, i+1, discriminator_loss))   

  if i == 0: # 처음 한번만 generator 업데이트
    gradients_of_generator = gen_tape.gradient(generator_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    print ("Step %d [G loss: %f]" % (step+1, generator_loss))    

def train(train_dataset, steps, batch_size,discriminator_updates):
    
  for step in range(steps):
    
    start = time.time()

    for i in range(discriminator_updates):
        input_image, original_domain = next(iter(train_dataset))
        target_domain = random_target_domain_generation(batch_size)
        train_step(input_image, original_domain, target_domain, step, i)  
        
    generate_and_save_images(generator,
                             step + 1,
                             test_dataset)

    if (step + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for step {} is {} sec'.format(step + 1, time.time()-start))

  generate_and_save_images(generator,
                           step,
                           test_dataset)


# ## Execution

# #### Dataset prepration

# In[22]:


attr_csv = '/home/Markkim/Celeb/list_attr_celeba.csv'
folder = '/home/Markkim/Celeb/img_align_celeba/'

num = 10
size = 128

X_train, original_domain = data_preprocessing(folder, attr_csv, num, size)
X_train, X_test, original_domain_train, original_domain_test = dataset_split(X_train, original_domain, test_size=0.2)


# #### tf.dataset generation
# 

# In[23]:


batch_size = 2
test_batch_size = 4

train_dataset = tf.data.Dataset.from_tensor_slices((X_train,original_domain_train))
train_dataset = train_dataset.batch(batch_size) 
train_dataset = train_dataset.map(lambda x, y: _preprocessing(x, y, train=True))

test_dataset = tf.data.Dataset.from_tensor_slices((X_test,original_domain_test))
test_dataset = test_dataset.batch(test_batch_size) 
test_dataset = test_dataset.map(lambda x, y: _preprocessing(x, y, train=False))


# #### model building

# In[24]:


generator = Build_generator()
discriminator = Build_discriminator(size, original_domain_train.shape[-1])


# ##### checkpoint

# In[25]:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator)


# #### Training

# In[5]:


steps = 20
lambda_cls = 1
lambda_rec = 10
lambda_gp = 10
train(train_dataset, steps, batch_size, discriminator_updates=5)


# In[ ]:




