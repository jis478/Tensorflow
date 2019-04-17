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
