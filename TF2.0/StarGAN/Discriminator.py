###### Discriminator 
# Discriminator class 정의
class Build_discriminator(tf.keras.Model):
  def __init__(self, image_size, label_1):
    super(Build_discriminator, self).__init__()
    self.conv1 = Conv2D(64, kernel_size = 4, strides = 2, padding = 'SAME')
    self.conv2 = Conv2D(128, kernel_size = 4, strides = 2, padding = 'SAME')
    self.conv3 = Conv2D(256, kernel_size = 4, strides = 2, padding = 'SAME')
    self.conv4 = Conv2D(512, kernel_size = 4, strides = 2, padding = 'SAME')
    self.conv5 = Conv2D(1024, kernel_size = 4, strides = 2, padding = 'SAME')
    self.conv6 = Conv2D(2048, kernel_size = 4, strides = 2, padding = 'SAME')    
    self.conv7_1 = Conv2D(1, kernel_size = 3, strides = 1, padding = 'SAME')    
    self.conv7_2 = Conv2D(label_1.shape[1], kernel_size = int(image_size/64), strides = 1, padding = 'VALID')    
    self.activation_LeakyReLU = LeakyReLU(alpha=0.01)

  def call(self, x):
    x = self.conv1(x)
    x = self.activation_LeakyReLU(x)
    assert x.shape == (2, 64, 64, 64) 
    print("The shape of the input tensor after Input layer :", x.shape)

    x = self.conv2(x)
    x = self.activation_LeakyReLU(x)
    assert x.shape == (2, 32, 32, 128) 
    
    x = self.conv3(x)
    x = self.activation_LeakyReLU(x)
    assert x.shape == (2, 16, 16, 256) 
    
    x = self.conv4(x)
    x = self.activation_LeakyReLU(x)
    assert x.shape == (2, 8, 8, 512) 

    x = self.conv5(x)
    x = self.activation_LeakyReLU(x)
    assert x.shape == (2, 4, 4, 1024) 

    x = self.conv6(x)
    x = self.activation_LeakyReLU(x)
    assert x.shape == (2, 2, 2, 2048) 
    print("The shape of the input tensor after Hidden layer :", x.shape)


    D_src = self.conv7_1(x)
    D_cls = self.conv7_2(x)
    assert D_src.shape == (2, 2, 2, 1) 
    assert D_cls.shape == (2, 1, 1, 3) 
    print("The shape of the input tensor after Output layer :", D_src.shape, D_cls.shape)
    
    return D_src, D_cls
