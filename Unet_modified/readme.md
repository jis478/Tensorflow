# Unet for imbalanced dataset

This is a modified version of the original Unet code (https://github.com/jakeret/tf_unet).  

#1. Loss function (for imbalanced datasets) modified
----------------------------------------------------

 - In the original Unet code the dice loss involves both background ([:,:,:,0]) and target ([:,:,:,1]) dimensions. After some
   experiments I've found that this behavior hinders the learning process for imbalanced datasets. 
   With the modified dice loss which only involves target dimension ([:,:,:,0] the model accuracy has significanly improved for my
   imbalanced dataset. Moreover, thanks to a Kaggle winning solution (https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge),
   I've also found that the dice loss combined with binary cross entroy loss can be a better choice of loss function addressing the class
   imbalance issue siding with neither background nor target class. 

#2. tfrecord, tf.data & tf.image added
--------------------------------------

 - All original images and masks are first converted to ".tfrecord" format for later being efficiently incorporated with tf.data 
 - tf.data is the official data pipeline recommended by Tensorflow 
 - tf.image is a high-level API for augmentation using GPU resources which allows high-speed real-time augmentation
 
#3. Input & ouput image size adjustment
---------------------------------------

 - Input and output images are all set to be the same size by removing an image crop function and replacing "VALID" with "SAME" for all
   conv layers in the original Unet code 
   

#4. Batch Normalization applied
---------------------------------------



# Usage (Please refer to '/example/membrane' in this github folder)

###Step 1: TFRecord creation
--------------------------
```
tfrecord = TFrecord_Create_For_Unet(train_test = 'train',
                        img_folder = '/home/mywork/Markkim/images/',
                        img_type = 'jpg',
                        label_name = 'labels',
                        tf_record_pre_fix = 'tfrecords',
                        nx = 512,
                        ny = 512
                       )
                       
tfrecord = TFrecord_Create_For_Unet(train_test = 'test',
                        img_folder = '/home/mywork/Markkim/images/',
                        img_type = 'jpg',
                        label_name = 'labels',
                        tf_record_pre_fix = 'tfrecords',
                        nx = 512,
                        ny = 512
                       )
                       
```

###Step 2: Setting up a data provider
-----------------------------------
```
data_provider = Tfrecord_ImageDataProvider(                 
                                        train_tfrecord_path = 'train.tfrecords', 
                                        test_tfrecord_path = 'test.tfrecords', 
                                        channels = 3, train_batch_size = 3, test_batch_size = 4, 
                                        nx = 512, ny = 512, n_imgs = 64)
                                     
```

###Step 3: Training
------------------
```
net = Unet(cost = "bce_dice_coefficient", layers=5, features_root=64, channels=3) 
trainer = Trainer(net, data_provider = data_provider, batch_size=3, validation_batch_size = 2,optimizer="adam", lr = 0.001, 
opt_kwargs={})
path = trainer.train(output_path='./output_path', prediction_path = ./prediction_path', training_iters=21, epochs=100)
```
