This code is based on Unet code (https://github.com/jakeret/tf_unet) and some functionalities are modified (or added/removed) 
as below. 

1. Loss function (for imbalanced datasets)
 - In the original Unet code the dice loss involves both background ([:,:,:,0]) and target ([:,:,:,1]) dimensions. After some
   experiments I've found that this behavior causes the learning process to poor. With the modified dice loss which only involves target 
   dimension the model accuracy has significanly improved for my imbalanced dataset. 
   
2. tfrecord, tf.data & tf.image 
 - All original images and masks are first converted to ".tfrecord" format for later being efficiently incorporated with tf.data 
 - tf.data is the official data pipeline recommended by Tensorflow 
 - tf.image is a high-level API for augmentation using GPU resources which allows high-speed real-time augmentation
 

3. Image ouput size issue
 - Input and output images are all set to be the same size by removing "crop" function and replacing "VALID" with "SAME" for all conv
   layers in the original Unet code 
