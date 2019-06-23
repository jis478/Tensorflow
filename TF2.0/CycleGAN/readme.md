# CycleGAN implemented in the Tensorflow 2.0 style 

This is a tensorflow 2.0 version of CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
(https://arxiv.org/pdf/1703.10593.pdf).  

This code has followed the official tensorflow 2.0 coding guideline (https://www.tensorflow.org/alpha/guide/effective_tf2) and basic & advanced tutorials. Please be aware of that the code has been written with Tensorflow 1.11 with eager mode as CUDA version 10 is required for tensorflow 2.0 and my machine wasn't ready for CUDA upgrade. All the parts of this code can easily be converted to tensorflow 2.0 with minimal efforts.  

Requirements: Tensorflow 1.11+

#1. Dataset
-----------------------
 - horse2zebra dataset has been used for performance verification and the performance turned out to be 'nearly' on par with that of the original code  

 - The model structure has followed Tensorflow implementation of CycleGANs (https://github.com/leehomyc/cyclegan-1) 
  
#2. Training conditions 
--------------------------------------
 All the below hyper-parameters are same as the original paper except the learning rate scheduling which is to be added shortly.
 - image resize = 286
 - batch_size = 1
 - epochs = 100
 - learning_rate = 0.002 
 - Generator (Domain A) = 10.0
 - Generator (Domain B) = 10.0
   
#3. Output sample
----------------------------------------


![Representative image](https://github.com/jis478/Tensorflow/blob/master/TF2.0/CycleGAN/imgs/horse.PNG)<br>![Representative image](https://github.com/jis478/Tensorflow/blob/master/TF2.0/CycleGAN/imgs/zebra.PNG)<br>

**Picture:** (Left) Original horse image (Right) Converted zebra image 

#4. Upcoming update notice
-----------------------------------------
Identity loss and skip connection technique will be added shortly.
