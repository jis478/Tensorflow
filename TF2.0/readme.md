# Tensorflow 2

Tensorflow 2 style implementations for various GAN Research papers 

## Tensorflow 1.13 + Keras API + Eager mode
  - CycleGAN (completed) : New implementation based on the paper
  - StarGAN (completed) : New implementation based on the paper
  - PGGAN (still working on) 
 
## Tensorflow 2.1
  - SinGAN (completed) : Converted from the original Pytorch code (all the Pytorch funtions coverted to the corresponding TF ones)
  - Cutmix (completed) : Converted from the original Pytorch code (all the Pytorch funtions coverted to the corresponding TF ones)  
                            - Tensorflow 2.x doesn't support slicing so instead masking has been used.
                            - ResNet-50 has also been implemented in Tensorflow 2.x based on the original Resnet-50 from the Cutmix repo.
