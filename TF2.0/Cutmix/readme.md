# Cutmix implemented in the Tensorflow 2.x
This is a tensorflow 2.x version of CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features 

paper: https://arxiv.org/abs/1905.04899

code: https://github.com/clovaai/CutMix-PyTorch

This code has followed the official tensorflow 2.0 coding guideline (https://www.tensorflow.org/alpha/guide/effective_tf2). 

Requirements: Tensorflow >= 2.0 , Python >= 3.6.0

- Currently only ResNet-50, CIFAR-10 and CIFAR-100 datasets are available. PyramidNet and ImageNet dataset will be available soon.
- Tensorflow 2.x doesn't support slicing so instead masking has been used. Please correct me know if slicing (assignment) functionality exists in Tensorflow.)  
- Hyper-parameters and training strategies all follow the original Pytorch version.



 ![Representative image](https://github.com/jis478/Tensorflow/blob/master/TF2.0/Cutmix/imgs/original.PNG) 

![Representative image](https://github.com/jis478/Tensorflow/blob/master/TF2.0/Cutmix/imgs/cutmix.PNG)

**Picture:** (UP) Original images (DOWN) Cutmix images (from Cutmix_display.ipynb)




## Training

```
python train.py \
--dataset cifar100 \  # or cifar10 
--depth 200 \
--batch_size 64 \
--lr 0.25 \
--expname ResNet50 \  # Only ResNet50 supported  
--epochs 300 \
--beta 1.0 \
--cutmix_prob 0.5 \
```


## Inference (to be implemented)
```
python test.py \
--net_type resnet \
--dataset cifar100 \
--batch_size 64 \
--depth 50 \
--pretrained /... 
```
