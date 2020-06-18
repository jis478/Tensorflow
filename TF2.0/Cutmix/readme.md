Cutmix implemented in the Tensorflow 2.0
This is a tensorflow 2.0 version of CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features 

(paper:https://arxiv.org/abs/1905.04899) (code: https://github.com/clovaai/CutMix-PyTorch)

This code has followed the official tensorflow 2.0 coding guideline (https://www.tensorflow.org/alpha/guide/effective_tf2) and basic & advanced tutorials. 
Requirements: Tensorflow >= 2.0 , Python >= 3.6.0

Currently, only ResNet-50 is supported for CIFAR-10 and CIFAR-100 datasets. PyramidNet and ImageNet will be supported soon.

## Training
python train.py \
--dataset cifar100 \
--depth 200 \
--batch_size 64 \
--lr 0.25 \
--expname ResNet50 \
--epochs 300 \
--beta 1.0 \
--cutmix_prob 0.5 \


## Inference
python test.py \
--net_type resnet \
--dataset cifar100 \
--batch_size 64 \
--depth 50 \
--pretrained /... 
