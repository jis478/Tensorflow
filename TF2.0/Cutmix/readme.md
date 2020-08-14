
# Cutmix implemented in the Tensorflow 2.x

#### This is a tensorflow 2.x version of CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features 

## Issue history
- (2020.7.6) I've raised an issue on a possibe Saved_Model API bug, and it is under investigation by the Tensorflow team.    (https://github.com/tensorflow/tensorflow/issues/41045) 
- (2020.8.11) Saved_Model has been replaced with Checkpoint due to the above unresolved investigation.


## Implementation
This code has followed the official tensorflow 2.0 coding guideline (https://www.tensorflow.org/alpha/guide/effective_tf2). 

Requirements: Tensorflow >= 2.0 , Python >= 3.6.0

- Currently only ResNet-50 is available as a backbone network.
- Tensorflow 2.x doesn't support slicing so instead masking has been used. Please correct me know if slicing (assignment) functionality exists in Tensorflow.)  

<br/>
<br/>

## Results

![Representative image](https://github.com/jis478/Tensorflow/blob/master/TF2.0/Cutmix/imgs/original.PNG) \
![Representative image](https://github.com/jis478/Tensorflow/blob/master/TF2.0/Cutmix/imgs/cutmix.PNG) \
**Picture:** (UP) Original images (DOWN) Cutmix images (from Cutmix_display.ipynb)

<br/>
<br/>

![Representative image](https://github.com/jis478/Tensorflow/blob/master/TF2.0/Cutmix/imgs/plots3.PNG) \
**Picture:**  (Left) Top-1 training error & loss (CIFAR-100)   (Right) Top-1 error & loss (CIFAR-100) 
                            
<br/>
<br/>

## Training & Inference 

``` 
python Train.py \
--epochs 300
--batch_size 128
--momentum 0.9
--print_freq 10
--layers [3, 4, 6, 3]                 # resnet50
--dataset cifar100
--beta 1.0
--cutmix_prob 0.5
--lr_boundaries [100,50,200]
--lr_values [0.2, 0.1, 0.05, 0.01]
--ckpt_dir ./ckpt_dir                 # a new folder inside './ckpt_dir' will be created using the current time stamp 
--tensorboard_dir ./tensorboard_dir   # for tensorboard logging
--verbose 1
```


``` 
python Test.py \
--batch_size 128
--layers [3,4,6,3]                # resnet 50
--dataset cifar100
--ckpt_dir ./ckpt_dir/20200815    # checkpoint directory  
--verbose 1
```

## Reference
- original paper: https://arxiv.org/abs/1905.04899
- original code (Pytorch) : https://github.com/clovaai/CutMix-PyTorch
