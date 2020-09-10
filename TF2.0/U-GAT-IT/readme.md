
# U-GAT-IT implemented in Tensorflow 2

#### This is a tensorflow 2 version of U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation

## Issue history
- (2020.9.7) U-GAT-IT requires more GPU resources compared to other traditional Image to Image Translation algorithms such as CycleGAN. With my GTX 1080 it was impossible to proceed as many iterations (1,000,000) as described in the paper. I'm planning to run this code on Google Cloud with V100 and the result will be reported here soon. If you have enough GPU resources to run this code, please share with me the result.

## Implementation
- This code has followed the official tensorflow 2.0 coding guideline (https://www.tensorflow.org/alpha/guide/effective_tf2). 
- There are two versions of original codes (Tensorflow 1.x and Pytorch) and I've thoroughly investigated them. This code is following
  the best parts of each version.

Requirements: Tensorflow >= 2.0 , Python >= 3.6.0


## Training & Inference 
- Please check 'main.py' for detail training conditions. 

#### training
``` 
python main.py --phase 'train' --dataset 'horse2zebra'    # Downloads 'horse2zebra' Tfds dataset automatically 

pythin main.py --phase 'train' --dataset 'dataset'        # Custom dataset to be used :  ./dataset/trainA, ./dataset/trainB (test data are not used for training)

```

#### Inference
``` 
python main.py --phase 'test' --dataset 'horse2zebra' --ckpt_path './20200812_0946/ckpt/.'    # (horse -> zebra) The latest checkpoint will be loaded 

python main.py --phase 'test' --dataset 'dataset' --ckpt_path './20200812_0946/ckpt/.'        # (Domain A -> Domain B) Custom dataset to be used:  ./dataset/TestA, ./dataset/TestB 

```

#### Custom dataset structure  
```
├── dataset
       ├── trainA
           ├── xxx.jpg
           ├── yyy.jpg
           └── ...
       ├── trainB
           ├── zzz.jpg
           ├── www.jpg
           └── ...
       ├── testA
           ├── aaa.jpg 
           ├── bbb.jpg
           └── ...
       └── testB
           ├── ccc.jpg 
           ├── ddd.jpg
           └── ...
           
``` 

## Intermediate training results
- As can be seen here, the training is incomplete.

![Representative image](https://github.com/jis478/Tensorflow/blob/master/TF2.0/U-GAT-IT/imgs/D_loss.png) \
![Representative image](https://github.com/jis478/Tensorflow/blob/master/TF2.0/U-GAT-IT/imgs/G_loss.png) 

**Picture:** (UP) Discriminator loss (DOWN) Generator loss



## Reference
- original paper: https://arxiv.org/abs/1907.10830
- original code (Tensorflow) : https://github.com/taki0112/UGATIT
                (Pytorch) : https://github.com/znxlwm/UGATIT-pytorch
                



