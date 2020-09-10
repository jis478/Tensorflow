
# U-GAT-IT implemented in the Tensorflow 2

#### This is a tensorflow 2 version of U-GAT-IT:

## Issue history
- (2020.9.7) U-GAT-IT requires more GPU resources compared to other Image to Image Translation algorithms such as CycleGAN and MUNIT. With my GTX 1080 it was impossible to proceed as many iterations (1,000,000) as described in the paper. I'm planning to run this code on Google Cloud with V100 and the result will be reported here soon.

## Implementation
- This code has followed the official tensorflow 2.0 coding guideline (https://www.tensorflow.org/alpha/guide/effective_tf2). 
- There are two versions of original codes (Tensorflow 1.x and Pytorch) and I've thoroughly investigated them. This code is following
  the best parts of each version.

Requirements: Tensorflow >= 2.0 , Python >= 3.6.0


## Training & Inference 

``` 
python main.py --dataset 'horse2zebra'    # Downloads Tfds dataset automatically 

pythin main.py --dataset 'dataset'        # Custom dataset to be used :  ./dataset/trainA, ./dataset/trainB

```

``` 
python Test.py --dataset 'horse2zebra' --ckpt_path './20200812_0946/ckpt/.'    # The latest checkpoint will be loaded 

python Test.py --dataset 'dataset' --ckpt_path './20200812_0946/ckpt/.'        # Custom dataset to be used:  ./dataset/TestA, ./dataset/TestB

```

## Reference
- original paper: https://arxiv.org/abs/1907.10830
- original code (Tensorflow) : https://github.com/taki0112/UGATIT
                (Pytorch) : https://github.com/znxlwm/UGATIT-pytorch
                



## Result (up to 20,000 iterations)

