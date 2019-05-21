# StarGAN implemented in Tensorflow 2.0 style 

This is a tensorflow 2.0 version of StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation(https://arxiv.org/abs/1711.09020).  

This code has followed the official tensorflow 2.0 coding guideline (https://www.tensorflow.org/alpha/guide/effective_tf2) and basic & advanced tutorials. Please be aware of that the code has been written with Tensorflow 1.11 with eager mode as CUDA version 10 is required for tensorflow 2.0 and my machine wasn't ready for CUDA upgrade. All the parts of this code can easily be converted to tensorflow 2.0 with minimal efforts.  

Requirements: Tensorflow 1.11+

#1. Dataset
-----------------------
 - The original paper used both CelebA and RaFD datasets, but here I only used CelebA data for the sake of the training time.

 - The original paper uses only five features from the CelebA dataset which are ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']. For clarification I've only included images with a distinct hair color (eg. Black_Hair (o),  Block_Hair + Brown_Hair (x)). This trick has reduced the number of images down to around 110,000 from 220,000.
 
```
    domain_list = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    list_attr_celeba = pd.read_csv(attr_csv)
    list_attr_celeba = list_attr_celeba.loc[(list_attr_celeba['Black_Hair'] == 1) | (list_attr_celeba['Blond_Hair'] == 1) | (list_attr_celeba['Brown_Hair'] == 1), domain_list]
    list_attr_celeba = list_attr_celeba.replace({-1:0})
    list_attr_celeba = list_attr_celeba.loc[list_attr_celeba.apply(lambda x: x['Black_Hair'] + x['Blond_Hair'] + x['Brown_Hair'], axis=1) == 1, :]
```
 
 - Central crop, which can be found on the original research paper is a bit ambiguous as the exact crop ratios aren't provided. Alternatively I've used the central crop function of the tf.image class and this could possibly cause a slight performance difference compared to the original pytorch code. (I need to fix this issue in the upcoming version)
 - The other dataset features also follow the official tensorflow coding guideline (eg. batch, repeat, make_one_shot_iterator)

#2. Loss function
--------------------------------------
 - Loss functions are the core parts of this code. Adverserial loss (WGAN-GP), domain classification loss and reconstruction loss are all implemented and lamda values are also the same as the original paper but they are definitely worth tuning.
 - Discriminator is updated five times per each generator update (same as the original paper)

#3. Training conditions 
--------------------------------------
 - image resize = 128 (same as the original paper)
 - batch_size = 16 (same as the original paper)
 - epochs = 20 (same as the original paper)
 - learning_rate = 0.0001 (same as the original paper, but learning rate decay not applied here)
 - classification lambda = 1
 - reconstruction lambda = 10
 - gradient penalty lambda = 10
 
#3. Tricks
--------------------------------------
The following tricks are also implemented according to the original research paper.
 - Instance Normalisation (https://github.com/ilguyi/generative.models.tensorflow.v2/tree/master/gans)
   
#4. Output sample
----------------------------------------
Training set (92, 247 images) for 20 epochs with a batch size of 16 took around 31 hrs on NVIDIA V100 GPU. I believe with more traning and sophiscated training schedule (eg. learning rate decay used in the original code) the result could be better than the below samples.

![Representative image](https://github.com/jis478/Tensorflow/blob/master/TF2.0/StarGAN/imgs/a.PNG)<br>
**Picture:** (Left) Original real image with original attributes (Right) Fake image with random attributes 

#5. Upcoming update notice
-----------------------------------------
Full training results (with 220,000 images and learaning rate schedule) will be updated shortly.
