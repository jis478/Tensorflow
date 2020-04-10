# SinGAN implemented in the Tensorflow 2.0

This is a tensorflow 2.0 version of SinGAN: Learning a Generative Model from a Single Natural Image.
(paper: https://arxiv.org/pdf/1905.01164.pdf)
(code: https://github.com/tamarott/SinGAN)

This code has followed the official tensorflow 2.0 coding guideline (https://www.tensorflow.org/alpha/guide/effective_tf2) and basic & advanced tutorials. Basically all the parts of the codes are exactly same as the original codes except that all the Pytorch functions have been converted into the corresponsing Tensorflow 2.1 functions.

Requirements: Tensorflow >= 2.0 , Python >= 3.6.0

## Currently only the below commands are supported. Please refer to the original github code (https://github.com/tamarott/SinGAN) for more details. 

### Train a model
python main_train.py --input_name <input_file_name>

### Random samples
python random_samples.py --input_name <training_image_file_name> --mode random_samples --gen_start_scale <generation start scale number>

### Random samples of arbitrery sizes
python random_samples.py --input_name <training_image_file_name> --mode random_samples_arbitrary_sizes --scale_h <horizontal scaling factor> --scale_v <vertical scaling factor>
