from __future__ import print_function
import SinGAN.functions
import SinGAN.models
import argparse
import os
import random
from SinGAN.imresize import imresize
from skimage import io as img
import numpy as np
from skimage import color
import math
import imageio
import matplotlib.pyplot as plt
from SinGAN.training import *
from config import get_arguments
import tensorflow as tf

def SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=50):
    if in_s is None: # torch NCWH tf NWHC
        in_s = tf.zeros_like(reals[0])
    images_cur = []
    for G,Z_opt,noise_amp in zip(Gs,Zs,NoiseAmp):        
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        m = tf.keras.layers.ZeroPadding2D(padding=(int(pad1), int(pad1)))
        nzx = (Z_opt.shape[1]-pad1*2)*scale_v
        nzy = (Z_opt.shape[2]-pad1*2)*scale_h

        images_prev = images_cur
        images_cur = []

        for i in range(0,num_samples,1):
            if n == 0:
                z_curr = functions.generate_noise([1,nzx,nzy])
                z_curr = tf.tile(z_curr, multiples=(1,1,1,3))
                z_curr = m(z_curr)
            else:        
                z_curr = functions.generate_noise([opt.nc_z,nzx,nzy])
                z_curr = m(z_curr)

            if images_prev == []:
                I_prev = m(in_s)
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev,1/opt.scale_factor, opt)
                if opt.mode != "SR":
                    I_prev = I_prev[:,0:round(scale_v * reals[n].shape[1]), 0:round(scale_h * reals[n].shape[2]),:]
                    I_prev = m(I_prev)
                    I_prev = I_prev[:,0:z_curr.shape[1],0:z_curr.shape[2],:]
                    I_prev = functions.upsampling(I_prev,z_curr.shape[1],z_curr.shape[2])
                else:
                    I_prev = m(I_prev)

            if n < gen_start_scale:
                z_curr = Z_opt

            z_in = noise_amp*(z_curr)+I_prev
            I_curr = G(z_in, I_prev, train=True)

            if n == len(reals)-1:
                if opt.mode == 'train':
                    dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.input_name[:-4], gen_start_scale)
                else:
                    dir2save = functions.generate_dir2save(opt)
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                if (opt.mode != "harmonization") & (opt.mode != "editing") & (opt.mode != "SR") & (opt.mode != "paint2image"):
                    plt.imsave('%s/%d.png' % (dir2save, i), functions.convert_image_np(I_curr))
            images_cur.append(I_curr)
        n+=1
    return I_curr

