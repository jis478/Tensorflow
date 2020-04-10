import SinGAN.functions as functions
import SinGAN.models as models
import os
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize
import tensorflow as tf
import pickle
import numpy as np

def train(opt,Gs,Zs,reals,NoiseAmp):
    real_ = functions.read_image(opt)
    in_s = 0
    scale_num = 0
    real = imresize(real_,opt.scale1,opt)
    reals = functions.creat_reals_pyramid(real,reals,opt)
    nfc_prev = 0
    netD_optimizer = tf.keras.optimizers.Adam(learning_rate=opt.lr_d, beta_1=opt.beta1, beta_2=0.999) 
    netG_optimizer = tf.keras.optimizers.Adam(learning_rate=opt.lr_g, beta_1=opt.beta1, beta_2=0.999) 

    
    while scale_num<opt.stop_scale+1:

        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)

        try:
            os.makedirs(opt.outf)
        except OSError:
                pass

        plt.imsave('%s/real_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)
        D_curr,G_curr = init_models(opt) 
        if nfc_prev == opt.nfc: 
            D_curr.load_weights('%s/%d/netD' % (opt.out_, scale_num-1))
            G_curr.load_weights('%s/%d/netG' % (opt.out_, scale_num-1))

        z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals,Gs,Zs,in_s,NoiseAmp,opt,scale_num, netG_optimizer, netD_optimizer)

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)
        with open('%s/Zs.pkl' % (opt.out_), 'wb') as f:
            pickle.dump(Zs, f)
        with open('%s/reals.pkl' % (opt.out_), 'wb') as f:
            pickle.dump(reals, f)
        with open('%s/NoiseAmp.pkl' % (opt.out_), 'wb') as f:
            pickle.dump(NoiseAmp, f)
        scale_num+=1
        nfc_prev = opt.nfc
        del D_curr,G_curr
    return None

def train_single_scale(netD,netG,reals,Gs,Zs,in_s,NoiseAmp,opt,scale_num, netG_optimizer, netD_optimizer):  

    real = reals[len(Gs)]
    opt.nzx = real.shape[1]#+(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = real.shape[2]#+(opt.ker_size-1)*(opt.num_layer)
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)  
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    m_noise = tf.keras.layers.ZeroPadding2D(padding=(int(pad_noise), int(pad_noise)))
    m_image = tf.keras.layers.ZeroPadding2D(padding=(int(pad_image), int(pad_image)))
    alpha = opt.alpha

    fixed_noise = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy])
    z_opt = tf.zeros_like(fixed_noise)
    z_opt = m_noise(z_opt)
    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []
        
    for epoch in range(opt.niter):

        if (Gs == []) & (opt.mode != 'SR_train'):
            z_opt = functions.generate_noise([1,opt.nzx,opt.nzy]) # (1,33,25)
            z_opt = tf.broadcast_to(z_opt, [1, z_opt.shape[1], z_opt.shape[2], 3]) # (1,33,25,3)
            z_opt = m_noise(z_opt)
            noise_ = functions.generate_noise([1,opt.nzx,opt.nzy]) 
            noise_ = tf.broadcast_to(noise_, [1, noise_.shape[1], noise_.shape[2], 3])    
            noise_ = m_noise(noise_)
        else:
            noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy])
            noise_ = m_noise(noise_)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################

        for j in range(opt.Dsteps):              
            with tf.GradientTape() as netD_tape:
            
                real = reals[len(Gs)]
                output_real = netD(real) 
                errD_real = -tf.reduce_mean(output_real)
                D_x = float(-errD_real.numpy()) # Conversion to numpy required
                # train with fake
                if (j==0) & (epoch == 0): 
                    if (Gs == []) & (opt.mode != 'SR_train'):
                        prev = tf.zeros([1,opt.nzx,opt.nzy,opt.nc_z])
                        in_s = prev
                        prev = m_image(prev)
                        z_prev = tf.zeros([1, opt.nzx, opt.nzy, opt.nc_z])
                        z_prev = m_noise(z_prev)
                        opt.noise_amp = 1
                    else:
                        prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                        prev = m_image(prev)
                        z_prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rec',m_noise,m_image,opt)
                        RMSE = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(real, z_prev))))
                        opt.noise_amp = opt.noise_amp_init*RMSE
                        z_prev = m_image(z_prev)
                else:
                    prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                    prev = m_image(prev)
                if (Gs == []) & (opt.mode != 'SR_train'):
                    noise = noise_
                else:
                    noise = opt.noise_amp*noise_+prev
                fake = netG(noise, prev, train=True)
                output_fake = netD(fake)
                errD_fake = tf.reduce_mean(output_fake)
                D_G_z = float(output_fake.numpy().mean())
                fake_gp = functions.fake_gp_generator(real, fake)
        
                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(fake_gp)
                    gp_D_src = netD(fake_gp) 
                gp_D_grad = gp_tape.gradient(gp_D_src, fake_gp)                  
                gp = opt.lambda_grad*tf.reduce_mean(((tf.norm(gp_D_grad, ord=2, axis=3)-1.0)**2))

                errD = errD_real + errD_fake + gp
                
            netD_gradients = netD_tape.gradient(errD, netD.trainable_variables)
            netD_optimizer.apply_gradients(zip(netD_gradients, netD.trainable_variables))

        for j in range(opt.Gsteps):
            with tf.GradientTape() as netG_tape:

                errG_fake = -tf.reduce_mean(output_fake)
                if alpha!=0:
                    Z_opt = opt.noise_amp*z_opt+z_prev
                    rec_loss = alpha * tf.reduce_mean(tf.square(tf.subtract(netG(Z_opt, z_prev, train=True), real)))
                else:
                    Z_opt = z_opt
                    rec_loss = 0
                errG = errG_fake + rec_loss
                
            netG_gradients = netG_tape.gradient(errG, netG.trainable_variables) # 오직 netG만 update! 
            netG_optimizer.apply_gradients(zip(netG_gradients, netG.trainable_variables))
    
        errG2plot.append(errG+rec_loss)
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        z_opt2plot.append(rec_loss)

        if epoch % 1 == 0 or epoch == (opt.niter-1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            plt.imsave('%s/fake_sample.png' %  (opt.outf), functions.convert_image_np(fake))
            plt.imsave('%s/G(z_opt).png'    % (opt.outf),  functions.convert_image_np(netG(Z_opt, z_prev, train=False)))
    functions.save_networks(netD,netG,z_opt,opt,scale_num) # single scale training 끝날때 마다 저장 함 
 
    return z_opt,in_s,netG   



def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,opt):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[1] - 2 * pad_noise, Z_opt.shape[2] - 2 * pad_noise])
                    z = tf.broadcast_to(z, [1, z.shape[1], z.shape[2], 3])
                else:
                    z = functions.generate_noise([opt.nc_z,Z_opt.shape[1] - 2 * pad_noise, Z_opt.shape[2] - 2 * pad_noise])
                
                z = m_noise(z)
                G_z = G_z[:,0:real_curr.shape[1],0:real_curr.shape[2],:] #PY: NCWH, TF:NWHC
                G_z = m_image(G_z)
                z_in = noise_amp*z+G_z
                G_z = G(z_in,G_z, train=True)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,0:real_next.shape[1],0:real_next.shape[2], :]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, 0:real_curr.shape[1], 0:real_curr.shape[2], :]
                G_z = m_image(G_z)
                z_in = noise_amp*Z_opt+G_z
                G_z = G(z_in,G_z, train=True)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,0:real_next.shape[1],0:real_next.shape[2], :]
                count += 1
    return G_z


def init_models(opt):
    netD = models.WDiscriminator(opt)    
    netG = models.GeneratorConcatSkip2CleanAdd(opt)
    return netD, netG
