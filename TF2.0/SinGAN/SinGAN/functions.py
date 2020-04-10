import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.io as sio
import math
from skimage import io as img
from skimage import color, morphology, filters
from SinGAN.imresize import imresize
import SinGAN.models as models
import os
import random
import tensorflow as tf
import pickle

def np2tensor(x,opt):
    if opt.nc_im == 3:
        x = x[None,:,:,:] # 3 dims -> 4 dims
    else:
        x = color.rgb2gray(x)
        x = x[None,:,:,None] 
    x = tf.convert_to_tensor(x, dtype='float32')/255. # NWHC
    x = norm(x)
    return x

def denorm(x):
    out = (x + 1) / 2
    return tf.clip_by_value(out, 0, 1)

def norm(x):
    out = (x -0.5) *2
    return tf.clip_by_value(out, -1, 1)

def read_image(opt):
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    return np2tensor(x, opt)

def convert_image_np(inp):
    if inp.shape[3]==3:
        inp = denorm(inp)
        inp = inp[-1,:,:,:]
    else:
        inp = denorm(inp)
        inp = inp[-1,:,:,-1]
    inp = np.clip(inp,0,1)
    return inp

def save_image(real_cpu,receptive_feild,ncs,epoch_num,file_name):
    fig,ax = plt.subplots(1)
    if ncs==1:
        ax.imshow(real_cpu.view(real_cpu.size(2),real_cpu.size(3)),cmap='gray')
    else:
        ax.imshow(convert_image_np(real_cpu.cpu()))
    rect = patches.Rectangle((0,0),receptive_feild,receptive_feild,linewidth=5,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(file_name)
    plt.close(fig)

def convert_image_np_2d(inp):
    inp = denorm(inp)
    inp = inp.numpy()
    return inp

def generate_noise(size,num_samp=1,type='gaussian', scale=1): # size[0]: c, size[1]: W, size[2]: H
    if type == 'gaussian':
        noise = tf.random.normal([num_samp, round(int(size[1])/scale), round(int(size[2])/scale), int(size[0])]) #NWHC
        noise = upsampling(noise, int(size[1]), int(size[2]))
    if type =='gaussian_mixture':
        noise1 = tf.random.normal([num_samp, int(size[1]), int(size[2]), int(size[0])])+5
        noise1 = tf.random.normal([num_samp, int(size[1]), int(size[2]), int(size[0])])
        noise = noise1+noise2
    if type == 'uniform':
        noise = tf.random.uniform((num_samp, int(size[1]), int(size[2]), int(size[0])))
    return noise

def plot_learning_curves(G_loss,D_loss,epochs,label1,label2,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,G_loss,n,D_loss)
    plt.xlabel('epochs')
    plt.legend([label1,label2],loc='upper right')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def plot_learning_curve(loss,epochs,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def upsampling(im,sx,xy):
    upsample = tf.image.resize(im,[round(sx), round(xy)])
    return upsample

def fake_gp_generator(input_image, fake_image):
    epsilon = tf.random.normal([1])
    fake_gp = (epsilon * input_image) + ((1-epsilon) * fake_image)     
    return tf.cast(fake_gp, tf.float32)

def gp_calc(fake_gp, opt):
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(fake_gp)
        gp_D_src = netD(fake_gp) ### batch norm 없다.
    gp_D_grad = gp_tape.gradient(gp_D_src, fake_gp) ############ 여기만 다르다.
    gp = opt.lambda_grad*gradient_penalty(gp_D_grad)
    return gp
   
def gradient_penalty(gp_D_grad):
    gp = tf.reduce_mean(((tf.norm(gp_D_grad, ord=2)-1.0)**2))
    return gp

def read_image_dir(dir,opt):
    x = img.imread('%s' % (dir))
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    return x

def torch2uint8(x):
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

def read_image2np(opt):
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = x[:, :, 0:3]
    return x


def init_models(opt):
    netD = models.WDiscriminator(opt)    
    netG = models.GeneratorConcatSkip2CleanAdd(opt)
    return netD, netG

def save_networks(netD,netG,z,opt,scale_num):
    netD.save_weights('%s/netD' % (opt.outf))
    netG.save_weights('%s/netG' % (opt.outf))
        
      
def adjust_scales2image(real_,opt):
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(int(real_.shape[1]), int(real_.shape[2]))), 1), opt.scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([opt.max_size, max([int(real_.shape[1]), int(real_.shape[2])])]) / max([int(real_.shape[1]), int(real_.shape[2])]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([int(real_.shape[1]), int(real_.shape[2])]),1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    opt.scale_factor = math.pow(opt.min_size/(min(int(real.shape[1]),int(real.shape[2]))),1/(opt.stop_scale))
    scale2stop = math.ceil(math.log(min([opt.max_size, max([int(real_.shape[1]), int(real_.shape[2])])]) / max([int(real_.shape[1]), int(real_.shape[2])]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real


def adjust_scales2image_SR(real_,opt):
    opt.min_size = 18
    opt.num_scales = int((math.log(opt.min_size / min(real_.shape[2], real_.shape[3]), opt.scale_factor_init))) + 1
    scale2stop = int(math.log(min(opt.max_size , max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
    scale2stop = int(math.log(min(opt.max_size, max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def creat_reals_pyramid(real,reals,opt):
    real = real[:,:,:,0:3]
    for i in range(0,opt.stop_scale+1,1):
        scale = math.pow(opt.scale_factor,opt.stop_scale-i)
        curr_real = imresize(real,scale,opt)
        reals.append(curr_real)
    return reals

def load_trained_pyramid(opt, mode_='train'):
    mode = opt.mode
    opt.mode = 'train'
    if (mode == 'animation_train') | (mode == 'SR_train') | (mode == 'paint_train'):
        opt.mode = mode
    dir = generate_dir2save(opt)
    i = 0
    Gs = []
    opt.out_ = os.path.join(opt.out, opt.out_)
    if(os.path.exists(dir)):
        while i >= 0:
            if os.path.exists('%s/%s/' % (opt.out_, str(i))): 
                netG = models.GeneratorConcatSkip2CleanAdd(opt)
                netG.load_weights('%s/%s/netG' % (opt.out_, str(i)))
                Gs.append(netG)
                i += 1 
            else:
                break
                
        with open('%s/Zs.pkl' % (opt.out_), 'rb') as f:
            Zs = pickle.load(f)
        with open('%s/reals.pkl' % (opt.out_), 'rb') as f:
            reals = pickle.load(f)
        with open('%s/NoiseAmp.pkl' % (opt.out_), 'rb') as f:
            NoiseAmp = pickle.load(f)        
    else:
        print('no appropriate trained model is exist, please train first')
    opt.mode = mode
    return Gs,Zs,reals,NoiseAmp


def generate_in2coarsest(reals,scale_v,scale_h,opt):
    real = reals[opt.gen_start_scale]
    real_down = upsampling(real, scale_v * real.shape[1], scale_h * real.shape[2]) 
    if opt.gen_start_scale == 0:
        in_s = tf.zeros_like((real_down))
    else: #if n!=0
        in_s = upsampling(real_down, real_down.shape[1], real_down.shape[2])
    return in_s

def generate_dir2save(opt):
    dir2save = None
    if (opt.mode == 'train') | (opt.mode == 'SR_train'):
        dir2save = '%s/TrainedModels/%s/scale_factor=%f' % (opt.out, opt.input_name[:-4], opt.scale_factor_init)
    elif (opt.mode == 'animation_train') :
        dir2save = 'TrainedModels/%s/scale_factor=%f_noise_padding' % (opt.input_name[:-4], opt.scale_factor_init)
    elif (opt.mode == 'paint_train') :
        dir2save = 'TrainedModels/%s/scale_factor=%f_paint/start_scale=%d' % (opt.input_name[:-4], opt.scale_factor_init,opt.paint_start_scale)
    elif opt.mode == 'random_samples':
        dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out,opt.input_name[:-4], opt.gen_start_scale)
    elif opt.mode == 'random_samples_arbitrary_sizes':
        dir2save = '%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (opt.out,opt.input_name[:-4], opt.scale_v, opt.scale_h)
    elif opt.mode == 'animation':
        dir2save = '%s/Animation/%s' % (opt.out, opt.input_name[:-4])
    elif opt.mode == 'SR':
        dir2save = '%s/SR/%s' % (opt.out, opt.sr_factor)
    elif opt.mode == 'harmonization':
        dir2save = '%s/Harmonization/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'editing':
        dir2save = '%s/Editing/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'paint2image':
        dir2save = '%s/Paint2image/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
        if opt.quantization_flag:
            dir2save = '%s_quantized' % dir2save
    return dir2save

def post_config(opt):
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out_ = 'TrainedModels/%s/scale_factor=%f' % (opt.input_name[:-4], opt.scale_factor)
    if opt.mode == 'SR':
        opt.alpha = 100

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    tf.random.set_seed(opt.manualSeed)
    return opt

def calc_init_scale(opt):
    in_scale = math.pow(1/2,1/3)
    iter_num = round(math.log(1 / opt.sr_factor, in_scale))
    in_scale = pow(opt.sr_factor, 1 / iter_num)
    return in_scale,iter_num
