from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
import shutil

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='./Input/Images/')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist and the model will be removed')
        shutil.rmtree(dir2save)
    try:
        os.makedirs(dir2save)
    except OSError:
        pass
    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    train(opt, Gs, Zs, reals, NoiseAmp)
    SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt) # 여기서는 생성이 잘됨.. 네트워크 불러올때 이상하게 불러옴 6번째만.
    
    # 여기서 생성되는 네트워크의 variable.shape 찍어보고
    # 로드해서 생성되는 네트워크의 variable.shape 찍어보자

