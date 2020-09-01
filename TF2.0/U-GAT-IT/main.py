from UGATIT import UGATIT
import argparse
from utils import *

def parse_args():

    ''' only Horses <-> Zebras dataset is available '''

    desc = "Tensorflow 2.x implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    parser.add_argument('--light', type=str2bool, default=False, help='[U-GAT-IT full version / U-GAT-IT light version]')

    parser.add_argument('--iterations', type=int, default=1000000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size for training')
    parser.add_argument('--sample_num', type=int, default=5, help='The size of batch size for saving images')
    parser.add_argument('--print_freq', type=int, default=10, help='The number of loss print freq')
    parser.add_argument('--sample_freq', type=int, default=1000, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=100000, help='The number of model save freq')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN')
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight for Cycle')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight for Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight for CAM')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--global_n_dis', type=int, default=7, help='The number of Global discriminator layer')
    parser.add_argument('--local_n_dis', type=int, default=5, help='The number of Local discriminator layer')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--ckpt_path', type=str, default='ckpt', help='Tensorflow ckpt folder')
    parser.add_argument('--tensorboard_path', type=str, default='tensorboard', help='Tensorflow tensorboard folder')
    parser.add_argument('--image_path', type=str, default='image', help='Sample translation image folder')
    return parser.parse_args()

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    gan = UGATIT(args)

    # build graph
    gan.build()

    if args.phase == 'train':
        gan.train()
        print(" [*] Training finished!")

    if args.phase == 'test':
        gan.test()
        print(" [*] Test finished!")

if __name__ == '__main__':
    main()
