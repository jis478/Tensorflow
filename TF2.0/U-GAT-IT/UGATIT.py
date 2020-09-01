import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from ops import *
from utils import *
from model import *
import datetime


AUTOTUNE = tf.data.experimental.AUTOTUNE

class UGATIT(object):
    def __init__(self, args):

        self.light = args.light
        self.iterations = args.iterations
        self.batch_size = args.batch_size
        self.sample_num = args.sample_num
        self.print_freq = args.print_freq
        self.sample_freq = args.sample_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        self.ch = args.ch
        self.n_res = args.n_res
        self.global_n_dis = args.global_n_dis
        self.local_n_dis = args.local_n_dis
        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.ckpt_path = args.ckpt_path
        self.tensorboard_path = args.tensorboard_path
        self.image_path = args.image_path

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", 'Horses <-> Zebras')
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iterations)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# Global discriminator layer : ", self.global_n_dis)
        print("# Local discriminator layer : ", self.local_n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build(self):

        # Dataset
        dataset, _ = tfds.load('cycle_gan/horse2zebra',
                               with_info=True, as_supervised=True)

        train_A, train_B = dataset['trainA'], dataset['trainB']
        test_A, test_B = dataset['testA'], dataset['testB']

        train_A = train_A.repeat()
        train_A = train_A.map(lambda x, y: train_augment(x, y, self.img_size), num_parallel_calls=AUTOTUNE)
        train_A = train_A.batch(self.batch_size)
        train_A = train_A.prefetch(AUTOTUNE)
        train_B = train_B.repeat()
        train_B = train_B.map(lambda x, y: train_augment(x, y, self.img_size), num_parallel_calls=AUTOTUNE)
        train_B = train_B.batch(self.batch_size)
        train_B = train_B.prefetch(AUTOTUNE)
        train_dataset = tf.data.Dataset.zip((train_A, train_B))
        self.train_iterator = iter(train_dataset)

        test_A = test_A.map(lambda x, y: test_augment(x, y, self.img_size), num_parallel_calls=AUTOTUNE)
        test_A = test_A.batch(self.sample_num)
        # test_B = test_B.map(test_augment, num_parallel_calls=AUTOTUNE)
        # test_B = test_B.batch(self.batch_size // 2)
        # test_dataset = tf.data.Dataset.zip((test_A, test_B))
        self.test_iterator = iter(test_A) # only Horses -> Zebras

        # Model building (Total 6 models)
        self.genA2B = ResnetGenerator(output_nc=3, ngf=self.ch, n_blocks=self.n_res, light=self.light)  # A -> B gen
        self.genB2A = ResnetGenerator(output_nc=3, ngf=self.ch, n_blocks=self.n_res, light=self.light)  # B -> A gen
        self.disGA = Discriminator(ndf=self.ch, n_layers=self.global_n_dis)  # Global A disc
        self.disGB = Discriminator(ndf=self.ch, n_layers=self.global_n_dis)  # Global B disc
        self.disLA = Discriminator(ndf=self.ch, n_layers=self.local_n_dis)  # Local A disc
        self.disLB = Discriminator(ndf=self.ch, n_layers=self.local_n_dis)  # Local B disc

        # Optimizer
        self.gen_opt = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.5, beta_2=0.999)
        self.disc_opt = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.5, beta_2=0.999)

        # paths
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.ckpt_path = os.path.join(current_time, self.ckpt_path)
        self.tensorboard_path = os.path.join(current_time, self.tensorboard_path)
        self.img_save_path = os.path.join(current_time, self.img_save_path)
        os.makedirs(self.tensorboard_path)
        os.makedirs(self.img_save_path)

        # ckpt manager
        ckpt = tf.train.Checkpoint(genA2B=self.genA2B,
                                   genB2A=self.genB2A,
                                   disGA=self.disGA,
                                   disGB=self.disGB,
                                   disLA=self.disLA,
                                   disLB=self.disLB)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, self.ckpt_path, max_to_keep=4)

        # summary writer
        train_log_path = os.path.join(self.tensorboard_path, 'train')
        self.train_summary_writer = tf.summary.create_file_writer(train_log_path)
        print(f'******* Train result log will be written to {train_log_path} ******')
        test_log_path = os.path.join(self.tensorboard_path, 'test')
        self.test_summary_writer = tf.summary.create_file_writer(test_log_path)
        print(f'******* Test result log will be written to {test_log_path} ******')

    def train(self):

        print('training start !')

        # for each epoch
        for iter in range(1, self.iterations + 1):

            start_time = time.time()

            real_A, real_B = self.train_iterator.get_next()

            # discriminator update
            with tf.GradientTape() as disc_tape:

                fake_A2B, _, _ = self.genA2B(real_A)
                fake_B2A, _, _ = self.genB2A(real_B)

                real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
                real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
                real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
                real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

                D_ad_loss_GA = ad_loss(real_GA_logit, tf.ones_like(real_GA_logit, dtype=tf.float32)) + ad_loss(
                    fake_GA_logit, tf.zeros_like(fake_GA_logit, dtype=tf.float32))
                D_ad_cam_loss_GA = ad_loss(real_GA_cam_logit,
                                           tf.ones_like(real_GA_cam_logit, dtype=tf.float32)) + ad_loss(
                    fake_GA_cam_logit, tf.zeros_like(fake_GA_cam_logit, dtype=tf.float32))
                D_ad_loss_LA = ad_loss(real_LA_logit, tf.ones_like(real_LA_logit, dtype=tf.float32)) + ad_loss(
                    fake_LA_logit, tf.zeros_like(fake_LA_logit, dtype=tf.float32))
                D_ad_cam_loss_LA = ad_loss(real_LA_cam_logit,
                                           tf.ones_like(real_LA_cam_logit, dtype=tf.float32)) + ad_loss(
                    fake_LA_cam_logit, tf.zeros_like(fake_LA_cam_logit, dtype=tf.float32))
                D_ad_loss_GB = ad_loss(real_GB_logit, tf.ones_like(real_GB_logit, dtype=tf.float32)) + ad_loss(
                    fake_GB_logit, tf.zeros_like(fake_GB_logit, dtype=tf.float32))
                D_ad_cam_loss_GB = ad_loss(real_GB_cam_logit,
                                           tf.ones_like(real_GB_cam_logit, dtype=tf.float32)) + ad_loss(
                    fake_GB_cam_logit, tf.zeros_like(fake_GB_cam_logit, dtype=tf.float32))
                D_ad_loss_LB = ad_loss(real_LB_logit, tf.ones_like(real_LB_logit, dtype=tf.float32)) + ad_loss(
                    fake_LB_logit, tf.zeros_like(fake_LB_logit, dtype=tf.float32))
                D_ad_cam_loss_LB = ad_loss(real_LB_cam_logit,
                                           tf.ones_like(real_LB_cam_logit, dtype=tf.float32)) + ad_loss(
                    fake_LB_cam_logit, tf.zeros_like(fake_LB_cam_logit, dtype=tf.float32))

                # print('loss1', D_ad_loss_GA, D_ad_cam_loss_GA)
                # print('loss2', D_ad_loss_LA,  D_ad_cam_loss_LA)

                D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
                D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)
                D_loss = D_loss_A + D_loss_B

                if iter == 1:
                    self.disc_tot_vars = []
                    for disc_model in [self.disGA, self.disGB, self.disLA, self.disLB]:
                        self.disc_tot_vars.extend(disc_model.trainable_variables)

            disc_grad = disc_tape.gradient(D_loss, self.disc_tot_vars)
            self.disc_opt.apply_gradients(zip(disc_grad, self.disc_tot_vars))

            # generator update
            with tf.GradientTape() as gen_tape:

                fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
                fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

                fake_A2B2A, _, _ = self.genB2A(fake_A2B)
                fake_B2A2B, _, _ = self.genA2B(fake_B2A)

                fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
                fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

                G_ad_loss_GA = ad_loss(fake_GA_logit, tf.ones_like(fake_GA_logit, dtype=tf.float32))
                G_ad_cam_loss_GA = ad_loss(fake_GA_cam_logit, tf.ones_like(fake_GA_cam_logit, dtype=tf.float32))
                G_ad_loss_LA = ad_loss(fake_LA_logit, tf.ones_like(fake_LA_logit, dtype=tf.float32))
                G_ad_cam_loss_LA = ad_loss(fake_LA_cam_logit, tf.ones_like(fake_LA_cam_logit, dtype=tf.float32))
                G_ad_loss_GB = ad_loss(fake_GB_logit, tf.ones_like(fake_GB_logit, dtype=tf.float32))
                G_ad_cam_loss_GB = ad_loss(fake_GB_cam_logit, tf.ones_like(fake_GB_cam_logit, dtype=tf.float32))
                G_ad_loss_LB = ad_loss(fake_LB_logit, tf.ones_like(fake_LB_logit, dtype=tf.float32))
                G_ad_cam_loss_LB = ad_loss(fake_LB_cam_logit, tf.ones_like(fake_LB_cam_logit, dtype=tf.float32))

                G_recon_loss_A = recon_loss(fake_A2B2A, real_A)
                G_recon_loss_B = recon_loss(fake_B2A2B, real_B)

                G_identity_loss_A = id_loss(fake_A2A, real_A)
                G_identity_loss_B = id_loss(fake_B2B, real_B)

                G_cam_loss_A = bce_loss(fake_B2A_cam_logit, tf.ones_like(fake_B2A_cam_logit)) + bce_loss(
                    fake_A2A_cam_logit, tf.zeros_like(fake_A2A_cam_logit))
                G_cam_loss_B = bce_loss(fake_A2B_cam_logit, tf.ones_like(fake_A2B_cam_logit)) + bce_loss(
                    fake_B2B_cam_logit, tf.zeros_like(fake_B2B_cam_logit))

                G_loss_A = self.adv_weight * (
                            G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
                G_loss_B = self.adv_weight * (
                            G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B
                G_loss = G_loss_A + G_loss_B

                if iter == 1:
                    self.gen_tot_vars = []
                    self.gen_tot_vars = self.genA2B.trainable_variables + self.genB2A.trainable_variables

            gen_grad = gen_tape.gradient(G_loss, self.gen_tot_vars)
            self.gen_opt.apply_gradients(zip(gen_grad, self.gen_tot_vars))

            with self.train_summary_writer.as_default():
                tf.summary.scalar('Discriminator_loss', D_loss, step=iter)
                tf.summary.scalar('Generator_loss', G_loss, step=iter)

            if iter % self.print_freq == 0:
                print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (
                iter, self.iterations, time.time() - start_time, D_loss, G_loss))

            if iter % self.print_freq == 0:  # Only A->B
                real_A = self.test_iterator.get_next()
                fake_A2B, _, _ = self.genA2B(real_A)
                image_save(real_A, fake_A2B, self.image_path, iter)

            if iter % self.save_freq == 0:
                self.ckpt_manager.save(checkpoint_number=iter)
                print(f'******* {str(iter)} checkpoint saved to {self.ckpt_path} ******')

    def test(self):
        ''' to be implemeneted '''
