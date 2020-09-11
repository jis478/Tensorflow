
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from ops import *
from utils import *
from model import *
import datetime
import time

AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.config.experimental_run_functions_eagerly(True)



class UGATIT(object):
    def __init__(self, args):
        self.phase = args.phase
        self.light = args.light
        self.iterations = args.iterations
        self.batch_size = args.batch_size
        self.sample_num = args.sample_num
        self.print_freq = args.print_freq
        self.sample_freq = args.sample_freq
        self.save_freq = args.save_freq
        self.dataset = args.dataset

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        self.ch = args.ch
        self.n_res = args.n_res
        self.n_dis = args.n_dis
        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.ckpt_path = args.ckpt_path
        self.tensorboard_path = args.tensorboard_path
        self.img_save_path = args.img_save_path

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iterations : ", self.iterations)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

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
        if self.dataset == 'dataset':
            train_A = tf.data.Dataset.list_files(f"./{self.dataset}/trainA/*.jpg")
            train_A = train_A.map(lambda x: train_augment(x, self.img_size), num_parallel_calls=AUTOTUNE)
            train_A = train_A.repeat()
            train_A = train_A.batch(self.batch_size)
            train_A = train_A.prefetch(AUTOTUNE)
            train_B = tf.data.Dataset.list_files(f"./{self.dataset}/trainB/*.jpg")
            train_B = train_B.map(lambda x: train_augment(x, self.img_size), num_parallel_calls=AUTOTUNE)
            train_B = train_B.repeat()
            train_B = train_B.batch(self.batch_size)
            train_B = train_B.prefetch(AUTOTUNE)
            train_dataset = tf.data.Dataset.zip((train_A, train_B))
            self.train_iterator = iter(train_dataset)

            test_A = tf.data.Dataset.list_files(f"./{self.dataset}/testA/*.jpg")
            test_A = test_A.map(lambda x: test_augment(x, self.img_size), num_parallel_calls=AUTOTUNE)
            test_A = test_A.batch(self.sample_num)
            self.test_dataset = test_A.prefetch(AUTOTUNE)

        elif self.dataset == 'horse2zebra':
            dataset, _ = tfds.load('cycle_gan/horse2zebra', with_info=True, as_supervised=True)
            train_A, train_B = dataset['trainA'], dataset['trainB']
            test_A, test_B = dataset['testA'], dataset['testB']
            train_A = train_A.map(lambda x, y: train_augment_horse2zebra(x, y, self.img_size), num_parallel_calls=AUTOTUNE)
            train_A = train_A.repeat()
            train_A = train_A.batch(self.batch_size)
            train_A = train_A.prefetch(AUTOTUNE)
            train_B = train_B.map(lambda x, y: train_augment_horse2zebra(x, y, self.img_size), num_parallel_calls=AUTOTUNE)
            train_B = train_B.repeat()
            train_B = train_B.batch(self.batch_size)
            train_B = train_B.prefetch(AUTOTUNE)
            train_dataset = tf.data.Dataset.zip((train_A, train_B))
            self.train_iterator = iter(train_dataset)

            test_A = test_A.map(lambda x, y: test_augment_horse2zebra(x, y, self.img_size), num_parallel_calls=AUTOTUNE)
            test_A = test_A.batch(self.sample_num)
            self.test_dataset = test_A.prefetch(AUTOTUNE)

        # Model building (Total 6 models)
        self.genA2B = ResnetGenerator(output_nc=3, ngf=self.ch, n_blocks=self.n_res,
                                      weight_decay=self.weight_decay, light=self.light)  # A -> B gen
        self.genB2A = ResnetGenerator(output_nc=3, ngf=self.ch, n_blocks=self.n_res,
                                      weight_decay=self.weight_decay, light=self.light)  # B -> A gen
        self.disGA = GlobalDiscriminator(ndf=self.ch, n_layers=self.n_dis,
                                         weight_decay=self.weight_decay)  # Global A disc
        self.disGB = GlobalDiscriminator(ndf=self.ch, n_layers=self.n_dis,
                                         weight_decay=self.weight_decay)  # Global B disc
        self.disLA = LocalDiscriminator(ndf=self.ch, n_layers=self.n_dis,
                                        weight_decay=self.weight_decay)  # Local A disc
        self.disLB = LocalDiscriminator(ndf=self.ch, n_layers=self.n_dis,
                                        weight_decay=self.weight_decay)  # Local B disc

        if self.phase == 'train':
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

        elif self.phase == 'test':
            ckpt_path = self.ckpt_path.split('/')[0]
            self.img_save_path = os.path.join(ckpt_path, self.img_save_path)
            os.makedirs(self.img_save_path)

            ckpt = tf.train.Checkpoint(genA2B=self.genA2B)
            self.ckpt_manager = tf.train.CheckpointManager(ckpt, self.ckpt_path, max_to_keep=1)

            if self.ckpt_manager.latest_checkpoint:
                ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
                print('the latest checkpoint restored')
            else:
                print('checkpoint not restored')
                raise Except

    @tf.function 
    def train_step(self, real_A, real_B, iter):
        with tf.GradientTape(persistent=True) as tape:

            fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
            fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

            fake_A2B2A, _, _ = self.genB2A(fake_A2B)
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)

            fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
            fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

            real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
            real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
            real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
            real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            D_ad_loss_GA = ad_loss(real_GA_logit,tf.ones_like(real_GA_logit, dtype=tf.float32)) + ad_loss(fake_GA_logit,tf.zeros_like(fake_GA_logit, dtype=tf.float32))
            D_ad_cam_loss_GA = ad_loss(real_GA_cam_logit, tf.ones_like(real_GA_cam_logit, dtype=tf.float32)) + ad_loss(fake_GA_cam_logit, tf.zeros_like(fake_GA_cam_logit, dtype=tf.float32))
            D_ad_loss_LA = ad_loss(real_LA_logit, tf.ones_like(real_LA_logit, dtype=tf.float32)) + ad_loss(fake_LA_logit, tf.zeros_like(fake_LA_logit, dtype=tf.float32))
            D_ad_cam_loss_LA = ad_loss(real_LA_cam_logit, tf.ones_like(real_LA_cam_logit, dtype=tf.float32)) + ad_loss(fake_LA_cam_logit, tf.zeros_like(fake_LA_cam_logit, dtype=tf.float32))
            D_ad_loss_GB = ad_loss(real_GB_logit, tf.ones_like(real_GB_logit, dtype=tf.float32)) + ad_loss(fake_GB_logit,tf.zeros_like(fake_GB_logit, dtype=tf.float32))
            D_ad_cam_loss_GB = ad_loss(real_GB_cam_logit, tf.ones_like(real_GB_cam_logit, dtype=tf.float32)) + ad_loss(fake_GB_cam_logit, tf.zeros_like(fake_GB_cam_logit, dtype=tf.float32))
            D_ad_loss_LB = ad_loss(real_LB_logit, tf.ones_like(real_LB_logit, dtype=tf.float32)) + ad_loss(fake_LB_logit, tf.zeros_like(fake_LB_logit, dtype=tf.float32))
            D_ad_cam_loss_LB = ad_loss(real_LB_cam_logit, tf.ones_like(real_LB_cam_logit, dtype=tf.float32)) + ad_loss(fake_LB_cam_logit, tf.zeros_like(fake_LB_cam_logit, dtype=tf.float32))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)
            D_reg_loss = tf.math.add_n(self.disGA.losses) + tf.math.add_n(self.disLA.losses) + tf.math.add_n(self.disGB.losses) + tf.math.add_n(self.disLB.losses)
            D_loss = D_loss_A + D_loss_B + D_reg_loss
            self.D_loss_tot += D_loss

            if iter == 1:
              self.disc_tot_vars = []
              for disc_model in [self.disGA, self.disGB, self.disLA, self.disLB]:
                self.disc_tot_vars.extend(disc_model.trainable_variables)

            G_ad_loss_GA = ad_loss(fake_GA_logit,tf.ones_like(fake_GA_logit, dtype=tf.float32))
            G_ad_cam_loss_GA = ad_loss(fake_GA_cam_logit, tf.ones_like(fake_GA_cam_logit, dtype=tf.float32))
            G_ad_loss_LA = ad_loss(fake_LA_logit, tf.ones_like(fake_LA_logit, dtype=tf.float32))
            G_ad_cam_loss_LA = ad_loss(fake_LA_cam_logit, tf.ones_like(fake_LA_cam_logit, dtype=tf.float32))
            G_ad_loss_GB = ad_loss(fake_GB_logit,tf.ones_like(fake_GB_logit, dtype=tf.float32))
            G_ad_cam_loss_GB = ad_loss(fake_GB_cam_logit, tf.ones_like(fake_GB_cam_logit, dtype=tf.float32))
            G_ad_loss_LB = ad_loss(fake_LB_logit, tf.ones_like(fake_LB_logit, dtype=tf.float32))
            G_ad_cam_loss_LB = ad_loss(fake_LB_cam_logit, tf.ones_like(fake_LB_cam_logit, dtype=tf.float32))
            G_recon_loss_A = recon_loss(fake_A2B2A, real_A)
            G_recon_loss_B = recon_loss(fake_B2A2B, real_B)
            G_identity_loss_A = id_loss(fake_A2A, real_A)
            G_identity_loss_B = id_loss(fake_B2B, real_B)
            G_cam_loss_A = cam_loss(fake_B2A_cam_logit, fake_A2A_cam_logit)
            G_cam_loss_B = cam_loss(fake_A2B_cam_logit, fake_B2B_cam_logit)

            G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B
            G_reg_loss = tf.math.add_n(self.genA2B.losses) + tf.math.add_n(self.genB2A.losses)
            G_loss = G_loss_A + G_loss_B + G_reg_loss
            self.G_loss_tot += G_loss

            if iter == 1:
              self.gen_tot_vars=[]
              self.gen_tot_vars=self.genA2B.trainable_variables + self.genB2A.trainable_variables

        disc_grad = tape.gradient(D_loss, self.disc_tot_vars)
        self.disc_opt.apply_gradients(zip(disc_grad, self.disc_tot_vars))

        gen_grad = tape.gradient(G_loss, self.gen_tot_vars)
        self.gen_opt.apply_gradients(zip(gen_grad, self.gen_tot_vars))

    def train(self):

      print('training start !')
      self.G_loss_tot = 0
      self.D_loss_tot = 0

      for iter in range(1, self.iterations+1):

          start_time = time.time()
          real_A, real_B = self.train_iterator.get_next()
          self.train_step(real_A, real_B, iter)

          with self.train_summary_writer.as_default():
            tf.summary.scalar('Discriminator_loss', self.D_loss_tot/iter*len(real_A), step=iter)
            tf.summary.scalar('Generator_loss', self.G_loss_tot/iter*len(real_A), step=iter)

          if iter % self.print_freq == 0:
            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (iter, self.iterations, time.time() - start_time, self.D_loss_tot/iter*len(real_A), self.G_loss_tot/iter*len(real_A)))

          if iter % self.sample_freq == 0: # Only A->B train sample
            real_A, _ = self.train_iterator.get_next()
            fake_A2B, _, _ = self.genA2B(real_A)
            image_save(real_A, fake_A2B, self.img_save_path, iter)

          if iter % self.save_freq == 0:
            self.ckpt_manager.save(checkpoint_number=iter)
            print(f'******* {str(iter)} checkpoint saved to {self.ckpt_path} ******')

    def test(self):
        try:
            idx = 1
            for real_A in self.test_dataset:
                fake_A2B, _, _ = self.genA2B(real_A)
                image_save(real_A, fake_A2B, self.img_save_path, idx)
                idx += 1
        except tf.errors.OutOfRangeError:
            print(f"Inference finished. Please check {self.image_save_path}")

            
            
            
