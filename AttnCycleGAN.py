import os, time, pdb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow import keras

from dataset import ImageData
from my_logger import MyLogger

from loss_fn import *
from utils import set_directions, LinearDecay, normalization
from networks import build_generator, build_discriminator
from vgg16 import VGG16Loss

class AttnCycleGAN(keras.Model):
    def __init__(self, args):
        super().__init__()
        """Train parameter"""
        self.args = args
        self.experiment_id = args["experiment_id"]
        self.networks = args["networks"]
        
        # size
        self.batch_size = args["batch_size"]
        self.image_size = args["image_size"]
        self.image_channels = args["image_channels"]
        self.image_shape = [self.image_size, self.image_size, self.image_channels]
        
        # frequency
        self.print_freq = args["print_freq"]
        self.save_freq = args["save_freq"]
        self.sample_freq = args["sample_freq"]
        
        # directory
        self.dataset_dir = args["dataset_dir"]
        
        self.load_ckpt = os.path.join(args["load_ckpt"], args["experiment_id"]) if args["experiment_id"] else args["load_ckpt"]
        self.checkpoint_dir = os.path.join(args["checkpoint_dir"], args["experiment_id"]) if args["experiment_id"] else args["checkpoint_dir"]
        self.log_dir = args["log_dir"]
        self.sample_dir = os.path.join(args["sample_dir"], args["experiment_id"]) if args["experiment_id"] else args["sample_dir"]
        self.test_dir = os.path.join(args["test_dir"], args["experiment_id"]) if args["experiment_id"] else args["test_dir"]
        
        """Set directories"""
        set_directions([self.checkpoint_dir, self.log_dir, self.sample_dir, self.test_dir])
        self._init_logger(log_fname=self.args["experiment_id"]+".log")
        
        """Build Model"""
        self.G_generator = build_generator(self.image_shape, self.networks)
        print(self.G_generator.summary())
        self.F_generator = build_generator(self.image_shape, self.networks)
        self.Dy_discriminator = build_discriminator(self.image_shape, self.networks)
        self.Dx_discriminator = build_discriminator(self.image_shape, self.networks)
        
        """Make dataset"""
        # train dataset
        if self.args["learning_mode"] == "TRAIN":
            self.A_train_dataset_dir = self.dataset_dir + f"/train/{self.args['domain_a']}"
            self.A_train_dataset = self.load_dataset(self.A_train_dataset_dir, training=True)
            self.B_train_dataset_dir = self.dataset_dir + f"/train/{self.args['domain_b']}"
            self.B_train_dataset = self.load_dataset(self.B_train_dataset_dir, training=True)
            self.train_dataset = tf.data.Dataset.zip((self.A_train_dataset, self.B_train_dataset))
            self.train_dataset_length = ImageData(self.A_train_dataset_dir, self.batch_size, self.image_size, training=True).len_dataset
        
        # test dataset
        self.A_test_dataset_dir = self.dataset_dir + f"/validation/{self.args['domain_a']}"
        self.A_test_dataset = self.load_dataset(self.A_test_dataset_dir, training=False)
        self.B_test_dataset_dir = self.dataset_dir + f"/validation/{self.args['domain_b']}"
        self.B_test_dataset = self.load_dataset(self.B_test_dataset_dir, training=False)
        self.test_dataset = iter(tf.data.Dataset.zip((self.A_test_dataset, self.B_test_dataset)))
        
        # train loop
        self.epochs = args['epochs']
        self.total_iteration = int(self.epochs * self.train_dataset_length / self.batch_size)
        self.iteration_decay = int(self.total_iteration/2)
        
        self.beta_1 = args['beta_1']
        
        """Set optimizer"""
        self.G_gen_optimizer = self.optimizer(self.args["G_learning_rate"])
        self.F_gen_optimizer = self.optimizer(self.args["F_learning_rate"])
        self.Dy_disc_optimizer = self.optimizer(self.args["G_learning_rate"])
        self.Dx_disc_optimizer = self.optimizer(self.args["F_learning_rate"])
        
        """Set checkpoint"""
        self.set_checkpoint()
        
        # content loss function
        self.content_loss_fn = VGG16Loss(input_shape=[self.image_size, self.image_size, 3])
    
    def _init_logger(self, log_fname):
        my_logger = MyLogger(self.log_dir, log_fname=log_fname)
        self.logger = my_logger.logger

        self.logger.info("[Argument Information]")
        self.logger.info("Current folder path: {}".format(os.path.abspath(os.getcwd())))
        for k, v in self.args.items():
            self.logger.info("{}: {}".format(k, v))
        
    def load_dataset(self, dataset_dir, training=True):
        dataset = ImageData(dataset_dir,
                            self.batch_size,
                            self.image_size,
                            training=training)
        dataset = dataset.make_dataset()
        
        return dataset
    
    def set_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(G_generator=self.G_generator,
                                              F_generator=self.F_generator,
                                              Dx_discriminator=self.Dx_discriminator,
                                              Dy_discriminator=self.Dy_discriminator,
                                              G_gen_optimizer=self.G_gen_optimizer,
                                              F_gen_optimizer=self.F_gen_optimizer,
                                              Dy_disc_optimizer=self.Dy_disc_optimizer,
                                              Dx_disc_optimizer=self.Dx_disc_optimizer,
                                              epoch=tf.Variable(0, trainable=False, dtype=tf.int32),
                                              step=tf.Variable(0, trainable=False, dtype=tf.int32))
        
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                       directory=self.checkpoint_dir,
                                                       max_to_keep=None,)
        
    def load_weights(self):
        self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            self.logger.info("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
            self.iteration = int(self.checkpoint.step)
            self.epoch = int(self.iteration / (self.train_dataset_length))
            
        else:
            self.logger.info("Initializing from scratch.")
            self.iteration = 0
    
    def optimizer(self, learning_rate, name='Adam'):            
        lr_schedule = LinearDecay(learning_rate, self.total_iteration, self.iteration_decay)
        return keras.optimizers.Adam(lr_schedule, name=name, beta_1=self.beta_1)
            
    def init_total_loss(self,):
        self.num_batch = 0
        self.G_gen_total_loss = 0.0
        self.G_gdl_total_loss = 0.
        self.G_content_total_loss = 0.
        self.G_ssim_total_loss = 0.
        self.G_voxel_total_loss = 0.
        self.G_identical_total_loss = 0.
        
        self.F_gen_total_loss = 0.0
        self.F_gdl_total_loss = 0.
        self.F_content_total_loss = 0.
        self.F_ssim_total_loss = 0.
        self.F_voxel_total_loss = 0.
        self.F_identical_total_loss = 0.
        
        self.Dy_disc_total_loss = 0.0
        self.Dx_disc_total_loss = 0.0
        
    def train(self):
        self.logger.info("[*] Train is start")
        self.load_weights()
        
        self.init_total_loss()
        epoch_time = time.time()
        iter_time = time.time()
        
        if self.iteration != 0:
            self.epoch = int(self.iteration / (self.train_dataset_length))
        else:
            self.epoch = 0
        
        for _ in range(self.epochs):
            self.logger.info("Epoch: [{}/{}], sec:{:.2f}".format(self.epoch, self.epochs, time.time()-epoch_time))
            
            for x_img, y_img in self.train_dataset:
                self.num_batch += 1
                self.checkpoint.step.assign_add(1)
                self.iteration +=1
                
                loss_dict = self.train_step(x_img, y_img)
                
                self.G_gen_total_loss += loss_dict["G_generator_loss"]
                self.G_gdl_total_loss += loss_dict["G_gdl_loss"]
                self.G_content_total_loss += loss_dict["G_content_loss"]
                self.G_ssim_total_loss += loss_dict["G_ssim_loss"]
                self.G_voxel_total_loss += loss_dict["G_voxel_loss"]
                self.G_identical_total_loss += loss_dict["G_identical_loss"]
                
                self.F_gen_total_loss += loss_dict["F_generator_loss"]
                self.F_gdl_total_loss += loss_dict["F_gdl_loss"]
                self.F_content_total_loss += loss_dict["F_content_loss"]
                self.F_ssim_total_loss += loss_dict["F_ssim_loss"]
                self.F_voxel_total_loss += loss_dict["F_voxel_loss"]
                self.F_identical_total_loss += loss_dict["F_identical_loss"]
                
                self.Dy_disc_total_loss += loss_dict["Dy_discriminator_loss"]
                self.Dx_disc_total_loss += loss_dict["Dx_discriminator_loss"]
                
                if self.iteration%self.print_freq==0:
                    self.logger.info("Iteration: [{}/{}], sec: {:.2f}".format(self.iteration, self.total_iteration, time.time()-iter_time))
                    
                    G_gen_mean_loss = self.G_gen_total_loss / self.num_batch
                    G_gdl_mean_loss = self.G_gdl_total_loss / self.num_batch
                    G_content_mean_loss = self.G_content_total_loss / self.num_batch
                    G_ssim_mean_loss = self.G_ssim_total_loss / self.num_batch
                    G_voxel_mean_loss = self.G_voxel_total_loss / self.num_batch
                    G_identical_mean_loss = self.G_identical_total_loss / self.num_batch
                    
                    F_gen_mean_loss = self.F_gen_total_loss / self.num_batch
                    F_gdl_mean_loss = self.F_gdl_total_loss / self.num_batch
                    F_content_mean_loss = self.F_content_total_loss / self.num_batch
                    F_ssim_mean_loss = self.F_ssim_total_loss / self.num_batch
                    F_voxel_mean_loss = self.F_voxel_total_loss / self.num_batch
                    F_identical_mean_loss = self.F_identical_total_loss / self.num_batch
                    
                    Dy_disc_mean_loss = self.Dy_disc_total_loss / self.num_batch
                    Dx_disc_mean_loss = self.Dx_disc_total_loss / self.num_batch
                
                    self.logger.info("G_Generator loss mean: %.3f" % (float(G_gen_mean_loss),))
                    self.logger.info("F_Generator loss mean: %.3f" % (float(F_gen_mean_loss),))
                    
                    self.logger.info("G_GDL loss mean: %.3f" % (float(G_gdl_mean_loss),))
                    self.logger.info("F_GDL loss mean: %.3f" % (float(F_gdl_mean_loss),))
                    
                    self.logger.info("G_Content loss mean: %.3f" % (float(G_content_mean_loss),))
                    self.logger.info("F_Content loss mean: %.3f" % (float(F_content_mean_loss),))
                    
                    self.logger.info("G_SSIM loss mean: %.3f" % (float(G_ssim_mean_loss),))
                    self.logger.info("F_SSIM loss mean: %.3f" % (float(F_ssim_mean_loss),))
                    
                    self.logger.info("G_Voxel loss mean: %.3f" % (float(G_voxel_mean_loss),))
                    self.logger.info("F_Voxel loss mean: %.3f" % (float(F_voxel_mean_loss),))
                    
                    self.logger.info("G_Identical loss mean: %.3f" % (float(G_identical_mean_loss),))
                    self.logger.info("F_Identical loss mean: %.3f" % (float(F_identical_mean_loss),))
                    self.logger.info("\n")
                    
                    self.logger.info("Dy_Discriminator loss mean: %.3f" % (float(Dy_disc_mean_loss),))
                    self.logger.info("Dx_Discriminator loss mean: %.3f" % (float(Dx_disc_mean_loss),))
                    self.logger.info("\n")
                    
                    iter_time = time.time()
                
                sample_freq = self.sample_freq if self.iteration > 10000 else 100
                if (self.iteration%sample_freq==0):
                    x_img_test_dataset, y_img_test_dataset = next(self.test_dataset)
                    x_img, _ = x_img_test_dataset
                    y_img, _ = y_img_test_dataset
                    
                    self.generate_and_save_images(x_img, y_img, self.iteration)
                    self.init_total_loss()
            
            self.epoch += 1
            
            if self.epoch == 1:
                self.ckpt_manager.save(int(self.epoch))
            
            if self.epoch % self.save_freq == 0:
                self.ckpt_manager.save(int(self.epoch))
            
            if self.epoch == self.epochs:
                self.logger.info("[!] Train is end!!")
                break
            
            epoch_time = time.time()
            self.logger.info("[!] Train is end!!")
            
    @ tf.function
    def train_step(self, x_img, y_img):          
        with tf.GradientTape(persistent=True) as tape:
            # fake image
            fake_x_img = self.F_generator(y_img, training=True)
            fake_y_img = self.G_generator(x_img, training=True)
            
            # fake logit
            disc_fake_x = self.Dx_discriminator(fake_x_img, training=True)
            disc_fake_y = self.Dy_discriminator(fake_y_img, training=True)

            # real logit
            disc_real_x = self.Dx_discriminator(x_img, training=True)
            disc_real_y = self.Dy_discriminator(y_img, training=True)            
            
            ### Discriminator total loss ###
            disc_x_loss = d_loss_fn(disc_real_x, disc_fake_x)
            disc_y_loss = d_loss_fn(disc_real_y, disc_fake_y)
            
            # reconstructed image
            rec_x_img = self.F_generator(fake_y_img, training=True)
            rec_y_img = self.G_generator(fake_x_img, training=True)
            
            # identical image
            same_x_img = self.F_generator(x_img, training=True)
            same_y_img = self.G_generator(y_img, training=True)
            
            ### Generator Loss ###
            # adv loss
            gen_g_loss = g_loss_fn(disc_fake_y)
            gen_f_loss = g_loss_fn(disc_fake_x)
            
            # cycle loss
            cycle_g_loss = self.args["Cycle_weight"]*cycle_loss_fn(y_img, rec_y_img)
            cycle_f_loss = self.args["Cycle_weight"]*cycle_loss_fn(x_img, rec_x_img)
            cycle_loss = (cycle_g_loss + cycle_f_loss)
            
            # gradient loss
            gdl_g_loss = self.args["GDL_weight"]*gradient_difference_loss_fn(y_img, fake_y_img)
            gdl_f_loss = self.args["GDL_weight"]*gradient_difference_loss_fn(x_img, fake_x_img)
            
            # content(perceputual) loss
            content_g_loss = self.content_loss_fn(y_img, fake_y_img)
            content_f_loss = self.content_loss_fn(x_img, fake_x_img)
            
            # ssim loss
            ssim_g_loss = self.args["SSIM_weight"]*ssim_loss_fn(y_img, fake_y_img)
            ssim_f_loss = self.args["SSIM_weight"]*ssim_loss_fn(x_img, fake_x_img)
            
            # voxel loss(L1 loss)
            voxel_g_loss = self.args["L1_weight"]*voxel_loss_fn(y_img, fake_y_img)
            voxel_f_loss = self.args["L1_weight"]*voxel_loss_fn(x_img, fake_x_img)
            
            # identical loss
            identical_g_loss = self.args["L1_weight"]*voxel_loss_fn(y_img, same_y_img)
            identical_f_loss = self.args["L1_weight"]*voxel_loss_fn(x_img, same_x_img)
            
            ### Generator total loss ###
            total_gen_g_loss = gen_g_loss + cycle_loss + gdl_g_loss + content_g_loss + ssim_g_loss + voxel_g_loss + identical_g_loss
            total_gen_f_loss = gen_f_loss + cycle_loss + gdl_f_loss + content_f_loss + ssim_f_loss + voxel_f_loss + identical_f_loss
        
        # Generator gradient backpropagation    
        G_generator_gradients = tape.gradient(total_gen_g_loss, self.G_generator.trainable_variables)
        F_generator_gradients = tape.gradient(total_gen_f_loss, self.F_generator.trainable_variables)
        self.G_gen_optimizer.apply_gradients(zip(G_generator_gradients, self.G_generator.trainable_variables))
        self.F_gen_optimizer.apply_gradients(zip(F_generator_gradients, self.F_generator.trainable_variables))
        
        # Discriminator gradient backpropagation
        Dy_discriminator_gradients = tape.gradient(disc_y_loss, self.Dy_discriminator.trainable_variables)
        Dx_discriminator_gradients = tape.gradient(disc_x_loss, self.Dx_discriminator.trainable_variables)
        self.Dy_disc_optimizer.apply_gradients(zip(Dy_discriminator_gradients, self.Dy_discriminator.trainable_variables))
        self.Dx_disc_optimizer.apply_gradients(zip(Dx_discriminator_gradients, self.Dx_discriminator.trainable_variables))
            
        return {
            "G_generator_loss": gen_g_loss,
            "F_generator_loss": gen_f_loss,
            "Cycle_loss": cycle_loss,
            "G_gdl_loss": gdl_g_loss,
            "F_gdl_loss": gdl_f_loss,
            "G_content_loss": content_g_loss,
            "F_content_loss": content_f_loss,
            "G_ssim_loss": ssim_g_loss,
            "F_ssim_loss": ssim_f_loss,
            "G_voxel_loss": voxel_g_loss,
            "F_voxel_loss": voxel_f_loss,
            "G_identical_loss": identical_g_loss,
            "F_identical_loss": identical_f_loss,
            "Dy_discriminator_loss": disc_y_loss,
            "Dx_discriminator_loss": disc_x_loss
        }
        
    def test(self):
        self.load_weights()
        start_time = time.time()
        total_time = 0
        test_length = 0
        subj_name_list = []
        for x_test, y_test in self.test_dataset:
            test_length += 1
            x_img, x_fname = x_test
            y_img, y_fname = y_test
            
            subj_name, slice_num = self.get_fname_info(x_fname)
            if not subj_name in subj_name_list:
                subj_name_list.append(subj_name)
            
            self.generate_and_save_images(x_img, y_img,
                                          self.iteration,
                                          training=False,
                                          subj_name=subj_name,
                                          slice_num=slice_num)
            time_diff = time.time() - start_time
            self.logger.info("Time: {:4f}".format(time_diff))
            
            total_time += time_diff
            start_time = time.time()
            
        self.logger.info("Subject list", subj_name_list)
        self.logger.info("Total test subject:", len(subj_name_list))
        self.logger.info("Total slice:", test_length)
        self.logger.info("Average conversion time: {:.3f} sec".format(total_time/test_length))
        
        
    def get_fname_info(self, fname):
        """
        fname: png file name, dtype: tensor
            - Returns
                - subject name(i.e S0964)
                - slicee number(0001)
        """
        fname = fname.numpy()
        fname = fname[0].decode()
        fname = os.path.basename(fname)
        fname = os.path.splitext(fname)[0]
        fname_split = fname.split("_")
        
        if len(fname_split)==3:
            subj_name = fname_split[0]
        elif len(fname_split)==4:
            subj_name = fname_split[1]
        slice_num = fname_split[-1]
        
        
        return subj_name, slice_num
    
    def generate_and_save_images(self, x_img, y_img, iteration, training=True, subj_name=None, slice_num=None):
        fake_y_img = self.G_generator(x_img, training=False)
        fake_x_img = self.F_generator(y_img, training=False)

        x_img = normalization(x_img[0, :, :, 0].numpy())
        y_img = normalization(y_img[0, :, :, 0].numpy())
        fake_y_img = normalization(fake_y_img[0, :, :, 0].numpy())
        fake_x_img = normalization(fake_x_img[0, :, :, 0].numpy())
        
        img_concat = np.concatenate((x_img, y_img, fake_y_img, fake_x_img), axis=1).reshape((self.image_size, self.image_size*4))
        img = Image.fromarray(np.uint8(img_concat))
        
        if training:
            fname = "sample_at_iteartion_{:04d}.png".format(iteration)
            save_path = os.path.join(self.sample_dir, fname)
        else:
            fname = "{}_{}_iteration_{:04d}.png".format(subj_name, slice_num, iteration)
            save_dir = os.path.join(self.test_dir, subj_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, fname)
        
        img.save(save_path)