import os, time, pdb, datetime, cv2
import pydicom as pdc
import numpy as np, pdb
from PIL import Image

import tensorflow as tf
from tensorflow import keras

from my_logger import MyLogger
from dataset import ImageData
from loss_fn import *
from utils import set_directions, LinearDecay, normalization, mapping_series, restore_window
from networks import build_generator, build_discriminator
from vgg16 import VGG16Loss

from pydicom.encaps import encapsulate
from pydicom.uid import ImplicitVRLittleEndian

def inverse_transform(img):
    return (img + 1.) / 2.


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
        self.test_dir = os.path.join(self.test_dir, args["file_format"])
        self.test_dir = os.path.join(self.test_dir, "epoch_" + str(args["test_epoch"]).zfill(3))
        
        """Set directories"""
        set_directions([self.checkpoint_dir, self.log_dir, self.sample_dir, self.test_dir])
        if self.args["learning_mode"] == "TRAIN":
            self._init_logger(log_fname=self.args["experiment_id"]+".log")
        elif self.args["learning_mode"] == "TEST":
            self._init_logger(log_fname=self.args["experiment_id"]+"_test.log")
        
        """Build Model"""
        self.G_generator = build_generator(self.image_shape, self.networks)
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
        else:
            if  self.args["file_format"] == "png":
                self.A_test_dataset_dir = self.dataset_dir + f"/validation/{self.args['domain_a']}"
                self.A_test_dataset = self.load_dataset(self.A_test_dataset_dir, training=False)
                self.B_test_dataset_dir = self.dataset_dir + f"/validation/{self.args['domain_b']}"
                self.B_test_dataset = self.load_dataset(self.B_test_dataset_dir, training=False)
                self.test_dataset = iter(tf.data.Dataset.zip((self.A_test_dataset, self.B_test_dataset)))
        
        # train loop
        self.epochs = args['epochs']
        self.total_iteration = 0
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
        if self.ckpt_manager.latest_checkpoint:
            ckpt_path = f'{self.checkpoint_dir}/ckpt-{self.args["test_epoch"]}'
            self.checkpoint.restore(ckpt_path)
            self.logger.info("Restored from {}".format(ckpt_path))
            self.iteration = int(self.checkpoint.step)
            self.epoch = int(os.path.basename(ckpt_path).split("-")[-1])
            
        else:
            self.logger.info("Initializing from scratch.")
            self.iteration = 0
            self.epoch = 0
    
    def optimizer(self, learning_rate, name='Adam'):            
        lr_schedule = LinearDecay(learning_rate, self.total_iteration, self.iteration_decay)
        return keras.optimizers.Adam(lr_schedule, name=name, beta_1=self.beta_1)
        
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
            print("Time: {:4f}".format(time_diff))
            
            total_time += time_diff
            start_time = time.time()
            
        print("Subject list", subj_name_list)
        print("Total test subject:", len(subj_name_list))
        print("Total slice:", test_length)
        print("Average conversion time: {:.3f} sec".format(total_time/test_length))


    def test_dicom(self, dataset):
        self.load_weights()
        start_time = time.time()
        total_time = 0
        test_length = 0
        subj_name_list = []
        test_dataset = dataset.make_dataset()
        
        for fpath in test_dataset:
            fpath_decode = fpath.numpy().decode()
            x_img, x_fname = dataset.map_fn(fpath_decode)
            test_length += 1
            
            
            subj_name, slice_num = self.get_fname_info(fpath_decode)
            
            if not subj_name in subj_name_list:
                subj_name_list.append(subj_name)
            
            self.generate_dicom_images(x_img,
                                       fpath_decode,
                                       self.epoch,
                                       training=False,
                                       subj_name=subj_name,
                                       slice_num=slice_num)
            time_diff = time.time() - start_time
            print("Time: {:4f}".format(time_diff))
            
            total_time += time_diff
            start_time = time.time()
            
        print("Subject list", subj_name_list)
        print("Total test subject:", len(subj_name_list))
        # print("Total slice:", test_length)
        # print("Average conversion time: {:.3f} sec".format(total_time/test_length))
        
                
    def get_fname_info(self, fname):
        """
        fname: png file name, dtype: numpy!!!!!!!!
            - Returns
                - subject name(i.e S0964)
                - slicee number(0001)
        """
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
        
 
    def generate_dicom_images(self, x_img, dicom_path, epoch, training=True, subj_name=None, slice_num=None):
        ds = pdc.dcmread(dicom_path)
        fake_y_img = self.G_generator(x_img, training=False)
        fname = "{}_{}_epoch_{:04d}.dcm".format(subj_name, slice_num, epoch)
        save_dir = os.path.join(self.test_dir, subj_name)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, fname)
        
        img_arr = np.array(fake_y_img[0])
        img_arr = restore_window(img_arr, 10, 380)
        img_arr = img_arr - ds.RescaleIntercept
        img_arr = img_arr.astype(np.uint16)
        restored_min, restored_max = img_arr.min(), img_arr.max()
        img_arr = np.expand_dims(cv2.resize(img_arr, (ds.Rows, ds.Columns), interpolation=cv2.INTER_LANCZOS4), 2)
        img_arr = np.clip(img_arr, restored_min, restored_max)
        
        now = datetime.datetime.now()
        
        ds.WindowWidth = 380
        ds.WindowCenter = 10
        ds.StudyDescription = 'CT Chest (enhanced) from Generative AI'
        new_series = mapping_series(self.args["domain_b"])
        
        img_max = np.max(img_arr)
        img_min = np.min(img_arr)
        
        img_bytes = img_arr.tobytes()
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        ds.is_implicit_VR = True

        ds.Modality = "CT"
        # ds.Rows = self.image_size
        # ds.Columns = self.image_size

        ds.LargestImagePixelValue = img_max
        ds["LargestImagePixelValue"].VR = "SS"
        ds.SmallestImagePixelValue = img_min
        ds["SmallestImagePixelValue"].VR = "SS"
        
        ds.PixelData = img_bytes
        
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
        ds.StudyDate = now.strftime('%Y%m%d')
        ds.StudyTime = now.strftime('%H%M%S')

        ds.Manufacturer = 'Dr.Magic'
        ds.ManufacturerModelName = 'Poderosa Inicial v0.1'
        ds.SoftwareVersion = 'Polestar Health Care Inc.'
            
        uid_info = ds.SOPInstanceUID.split('.')
        uid_info[-1] = str(int(uid_info[-1]) + 200)
        uid_info[-2] = '1000'
        uid_info[-3] = str(int(uid_info[-3]) + 10)
        new_uid_info = '.'.join(uid_info)
        ds.SOPInstanceUID = new_uid_info
        study_uid = '.'.join(uid_info[:-2])
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = study_uid + new_series
        print("series instance UID :", ds.SeriesInstanceUID)
        
        ds.save_as(save_path)
        print("dcm save path : ", save_path)
        
        
        
        
                