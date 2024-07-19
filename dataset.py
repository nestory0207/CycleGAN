import os, time, pdb
from glob import glob
import numpy as np
import tensorflow as tf
import pydicom as pdc

from numpy.random import seed
seed(0)

class ImageData:
    def __init__(self, dataset_path, batch_size, img_size, training):
        self.dataset_path = dataset_path
        self.img_paths = sorted(glob(os.path.join(self.dataset_path, "*.png")))
        self.batch_size = batch_size
        self.img_size = img_size
        self.training = training
        self.len_dataset = len(self.img_paths)
        self.random_seed = tf.compat.v1.set_random_seed(seed=1004)
        
    def __len__(self):
        return len(self.img_paths)
    
    def make_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.img_paths)
        if self.training:
            dataset = dataset.shuffle(2048, seed=2023, reshuffle_each_iteration=True)
            dataset = dataset.map(self.map_fn, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
            dataset = dataset.prefetch(1)
        else:
            dataset = dataset.shuffle(100, seed=2023)
            dataset = dataset.map(self.map_fn, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(self.batch_size, drop_remainder=False)
            dataset = dataset.repeat().prefetch(1)
            
        return dataset
    
    @tf.function
    def map_fn(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=1, dct_method='INTEGER_ACCURATE')
        
        if self.training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.resize(img, [int(self.img_size * 1.1), int(self.img_size * 1.1)])
            img = tf.image.random_crop(img, [self.img_size, self.img_size, tf.shape(img)[-1]])
        
        else:
            img = tf.image.resize(img, [self.img_size, self.img_size])\
            
        img = adjust_dynamic_range(img)
        
        if self.training:
            return img
        
        else:
            return img, img_path
    
def adjust_dynamic_range(img):
    range_in = (tf.math.reduce_min(img), tf.math.reduce_max(img))
    range_out = (tf.cast(-1.0, dtype=tf.float32),tf.cast(1.0, dtype=tf.float32))

    scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
    bias = range_out[0] - range_in[0] * scale
    img = img * scale + bias
    img = tf.clip_by_value(img, range_out[0], range_out[1])
    img = tf.cast(img, dtype=tf.float32)
    
    return img


class DicomData:
    def __init__(self, dataset_path, batch_size, img_size, training):
        self.dataset_path = dataset_path
        # self.img_paths = sorted(glob(os.path.join(self.dataset_path, "*.dcm")))
        self.img_paths = self.get_img_paths()
        self.batch_size = batch_size
        self.img_size = img_size
        self.training = training
        self.len_dataset = len(self.img_paths)
        self.random_seed = tf.compat.v1.set_random_seed(seed=1004)
        
    def get_img_paths(self):
        test_subjects = []
        # test_subjects = ['S0011', 'S0013', 'S0018', 'S0057', 'S0065', 'S0067', 'S0068', 'S0072', 'S0080', 'S0088', 'S0098', 'S0112', 'S0119', 'S0121', 'S0130', 'S0132', 'S0137', 'S0143', 'S0171', 'S0196', 'S0216', 'S0217', 'S0220', 'S0244', 'S0260', 'S0272', 'S0278', 'S0281', 'S0292', 'S0309', 'S0316', 'S0323', 'S0327', 'S0346', 'S0348', 'S0355', 'S0359', 'S0360', 'S0380', 'S0385']
        # test_subjects = ['S0011']
        self.subject_paths = sorted(glob(os.path.join(self.dataset_path, "S*")))
        
        if len(test_subjects) != 0:
            self.subject_paths = sorted([i for i in self.subject_paths if os.path.basename(i) in test_subjects])
        else:
            self.subject_paths = sorted([i for i in self.subject_paths])
        
        img_paths = []
        for i in self.subject_paths:
            dicom_list = glob(os.path.join(i, "0.raw", "NECT", "*.dcm"))
            img_paths += dicom_list
            
        return img_paths
            
    def __len__(self):
        return len(self.img_paths)
    
    def make_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.img_paths)
        
        if self.training:
            dataset = dataset.shuffle(self.len_dataset, seed=2023, reshuffle_each_iteration=True)
            dataset = dataset.map(self.map_fn, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
            dataset = dataset.repeat().prefetch(1)
        
        return dataset
    
    def map_fn(self, img_path):
        ds = pdc.dcmread(img_path)
        img = ds.pixel_array
        img = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        img = np.where(img<-180, -180, np.where(img>200, 200, img))
        img = (((img - img.min()) / (img.max() - img.min())) * 255.9).astype(np.uint8)
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        
        if self.training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.resize(img, [int(self.img_size * 1.1), int(self.img_size * 1.1)])
            img = tf.image.random_crop(img, [self.img_size, self.img_size, tf.shape(img)[-1]])
        
        else:
            img = tf.expand_dims(img, axis=-1)
            img = tf.image.resize(img, [self.img_size, self.img_size])
            img = tf.reshape(img, shape=[self.batch_size, self.img_size, self.img_size, 1])
            
        img = adjust_dynamic_range(img)
        
        if self.training:
            return img
        
        else:
            return img, img_path
    
def adjust_dynamic_range(img):
    range_in = (tf.math.reduce_min(img), tf.math.reduce_max(img))
    range_out = (tf.cast(-1.0, dtype=tf.float32),tf.cast(1.0, dtype=tf.float32))

    scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
    bias = range_out[0] - range_in[0] * scale
    img = img * scale + bias
    img = tf.clip_by_value(img, range_out[0], range_out[1])
    img = tf.cast(img, dtype=tf.float32)
    
    return img