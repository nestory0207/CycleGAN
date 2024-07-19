import os, cv2
import SimpleITK as sitk
import numpy as np
import tensorflow as tf

def normalization(img_arr, dtype=np.uint8):
    img_arr = (((img_arr - img_arr.min()) / (img_arr.max() - img_arr.min()))*255.9).astype(dtype)
    return img_arr

def restore_window(img_arr, window_center, window_width):
    img_arr = np.clip(img_arr, -1., 1.)
    img_arr = (img_arr + 1.)/2. * window_width
    img_arr = img_arr - window_width/2. + window_center
    return img_arr

def inverse_transform(img):
    return (img + 1.) / 2.

def automatic_gpu_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
            
def set_directions(dir_list):
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)
            
class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate
    
def mapping_series(target_domain):
    if target_domain == "GD_T":
        series = ".20.20"
    elif target_domain == "T2":
        series = ".20.21"
    elif target_domain == "CECT":
        series = ".20.22"
    elif target_domain == "T1":
        series = ".20.23"
    elif target_domain == "NECT":
        series = ".20.24"
    elif target_domain == "DeCBCT":
        series = ".21.20"
    elif target_domain == "PlanCT":
        series = ".21.21"
    else:
        series = ".20.25"

    return series

class Cropper():
    def __init__(self, target_size=512, window:bool=True, modality="Spine"):
        self.target_size = target_size
        self.window = window
        self.modality = modality
        self.get_window_range()
        
    def get_window_range(self):        
        if self.window:
            if self.modality == "Spine":
                self.window_range=  [-300, 500]
            elif self.modality == "Brain":
                self.window_range = [0, 80]

    def set_window(self, image):
        min_value, max_value = self.window_range
        image = np.where(image < min_value, min_value, np.where(image > max_value, max_value, image))
        return image

    def crop_image(self, image:np.ndarray, axis='sagittal'):
        """
        Arguments
        - img: An array of 3D volume
        - target_size: Target shape of the image to be cropped
        """
        d, h, w = image.shape
        assert h == w, f"Height and Width must be the same. Height: {h}, Width: {w}"
            
        if axis == "sagittal":
            depth = h
        elif axis == "axial":
            depth = d    
        
        z_range = [int((h-self.target_size)/2), int((h-self.target_size)/2)+self.target_size]
        cropped_slice_list = []
        
        # set width range
        for idx in range(depth):
            if axis == "sagittal":
                slc = image[:, idx, z_range[0]:z_range[1]]
            elif axis == 'axial':
                slc = image[idx, z_range[0]:z_range[1], z_range[0]:z_range[1]]
                
            if self.window:
                slc = self.set_window(slc)
                
            if axis == "sagittal":
                slc = cv2.rotate(slc, 0)
            cropped_slice_list.append(slc)
        return np.array(cropped_slice_list, dtype=np.int16)

class Resampler():
    def __init__(self):
        self.reader = sitk.ImageSeriesReader()
    
    def process(self, dicom_series_path):
        image = self.load_dicom_series(dicom_series_path)
        target_size = [*image.GetSize()[:-1], 512]
        resampled_img = self.resampling(img=image, target_size=target_size)
        resampled_img_arr = sitk.GetArrayFromImage(resampled_img)
        return resampled_img_arr
    
    def load_dicom_series(self, dicom_series_path):
        # DICOM 시리즈 로드
        dicom_series = self.reader.GetGDCMSeriesFileNames(dicom_series_path)

        # DICOM 시리즈를 SimpleITK 이미지로 변환
        sitk_image = sitk.ReadImage(dicom_series)
        return sitk_image
    
    def resampling(self, img:sitk.SimpleITK.Image, target_size, interpolator=sitk.sitkLinear):
        original_size = img.GetSize()
        original_spacing = img.GetSpacing()
        
        new_spacing = [round(ospc*osz/tsz, 3) for ospc, osz, tsz in zip(original_spacing, original_size, target_size)]
        resampled_img = sitk.Resample(img, target_size, sitk.Transform(), interpolator,
                                img.GetOrigin(), new_spacing, img.GetDirection(), 0,
                                img.GetPixelID())
        return resampled_img

def split_sqaure_brackets(input_string):
    result = [item.strip() for item in input_string.split('],')]
    result = [item.strip('[]').split(', ') for item in result]
    result = [[int(num) for num in sublist] for sublist in result]
    return result

def load_dicom_series(dicom_series_path):
    # DICOM 시리즈 로드
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_series_path)

    # DICOM 시리즈를 SimpleITK 이미지로 변환
    sitk_image = sitk.ReadImage(dicom_series)
    return sitk_image