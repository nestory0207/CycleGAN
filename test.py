import sys, os, argparse, pdb
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from device import DeviceController, set_cuda_visible
from utils import automatic_gpu_usage
from dataset import DicomData

def parse_args():
    parser = argparse.ArgumentParser("CycleGan")
    parser.add_argument('--experiment_id', type=str, default='base', help="")
    parser.add_argument('--file_format', type=str, default='png', help="")
    parser.add_argument('--networks', type=str, default='attention', help="[attention, basic]")
    
    parser.add_argument('--learning_mode', type=str, default='TEST', help="Select learning mode [TRAIN/TEST]")
    parser.add_argument('--gpu_num', type=int, default=1, help="GPU num")
    parser.add_argument('--dataset_dir', type=str, default='/data/nestory/png/Spine/Bucheon_CMC_RD/dataset', help="Dataset directory path")
    parser.add_argument('--domain_a', type=str, default='NECT', help="Dataset directory path")
    parser.add_argument('--domain_b', type=str, default='CECT', help="Dataset directory path")
    
    parser.add_argument('--image_size', type=int, default=256, help="Number of iterations")
    parser.add_argument('--image_channels', type=int, default=1, help="Number of iterations")
    
    parser.add_argument('--iteration', type=int, default=127100, help="Number of iterations")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--test_epoch', type=int, default=30, help="Number of epochs")
    parser.add_argument('--epoch_decay', type=int, default=50)
    parser.add_argument('--G_learning_rate', type=float, default=1e-4, help="Initial learning rate for Adam")
    parser.add_argument('--F_learning_rate', type=float, default=1e-4, help="Initial learning rate for Adam")
    parser.add_argument('--beta_1', type=float, default=0.5, help="Momentum term of Adam")
    
    parser.add_argument('--GDL_weight', type=float, default=100., help="Momentum term of Adam")
    parser.add_argument('--L1_weight', type=float, default=5., help="Momentum term of Adam")
    parser.add_argument('--SSIM_weight', type=float, default=1., help="Momentum term of Adam")
    parser.add_argument('--Cycle_weight', type=float, default=10., help="Momentum term of Adam")

    parser.add_argument('--print_freq', type=int, default=100, help="Print frequency for loss")
    parser.add_argument('--save_freq', type=int, default=12710, help="Save frequency for model")
    parser.add_argument('--sample_freq', type=int, default=500, help="Sampling frequency for saving image")

    parser.add_argument('--load_ckpt', type=str, default="", required=False, help="Folder of saved model that you wish to continue training")
    parser.add_argument('--checkpoint_dir', type=str, default='/data/nestory/model/main/Chest/AttnCycleGAN', help="Directory name to save the checkpoints")
    parser.add_argument('--log_dir', type=str, default='./logs', help="Directory name to save training logs")
    parser.add_argument('--sample_dir', type=str, default='./samples', help="Directory name to save the samples on training")
    parser.add_argument('--test_dir', type=str, default='./test', help="Directory name to save the samples on training")

    return parser.parse_args()

def main():
    args = vars(parse_args())
    args["dataset_dir"] = args["dataset_dir"] + f"/{args['domain_a']}2{args['domain_b']}"
    args["sample_dir"] = args["sample_dir"] + f"/image_size_{args['image_size']}"
    args["test_dir"] = args["test_dir"] + f"/image_size_{args['image_size']}"
    
    set_cuda_visible(args['gpu_num'])
    automatic_gpu_usage()
    
    training_parameters = {
    **args,
    }
    
    if args["file_format"] == "png":
        from AttnCycleGAN import AttnCycleGAN
        attn_cycle_gan = AttnCycleGAN(training_parameters)
        attn_cycle_gan.test()
        
    elif args["file_format"] == "dicom":
        from AttnCycleGAN_test import AttnCycleGAN
        attn_cycle_gan = AttnCycleGAN(training_parameters)
        
        test_data = DicomData("/data/Chest/Bucheon_CMC_DR/Chest",
                              args["batch_size"],
                              args["image_size"],
                              False)
        attn_cycle_gan.test_dicom(test_data)
    print("[*] Test is finished!")

if __name__ == '__main__':
    main()
    