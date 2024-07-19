import tensorflow as tf
from tensorflow import keras
from vgg16 import VGG16Loss
import pdb

BCE = keras.losses.BinaryCrossentropy(from_logits=True)

WEIGHT_GD = 100
WEIGHT_CYCLE = 10
WEIGHT_L1_LAMBDA = 10
WEIGHT_SSIM = 1

def gradient_difference_loss_fn(y, y_pred):
    gdl_loss = tf.constant(0., dtype=tf.float32)
    preds_dy, preds_dx = tf.image.image_gradients(y_pred)
    gts_dy, gts_dx = tf.image.image_gradients(y)

    gdl_loss = tf.reduce_mean(tf.abs(tf.abs(preds_dy) - tf.abs(gts_dy)) +
                                                    tf.abs(tf.abs(preds_dx) - tf.abs(gts_dx)))
    return gdl_loss


def ssim_loss_fn(y, y_pred):
    ssim_loss = tf.constant(0., dtype=tf.float32)
    preds = (y_pred + 1.) / 2.
    gts = (y + 1.) / 2.
    ssim_positive = tf.math.maximum(
        0., tf.image.ssim(preds, gts, max_val=1.0))
    ssim_loss = -tf.math.log(ssim_positive)
    ssim_loss = tf.reduce_mean(ssim_loss)
    # ssim_loss = tf.expand_dims(ssim_loss, -1)
    return ssim_loss
    

def cycle_loss_fn(origin_img, rec_img):
    cycle_loss = tf.reduce_mean(tf.abs(origin_img - rec_img))
    return cycle_loss


def voxel_loss_fn(y, y_pred):
    voxel_loss = tf.reduce_mean(tf.abs(y - y_pred))
    return voxel_loss


def g_loss_fn(y_preds_logits, use_lsgan=False):
    # if use_lsgan:
    #     g_loss = MSE(tf.ones_like(y_preds_logits))
    # else:
    g_loss = BCE(tf.ones_like(y_preds_logits), y_preds_logits)
    g_loss = tf.reduce_mean(g_loss)
    return g_loss


def content_loss_fn(y, y_pred):
    content_loss = VGG16Loss()(y, y_pred)
    return content_loss

def d_loss_fn(y_logits, y_preds_logits, use_lsgan=False):
    # if use_lsgan:
    #     real_loss = MSE(tf.ones_like(y_logits), y_logits)
    #     fake_loss = MSE(tf.zeros_like(y_preds_logits), y_preds_logits)
    # else:
    real_loss = BCE(tf.ones_like(y_logits), y_logits)
    fake_loss = BCE(tf.zeros_like(y_preds_logits), y_preds_logits)
    
    d_loss = (real_loss + fake_loss)/2
    d_loss = tf.reduce_mean(d_loss)    
    return d_loss