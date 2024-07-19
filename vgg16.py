import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16

class VGG16Loss(object):
    def __init__(self, input_shape=(512, 512, 3)):
        self.input_shape = input_shape
        self.vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
        self.vgg16.trainable = False
        
        for l in self.vgg16.layers:
            l.trainable = False
        
        self.model_1 = keras.Model(inputs=self.vgg16.input, outputs=self.vgg16.layers[2].output)
        self.model_1.trainable = False
        self.model_2 = keras.Model(inputs=self.vgg16.input, outputs=self.vgg16.layers[5].output)
        self.model_2.trainable = False
        self.model_3 = keras.Model(inputs=self.vgg16.input, outputs=self.vgg16.layers[9].output)
        self.model_3.trainable = False
        self.model_4 = keras.Model(inputs=self.vgg16.input, outputs=self.vgg16.layers[13].output)
        self.model_4.trainable = False
        self.model_5 = keras.Model(inputs=self.vgg16.input, outputs=self.vgg16.layers[17].output)
        self.model_5.trainable = False
        
        self. models = [self.model_1, self.model_2, self.model_3, self.model_4, self.model_5]
        
    def __call__(self, y, y_pred):
        content_loss = tf.constant(0., dtype=tf.float32)
        y_concat = tf.concat([y, y, y], axis=3)
        y_pred_concat = tf.concat([y_pred, y_pred, y_pred], axis=3)
        
        preds_features = []
        gts_features = []
        
        for model in self.models:
            preds_features.append(model(y_pred_concat))
            gts_features.append(model(y_concat))
        
        for preds_feature, gts_feature in zip(preds_features, gts_features):
            content_loss += tf.reduce_mean(tf.abs(preds_feature - gts_feature))
            
        content_loss = content_loss / len(preds_features)
        
        return content_loss