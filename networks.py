import pdb
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_addons as tfa

first_conv_channels = 64
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 2

# Kernel initializer to use
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )

kernel_truncated_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
kernel_normal_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)


"""
Attention CycleGAN
""" 
class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj
    
def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish):
    def apply(inputs):
        x = inputs
        input_width = x.shape[3]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0)
            )(x)

        x = tfa.layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        
        x = tfa.layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0)
        )(x)
        x = layers.Add()([x, residual])
        return x

    return apply

def DownSample(width):
    def apply(x):
        x = layers.Conv2D(
            width,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=kernel_init(1.0),
        )(x)
        return x

    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        return x

    return apply

def build_attn_generator(input_shape, norm_groups=8, init_filter_num=64, activation_fn=keras.activations.swish):
    input_layer = layers.Input(input_shape)

    """Init convolution"""
    # (N, H, W, C) -> (N, H, W, 64)
    # 512, 512, 1 -> 512, 512, 64
    x = tf.pad(input_layer, paddings=[[0,0], [3,3], [3,3], [0,0]], mode="REFLECT", name="conv1_padding")
    x = layers.Conv2D(init_filter_num, 7, strides=(1, 1), padding="VALID",
                            kernel_initializer=kernel_truncated_init, use_bias=True, name="conv1_conv")(x)

    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)(x)
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)(x)
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)(x)

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )(x)
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i])(x)

    # End block
    x = tfa.layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.Conv2D(1, (3, 3), padding="same", kernel_initializer=kernel_init(0.0))(x)
    x = layers.Activation('tanh', name="output_tanh")(x)
    
    return keras.Model([input_layer], x, name="Attention_Generator")

def build_attn_discriminator(input_shape, norm_groups=8, init_filter_num=64, activation_fn=keras.activations.swish):
    # input_layer = layers.Input(input_shape)

    # # (N, H, W, C) -> (N, H/2, W/2, 64)
    # conv1 = layers.Conv2D(init_filter_num, 4, strides=(2, 2), padding="SAME",
    #                       kernel_initializer=kernel_truncated_init, use_bias=True, name="conv1_conv")(input_layer)
    # conv1 = layers.LeakyReLU(0.2, name="conv1_lrelu")(conv1)

    # # (N, H/2, W/2, 64) -> (N, H/4, W/4, 128)
    # conv2 = layers.Conv2D(init_filter_num*2, 4, strides=(2, 2), padding="SAME",
    #                       kernel_initializer=kernel_truncated_init, use_bias=True, name="conv2_conv")(conv1)
    # conv2 = tfa.layers.InstanceNormalization(name="conv2_norm")(conv2)
    # conv2 = layers.LeakyReLU(0.2, name="conv2_lrelu")(conv2)

    # # (N, H/4, W/4, 128) -> (N, H/8, W/8, 256)
    # conv3 = layers.Conv2D(init_filter_num*4, 4, strides=(2, 2), padding="SAME",
    #                       kernel_initializer=kernel_truncated_init, use_bias=True, name="conv3_conv")(conv2)
    # conv3 = tfa.layers.InstanceNormalization(name="conv3_norm")(conv3)
    # conv3 = layers.LeakyReLU(0.2, name="conv3_lrelu")(conv3)

    # # (N, H/8, W/8, 256) -> (N, H/16, W/16, 512)
    # conv4 = layers.Conv2D(init_filter_num*8, 4, strides=(2, 2), padding="SAME",
    #                       kernel_initializer=kernel_truncated_init, use_bias=True, name="conv4_conv")(conv3)
    # conv4 = tfa.layers.InstanceNormalization(name="conv4_norm")(conv4)
    # conv4 = layers.LeakyReLU(0.2, name="conv4_lrelu")(conv4)

    # # (N, H/16, W/16, 512) -> (N, H/16, W/16, 1)
    # conv5 = layers.Conv2D(1, 4, strides=(1, 1), padding="SAME",
    #                       kernel_initializer=kernel_truncated_init, use_bias=True, name="conv5_conv")(conv4)

    # output = tf.identity(conv5, name="output_without_sigmoid")

    # return keras.Model(inputs=input_layer, outputs=output, name="Cycle_Discriminator")


    input_layer = layers.Input(input_shape)

    # (N, H, W, C) -> (N, H/2, W/2, 64)
    conv1 = layers.Conv2D(init_filter_num, 4, strides=(2, 2), padding="SAME",
                          kernel_initializer=kernel_truncated_init, use_bias=True, name="conv1_conv")(input_layer)
    conv1 = layers.LeakyReLU(0.2, name="conv1_lrelu")(conv1)

    # (N, H/2, W/2, 64) -> (N, H/4, W/4, 128)
    conv2 = layers.Conv2D(init_filter_num*2, 4, strides=(2, 2), padding="SAME",
                          kernel_initializer=kernel_truncated_init, use_bias=True, name="conv2_conv")(conv1)
    conv2 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init, name="conv2_norm")(conv2)
    conv2 = layers.LeakyReLU(0.2, name="conv2_lrelu")(conv2)
    
    # (N, H/4, W/4, 128) -> (N, H/8, W/8, 256)
    conv3 = AttentionBlock(init_filter_num*2, groups=norm_groups)(conv2)
    conv3 = layers.Conv2D(init_filter_num*4, 4, strides=(2, 2), padding="SAME",
                          kernel_initializer=kernel_truncated_init, use_bias=True, name="conv3_conv")(conv3)
    conv3 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init, name="conv3_norm")(conv3)
    conv3 = layers.LeakyReLU(0.2, name="conv3_lrelu")(conv3)


    # (N, H/8, W/8, 256) -> (N, H/16, W/16, 512)
    conv4 = AttentionBlock(init_filter_num*4, groups=norm_groups)(conv3)
    conv4 = layers.Conv2D(init_filter_num*8, 4, strides=(2, 2), padding="SAME",
                          kernel_initializer=kernel_truncated_init, use_bias=True, name="conv4_conv")(conv3)
    conv4 = tfa.layers.InstanceNormalization(name="conv4_norm")(conv4)
    conv4 = layers.LeakyReLU(0.2, name="conv4_lrelu")(conv4)

    # (N, H/16, W/16, 512) -> (N, H/16, W/16, 1)
    conv5 = layers.Conv2D(1, 4, strides=(1, 1), padding="SAME",
                          kernel_initializer=kernel_truncated_init, use_bias=True, name="conv5_conv")(conv4)

    output = tf.identity(conv5, name="output_without_sigmoid")
    
    return keras.Model(inputs=input_layer, outputs=output, name="Attention_Discriminator")


def build_basic_generator(input_shape, init_filter_num=64):
    input_layer = layers.Input(input_shape)

    # (N, H, W, C) -> (N, H, W, 64)
    conv1 = tf.pad(input_layer, paddings=[[0,0], [3,3], [3,3], [0,0]], mode="REFLECT", name="conv1_padding")
    conv1 = layers.Conv2D(init_filter_num, 7, strides=(1, 1), padding="VALID",
                          kernel_initializer=kernel_truncated_init, use_bias=True, name="conv1_conv")(conv1)
    conv1 = tfa.layers.InstanceNormalization(name="conv1_norm")(conv1)
    conv1 = layers.ReLU(name="conv1_relu")(conv1)

    # (N, H, W, 64)  -> (N, H/2, W/2, 128)
    conv2 = layers.Conv2D(init_filter_num*2, 3, strides=(2, 2), padding="SAME",
                          kernel_initializer=kernel_truncated_init, use_bias=True, name="conv2_conv")(conv1)
    conv2 = tfa.layers.InstanceNormalization(name="conv2_norm")(conv2)
    conv2 = layers.ReLU(name="conv2_relu")(conv2)

    # (N, H/2, W/2, 128) -> (N, H/4, W/4, 256)
    conv3 = layers.Conv2D(init_filter_num*4, 3, strides=(2, 2), padding="SAME",
                          kernel_initializer=kernel_truncated_init, use_bias=True, name="conv3_conv")(conv2)
    conv3 = tfa.layers.InstanceNormalization(name="conv3_norm")(conv3)
    conv3 = layers.ReLU(name="conv3_relu")(conv3)

    # (N, H/4, W/4, 256) -> (N, H/4, W/4, 256)
    if (input_shape[0] <= 128) and (input_shape[1] <= 128):
        res_out = n_res_blocks(conv3, num_blocks=6)
    else:
        res_out = n_res_blocks(conv3, num_blocks=9)

    # (N, H/4, W/4, 256) -> (N, H/2, W/2, 128)
    conv4 = layers.Conv2DTranspose(init_filter_num*2, 3, strides=(2, 2), padding="SAME", name="conv4_deconv")(res_out)
    conv4 = tfa.layers.InstanceNormalization(name="conv4_norm")(conv4)
    conv4 = layers.ReLU(name="conv4_relu")(conv4)

    # (N, H/2, W/2, 128) -> (N, H, W, 64)
    conv5 = layers.Conv2DTranspose(init_filter_num, 3, strides=(2, 2), padding="SAME", name="conv5_deconv")(conv4)
    conv5 = tfa.layers.InstanceNormalization(name="conv5_norm")(conv5)
    conv5 = layers.ReLU(name="conv5_relu")(conv5)

    # (N, H, W, 64) -> (N, H, W, 3)
    conv6 = tf.pad(conv5, paddings=[[0,0], [3,3], [3,3], [0,0]], mode="REFLECT", name="output_padding")
    conv6 = layers.Conv2D(1, 7, strides=(1, 1), padding="VALID",
                          kernel_initializer=kernel_truncated_init, use_bias=True, name="output_conv")(conv6)
    output = layers.Activation('tanh', name="output_tanh")(conv6)

    return keras.Model(inputs=input_layer, outputs=output, name="Cycle_Generator")


def n_res_blocks(x, num_blocks=6):
    output = None
    for idx in range(1, num_blocks+1):
        output = res_block(x, x.shape[3], idx)

        x = output

    return output


def res_block(x, k, idx):
    conv1 = layers.Conv2D(k, 3, strides=(1, 1), padding="SAME",
                          kernel_initializer=kernel_normal_init, use_bias=True, name=f'res_block{idx}_conv1')(x)
    normalized1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init,
                                                   name=f"res_block{idx}_norm1")(conv1)
    relu1 = layers.ReLU()(normalized1)

    conv2 = layers.Conv2D(k, 3, strides=(1, 1), padding="SAME",
                          kernel_initializer=kernel_normal_init, use_bias=True, name=f'res_block{idx}_conv2')(relu1)
    normalized2 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init,
                                                   name=f"res_block{idx}_norm2")(conv2)
    output = x + normalized2
    return output


"""
CycleGAN
"""

def build_basic_discriminator(input_shape, init_filter_num=64):
    input_layer = layers.Input(input_shape)

    # (N, H, W, C) -> (N, H/2, W/2, 64)
    conv1 = layers.Conv2D(init_filter_num, 4, strides=(2, 2), padding="SAME",
                          kernel_initializer=kernel_truncated_init, use_bias=True, name="conv1_conv")(input_layer)
    conv1 = layers.LeakyReLU(0.2, name="conv1_lrelu")(conv1)

    # (N, H/2, W/2, 64) -> (N, H/4, W/4, 128)
    conv2 = layers.Conv2D(init_filter_num*2, 4, strides=(2, 2), padding="SAME",
                          kernel_initializer=kernel_truncated_init, use_bias=True, name="conv2_conv")(conv1)
    conv2 = tfa.layers.InstanceNormalization(name="conv2_norm")(conv2)
    conv2 = layers.LeakyReLU(0.2, name="conv2_lrelu")(conv2)

    # (N, H/4, W/4, 128) -> (N, H/8, W/8, 256)
    conv3 = layers.Conv2D(init_filter_num*4, 4, strides=(2, 2), padding="SAME",
                          kernel_initializer=kernel_truncated_init, use_bias=True, name="conv3_conv")(conv2)
    conv3 = tfa.layers.InstanceNormalization(name="conv3_norm")(conv3)
    conv3 = layers.LeakyReLU(0.2, name="conv3_lrelu")(conv3)

    # (N, H/8, W/8, 256) -> (N, H/16, W/16, 512)
    conv4 = layers.Conv2D(init_filter_num*8, 4, strides=(2, 2), padding="SAME",
                          kernel_initializer=kernel_truncated_init, use_bias=True, name="conv4_conv")(conv3)
    conv4 = tfa.layers.InstanceNormalization(name="conv4_norm")(conv4)
    conv4 = layers.LeakyReLU(0.2, name="conv4_lrelu")(conv4)

    # (N, H/16, W/16, 512) -> (N, H/16, W/16, 1)
    conv5 = layers.Conv2D(1, 4, strides=(1, 1), padding="SAME",
                          kernel_initializer=kernel_truncated_init, use_bias=True, name="conv5_conv")(conv4)

    output = tf.identity(conv5, name="output_without_sigmoid")

    return keras.Model(inputs=input_layer, outputs=output, name="Cycle_Discriminator")

"""
select networks
"""

def build_generator(input_shape, networks):
    if networks == "attention":
        model = build_attn_generator(input_shape)
    elif networks == "basic":
        model = build_basic_generator(input_shape)
    
    return model
    
def build_discriminator(input_shape, networks):
    if networks == "attention":
        model = build_attn_discriminator(input_shape)
    elif networks == "basic":
        model = build_basic_discriminator(input_shape)
        
    return model