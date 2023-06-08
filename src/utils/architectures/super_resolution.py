"""
    author: SPDKH
"""
import tensorflow as tf
from tensorflow.keras.layers import add, multiply
from tensorflow.keras.layers import Conv3D, UpSampling3D, LeakyReLU, Lambda, ReLU


def srcnn(net_input, scale=2, filters=(9, 5, 5), coeffs=(128, 64)):
    """
    source: https://arxiv.org/pdf/1501.00092.pdf
    important: deeper is not better here
    The best performance is as in the examples but takes more time

    Parameters
    ----------
    filters:
        a list of filters (int)
        example(9, 5, 5)

    coeffs:
        a list of coefficient values (1 item less than filters)
        example(128, 64)

    Returns
    -------
    the output tensorflow layer from the architecture

    todo: start with imagenet weights
    """
    # net_input = x
    # x = layers.Conv2D(64, 9, activation="relu", padding="same")(net_input)
    # x = layers.Conv2D(32, 5, activation="relu", padding="same")(x)
    #
    # x = layers.Conv2DTranspose(1, 5, activation="relu", padding="same")(x)

    conv = net_input
    for i, n_i in enumerate(coeffs):
        conv = Conv3D(n_i,
                      kernel_size=filters[i],
                      padding='same')(conv)
        conv = ReLU()(conv)

    conv = Conv3D(1, kernel_size=filters[-1], padding='same')(conv)
    output = UpSampling3D(size=(scale,
                                scale,
                                1))(conv)
    return output


def ca_layer(net_input, channel, reduction=16):
    """
        Channel Attention Layer used in RCAN super-resolution
    """
    conv = Lambda(global_average_pooling3d)(net_input)
    conv = Conv3D(channel // reduction,
                  kernel_size=1,
                  activation='relu',
                  padding='same')(conv)
    conv = Conv3D(channel,
                  kernel_size=1,
                  activation='sigmoid',
                  padding='same')(conv)
    mul = multiply([net_input, conv])
    return mul


def rcab(net_input, channel):
    """
        Residucal Channel Attention Block used in RCAN super-resolution
    """
    conv = net_input
    for _ in range(2):
        conv = Conv3D(channel, kernel_size=3, padding='same')(conv)
        conv = LeakyReLU(alpha=0.2)(conv)
    att = ca_layer(conv, channel, reduction=16)
    output = add([att, net_input])
    return output


def res_group(net_input, channel, n_rcab):
    """
        residual group used in RCAN super-resolution
    """
    conv = net_input
    for _ in range(n_rcab):
        conv = rcab(conv, channel)
    return conv


def rcan(net_input, scale=2, channel=64, n_res_group=3, n_rcab=5):
    """
        RCAN super-resolution arch
    """
    conv = Conv3D(channel, kernel_size=3, padding='same')(net_input)
    for _ in range(n_res_group):
        conv = res_group(conv, channel=channel, n_rcab=n_rcab)

    up_sample = UpSampling3D(size=(scale, scale, 1))(conv)
    conv = Conv3D(channel, kernel_size=3, padding='same')(up_sample)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(1, kernel_size=3, padding='same')(conv)
    output = LeakyReLU(alpha=0.2)(conv)
    return output


def conv_block(net_input, channel_size):
    """

    Parameters
    ----------
    net_input
    channel_size

    Returns
    -------
    """
    conv = net_input
    for _ in range(2):
        conv = Conv3D(channel_size[0],
                      kernel_size=3,
                      padding='same')(net_input)
        conv = LeakyReLU(alpha=0.1)(conv)
    return conv


def global_average_pooling3d(layer_in):
    """
        Not sure how different this is than
        built-in global average pooling
    """
    return tf.reduce_mean(layer_in, axis=(1, 2, 3), keepdims=True)
