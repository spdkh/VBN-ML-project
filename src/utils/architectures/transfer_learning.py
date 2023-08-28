"""
    author: SPDKH
    date: June 2023
"""
import numpy as np

from keras import models
from keras.applications import VGG16

from src.utils.architectures import basic_arch


def vgg16(net_input, n_classes):
    """
        VGG16 transfer learning
    :param net_input:
    :param n_classes:
    :return:
    """
    image_size = np.shape(net_input)[1:]

    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=image_size)

    # Freeze the layers except the last 4 layers
    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    # Check the trainable status of the individual layers
    for layer in vgg_conv.layers:
        print(layer, layer.trainable)

    extraction_model = models.Model(
        inputs=[vgg_conv.input],
        outputs=[vgg_conv.output])
    # Create the model
    output = basic_arch.simple_cnn(extraction_model(net_input), n_classes, [(4, 2)])

    print('after passing vgg output to basic arch:', output)

    return output
