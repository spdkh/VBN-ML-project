import argparse
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Flatten, Input, add, multiply
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import imageio
import os
from src.models import *
from src.utils.norm_helper import prctile_norm

from src.utils.config import parse_args

"""
    author: SPDKH
    date: Nov 2, 2023
"""


def main():
    """
        Main Predicting Function
    """
    # parse arguments
    args = parse_args()

    if args is None:
        sys.exit()

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    _ = InteractiveSession(config=config)

    # dnn = tf.keras.models.load_model(self.args.model_weights)


    # # open session
    if args.task is not None:
        module_name = '.'.join(['src.models',
                                args.task,
                                args.dnn_type])
    else:
        module_name = '.'.join(['src.models',
                                args.dnn_type])
    print(module_name)
    dnn_module = __import__(module_name,
                            fromlist=[args.dnn_type.upper()])
    dnn = getattr(dnn_module,
                  args.dnn_type.upper())(args)

    dnn.build_model()

    dnn.model.load_weights(args.model_weights, by_name=True, skip_mismatch=True)

    dnn.predict()
    print("\n [*] Testing finished!")


if __name__ == '__main__':
    main()
