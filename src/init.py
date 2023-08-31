"""
    author: spdkh
    date: Aug 2023
"""
import sys

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from src.models import *

from src.utils.config import parse_args


# pylint: disable=W0401:
def run():
    """
    Import automated modules based on arguments and build the DNN
    for both train and prediction tasks
    :return:
    """
    # parse arguments
    args = parse_args()

    if args is None:
        sys.exit()

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    _ = InteractiveSession(config=config)

    # # open session
    if args.task is not None:
        module_name = '.'.join(['src.models',
                                args.task,
                                args.dnn_type.lower()])
    else:
        module_name = '.'.join(['src.models',
                                args.dnn_type.lower()])
    print(module_name)
    dnn_module = __import__(module_name,
                            fromlist=[args.dnn_type])
    dnn = getattr(dnn_module,
                  args.dnn_type)(args)


    dnn.build_model()

    if args.model_weights:
        dnn.model.load_weights(args.model_weights, by_name=True, skip_mismatch=True)

    return dnn
