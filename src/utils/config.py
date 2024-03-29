"""
|
    author: Parisa Daj
    date: May 10, 2022
    parsing and configuration
"""
import pathlib
import argparse

import datetime
import pytz


def check_args(args):
    """
    checking arguments
    """
    # --iteration
    assert args.iteration >= 1, 'number of iterations must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    return args


ITER = 3000


def parse_args():
    """
        Define terminal input arguments
    Returns
    -------
    arguments
    """
    dnn_parser = dnn_pars_args(ITER)
    dir_parser = dir_pars_args()

    parser = argparse.ArgumentParser(parents=[dir_parser, dnn_parser],
                                     conflict_handler='resolve'
                                     )
    parser.add_argument("--sample_interval",
                        type=int,
                        default=1
                        # default=int(1 + 10 * (np.log10(1 + ITER // 50)))
                        )
    parser.add_argument("--validate_interval", type=int, default=1)
    parser.add_argument("--coords", type=float, nargs='+',
                        default=[35.134357, -89.820009, 35.118795, -89.788385, 14],
                        help='top left latitude, longitude, buttom right latitude, longitude, zoom')

    return check_args(parser.parse_args())


def dir_pars_args():
    """
    directories
    :return:
    """
    dir_parser = argparse.ArgumentParser(add_help=False)
    dir_parser.add_argument("--data_dir", type=str,
                            default="/home/sdjkhosh/Datasets/VisnavPNGFiles/jpg Simulated Files/Raster",
                            help='The directory of the data')
    dir_parser.add_argument('--result_dir', type=str, default='results_ablation',
                            help='Directory name to save the generated images')
    dir_parser.add_argument("--model_weights", type=str,
                            default=pathlib.Path(
                                "results/Satellite_vbnnet_24-08-2023_time1647/weights_gen_best.h5"))
    dir_parser.add_argument('--extra_test', type=str,
                            default='/home/sdjkhosh/Datasets/VisnavPNGFiles/DJI_images',
                            help='Address to the folder of images outside the test folder to be tested')
    log_name = datetime.datetime.now(pytz.timezone('US/Central')).strftime("%d-%m-%Y_time%H%M")
    dir_parser.add_argument('--log_name', type=str,
                            default=log_name,
                            help='Desired name for the log file instead of date and time.')
    return dir_parser

def dnn_pars_args(iterations=ITER):
    """
    DNN hyper parameters, related parameters
    :param iterations:
    :return:
    """
    dnn_parser = argparse.ArgumentParser(add_help=False)
    dnn_parser.add_argument("--norm", type=str, default='min_max',
                            help='Image normalization Method.',
                            choices=['max',
                                     'min_max',
                                     'prctile'])
    dnn_parser.add_argument('--dataset', type=str, default='VBN',
                            choices=['FixedCell', 'FairSIM', 'VBN'])
    dnn_parser.add_argument('--task', type=str, default=None,
                            choices=[None, 'super_resolution'],
                            help='What type of task are you trying to solve?')
    dnn_parser.add_argument('--dnn_type', type=str, default='VBNNET',
                            choices=['DNN',
                                     'Simese',
                                     'VBNNET'],
                            help='The type of DNN')

    dnn_parser.add_argument("--load_weights", type=int, default=0,
                            choices=range(2))

    # Generator Setup
    dnn_parser.add_argument("--start_lr", type=float, default=1e-3)
    dnn_parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    dnn_parser.add_argument("--opt", type=str, default="adam")
    dnn_parser.add_argument('--batch_size', type=int, default=32,
                            choices=range(1, 128),
                            help='The size of batch')
    dnn_parser.add_argument('--iteration', type=int,
                            default=iterations, help='The number of epochs to run')
    dnn_parser.add_argument('--batch_iter', type=int,
                            default=5, help='The number of iterations to load from the batch')
    dnn_parser.add_argument('--n_augment', type=int,
                            default=1, help='The number of augmented images for each batch')
    dnn_parser.add_argument("--seed", type=int, default=1357)

    return dnn_parser


def gan_parse_args():
    """
    GAN parameters
    :return:
    """
    gan_parser = argparse.ArgumentParser(add_help=False)
    # Discriminator Setup
    gan_parser.add_argument("--d_start_lr", type=float, default=1e-6)  # 2e-5
    gan_parser.add_argument("--d_lr_decay_factor", type=float, default=0.5)
    gan_parser.add_argument("--train_discriminator_times", type=int, default=1)
    gan_parser.add_argument("--d_opt", type=str, default="adam")
    gan_parser.add_argument("--alpha", type=float, default=0.25)  # gan_loss
    gan_parser.add_argument("--train_generator_times", type=int, default=10)

    return gan_parser


def rcan_parse_args():
    """
    Parameters for the RCAN architecture
    :return:
    """
    rcan_parser = argparse.ArgumentParser(add_help=False)
    rcan_parser.add_argument("--n_ResGroup", type=int, default=2)
    rcan_parser.add_argument("--n_rcab", type=int, default=3)
    rcan_parser.add_argument("--n_channel", type=int, default=64)

    return rcan_parser
