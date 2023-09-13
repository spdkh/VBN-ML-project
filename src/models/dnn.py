"""
    author: SPDKH
    todo: complete
"""
from __future__ import division
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input

from src.utils import const


class DNN(ABC):
    """
        Abstract class for DNN architectures
    """

    def __init__(self, args):
        """
            args: argparse object
        """
        self.batch_id = {'train': 0, 'val': 0, 'test': 0}

        self.args = args
        # text_color = "\033[0m"
        # print(text_color)
        print('\n[DNN] Initiated Parameters:\n', self.args)
        print('\n[DNN] Init DNN Arch:', self.args.dnn_type)

        module_name = '.'.join(['src.data',
                                args.dataset.lower()])
        dataset_module = __import__(module_name,
                                    fromlist=[args.dataset])
        print('[DNN] Dataset:', module_name)
        self.data = getattr(dataset_module,
                            args.dataset)(self.args)
        self.data.config()

        self.model_input = Input(self.data.input_dim)
        self.model_output = None
        self.model = None
        self.lr_controller = None
        self.loss_record = []

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.n_batches = {'train': np.shape(self.data.data_info['ytrain'])[0]
                                   // self.args.batch_size,
                          'val': np.shape(self.data.data_info['yval'])[0]
                                 // self.args.batch_size,
                          'test': np.shape(self.data.data_info['ytest'])[0]
                                  // self.args.batch_size}
        print(self.n_batches)
        self.writer = tf.summary.create_file_writer(str(const.LOG_DIR))
        print('Writing Logs to:', const.LOG_DIR)

        super().__init__()

    def batch_iterator(self, mode='train'):
        """
            takes care of loading batches iteratively
        """
        # how many total data in that mode exists
        # data_size = len(self.data.data_info['x' + mode])
        if self.n_batches[mode] - 1 <= self.batch_id[mode]:
            self.batch_id[mode] = 0
        else:
            self.batch_id[mode] += 1

        return self.batch_id[mode]

    @abstractmethod
    def train(self):
        """
            iterate over epochs
        """

    @abstractmethod
    def predict(self):
        """

        :return:
        """

    @abstractmethod
    def train_epoch(self, iteration, batch_log):
        """
            Training process per epoch
            loop over all data samples / number of batches
            train per batch to complete an epoch
        :param iteration: int
        :param batch_log: bool
        :return:
        """

    @abstractmethod
    def build_model(self):
        """
            function to define the model,
            loss functions,
            optimizers,
            learning rate controller,
            compile model,
            load initial weights if available
        """

    @abstractmethod
    def validate(self, sample, sample_id, mode):
        """
            validate and write logs
        """

    def write_log(self, names, logs, batch_no=0, mode='float'):
        """
        todo: test
        Parameters
        ----------
        names
        logs
        batch_no
        mode
        """
        writer = self.writer
        with writer.as_default():
            if mode == 'float':
                tf.summary.scalar(names, logs, step=batch_no)
            elif mode == 'image':
                tf.summary.image(names, [logs], step=batch_no)
            else:
                tf.summary.text(names,
                                tf.convert_to_tensor(str(logs),
                                                     dtype=tf.string),
                                step=batch_no)
            writer.flush()
