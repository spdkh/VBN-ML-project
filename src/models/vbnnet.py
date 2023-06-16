# -*- coding: utf-8 -*-
"""
    VBN-NET
    author: SPDKH
    date: 2023
"""
import sys
import datetime

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model

from src.models.dnn import DNN
from src.utils.img_helper import img_comp
from src.utils import const, data_helper, norm_helper
from src.utils.architectures.regression import simple_dense
from src.utils.architectures.basic_arch import simple_cnn


class VBNNET(DNN):
    """
        Visual Based Navigation
        DLL implementation
    """

    def __init__(self, args):
        """
            params:
                args: argparse object
        """
        DNN.__init__(self, args)
        self.model_output = simple_cnn(self.model_input, 3)
        self.model = Model(self.model_input,
                           self.model_output)

    def build_model(self):
        self.model.compile(loss='mean_absolute_error',
                           optimizer='adam',
                           metrics=['mean_absolute_error'])
        print(self.model.summary())

    def train_epoch(self, batch_log: bool = False):
        """
            Training process per epoch
            loop over all data samples / number of batches
            train per batch to complete an epoch
            todo: update the loop
        """
        start_time = datetime.datetime.now()
        self.batch_id['train'] = 1
        batch_id = 1
        loss_record = []
        while batch_id != 0:
            batch_imgs = \
                data_helper.img_batch_load(self.data.data_info['xtrain'],
                                           self.args.batch_size,
                                           batch_id)
            batch_output \
                = self.data.data_info['ytrain'][2*batch_id:
                                                2*batch_id + self.args.batch_size]

            batch_loss = self.model.train_on_batch(batch_imgs, batch_output)

            elapsed_time = datetime.datetime.now() - start_time

            if batch_log:
                print("%d batch iteration: time: %s, batch_loss = %s" % (
                        batch_id,
                        elapsed_time,
                        batch_loss))

            loss_record.append(batch_loss)
            batch_id = self.batch_iterator('train')

        return np.mean(loss_record)

    def train(self):
        """
            iterate over epochs
        """
        start_time = datetime.datetime.now()

        print('Training...')
        self.loss_record = []
        for iteration in range(self.args.iteration):
            elapsed_time = datetime.datetime.now() - start_time

            model_loss = self.train_epoch()
            self.loss_record.append(model_loss)
            print("%d epoch: time: %s, loss = %s" % (
                iteration + 1,
                elapsed_time,
                model_loss))

            if iteration % self.args.sample_interval == 0:
                self.validate(sample=1, sample_id=iteration)

            if iteration % self.args.validate_interval == 0:
                self.validate(sample=0)

                self.write_log(self.writer,
                               'simple_dense',
                               np.mean(self.loss_record),
                               iteration)
                self.loss_record = []

    def validate(self, sample=0, sample_id=None):
        """
                :param iteration: current iteration number
                :param sample: sample id
                :return:
                todo: review
        """
        batch_id = self.batch_iterator('val')
        err = [np.Inf]

        val_names = {'val_MAE': []}

        imgs = \
            data_helper.img_batch_load(self.data.data_info['xval'],
                                       self.args.batch_size,
                                       batch_id)
        batch_output \
            = self.data.data_info['yval'][batch_id*2:
                                          2*batch_id + self.args.batch_size]

        outputs = self.model.predict(imgs)
        # print(batch_id, 'Validating...')
        for i, (img, output) in enumerate(zip(imgs, outputs)):
            output = norm_helper.min_max_norm(output)
            # print('predicted:', output)
            img_gt = np.asarray(batch_output.iloc[i, :])
            # print('gt:', img_gt)
            val_names['val_MAE'].append(np.mean(np.abs(output - img_gt)))

        if sample == 0:
            self.model.save_weights(const.WEIGHTS_DIR
                                    / 'weights_gen_latest.h5')

            if min(err) > np.mean(val_names['val_MAE']):
                self.model.save_weights(const.WEIGHTS_DIR
                                        / 'weights_gen_best.h5')

            err.append(np.mean(val_names['val_MAE']))

            self.write_log(self.writer, 'val_MAE', np.mean(val_names['val_MAE']), batch_id)

        else:
            plt.figure()
            # figures equal to the number of z patches in columns

            plt.title('original lat/long/alt = ' + str(img_gt) + '\nPredicted lat/long/alt =' + str(output))

            plt.imshow(img)

            plt.gca().axes.yaxis.set_ticklabels([])
            plt.gca().axes.xaxis.set_ticklabels([])
            plt.gca().axes.yaxis.set_ticks([])
            plt.gca().axes.xaxis.set_ticks([])

            result_name = str(sample_id) + '_batch' + str(batch_id) + '.png'
            plt.savefig(const.SAMPLE_DIR / result_name)  # Save sample results
            plt.close("all")  # Close figures to avoid memory leak
