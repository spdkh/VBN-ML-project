# -*- coding: utf-8 -*-
"""
    VBN-NET
    author: SPDKH
    date: 2023
"""
import sys
from abc import abstractmethod
import datetime
import glob

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from src.models.dnn import DNN
from src.utils.img_helper import img_comp
from src.utils import const, data_helper
from src.utils.architectures.regression import simple_dense


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
        self.model_output = simple_dense(self.model_input, 3)
        print(self.model_output)
        print(self.model_input)
        self.model = Model(self.model_input,
                           self.model_output)

    def build_model(self):
        self.model.compile(loss='mean_absolute_error',
                           optimizer='adam',
                           metrics=['mean_absolute_error'])
        print(self.model.summary())

    def train_epoch(self, batch_log: bool = 0):
        """
            Training process per epoch
            loop over all data samples / number of batches
            train per batch to complete an epoch
            todo: update the loop
        """
        start_time = datetime.datetime.now()
        batch_id = -1
        iteration = 0
        while batch_id != 0:
            iteration += 1
            batch_id = self.batch_iterator('train')
            print(iteration, batch_id)

            batch_imgs = \
                data_helper.img_batch_load(self.data.data_info['xtrain'],
                                           self.args.batch_size,
                                           batch_id)
            batch_output \
                = np.squeeze(self.data.data_info['ytrain'][batch_id:batch_id
                                                         + self.args.batch_size])

            print(np.shape(self.model_input), np.shape(self.model_output))
            print(np.shape(batch_imgs), np.shape(batch_output))
            batch_loss = self.model.train_on_batch(batch_imgs, batch_output)

            elapsed_time = datetime.datetime.now() - start_time

            if batch_log:
                tf.print("%d batch iteration: time: %s, batch_loss = %s" % (
                    batch_id + 1,
                    elapsed_time,
                    batch_loss), output_stream=sys.stdout)

            if (iteration) % self.args.sample_interval == 0:
                self.validate(iteration, sample=1)

            self.loss_record.append(batch_loss)

        return np.mean(self.loss_record)

    def train(self):
        """
                iterate over epochs
            """
        start_time = datetime.datetime.now()

        print('Training...')

        for iteration in range(self.args.iteration):
            elapsed_time = datetime.datetime.now() - start_time

            model_loss = self.train_epoch(0)

            tf.print("%d epoch: time: %s, loss = %s" % (
                iteration + 1,
                elapsed_time,
                model_loss), output_stream=sys.stdout)

            if iteration % self.args.validate_interval == 0:
                self.validate(iteration, sample=0)
                self.write_log(self.writer,
                               'simple_dense',
                               np.mean(self.loss_record),
                               iteration)
                self.loss_record = []

            self.loss_record.append(model_loss)

    def validate(self, iteration, sample=0):
        """
                :param iteration: current iteration number
                :param sample: sample id
                :return:
                todo: review
        """
        validate_nrmse = [np.Inf]

        val_names = ['val_MSE',
                     'val_SSIM',
                     'val_PSNR',
                     'val_NRMSE',
                     'val_UQI']

        validate_path = glob.glob(self.data.data_info['val'] + '*')
        validate_path.sort()

        metrics = {'mse': [],
                   'nrmse': [],
                   'psnr': [],
                   'ssim': [],
                   'uqi': []}

        imgs, imgs_gt = \
            self.data.image2image_batch_load(self.args.batch_size,
                                             self.batch_iterator('val'),
                                             self.scale_factor,
                                             'val')

        outputs = self.model.predict(imgs)
        for output, img_gt in zip(outputs, imgs_gt):
            # predict generates [1, x, y, z, 1]
            # It is converted to [x, y, z] below
            output = np.reshape(output,
                                self.data.output_dim[:-1])

            output_proj = np.max(output, 2)

            gt_proj = np.max(np.reshape(img_gt,
                                        self.data.output_dim[:-1]),
                             2)
            metrics.values = \
                img_comp(gt_proj,
                         output_proj,
                         metrics['mse'],
                         metrics['nrmse'],
                         metrics['psnr'],
                         metrics['ssim'],
                         metrics['uqi'])

        if sample == 0:
            self.model.save_weights(const.WEIGHTS_DIR
                                    / 'weights_gen_latest.h5')

            if min(validate_nrmse) > np.mean(metrics['nrmse']):
                self.model.save_weights(const.WEIGHTS_DIR
                                        / 'weights_gen_best.h5')

            validate_nrmse.append(np.mean(metrics['nrmse']))
            cur_lr = self.lr_controller.on_iteration_end(iteration, np.mean(metrics['nrmse']))
            self.write_log(self.writer, 'lr_sr', cur_lr, iteration)
            cur_lr_d = self.lr_controller.on_iteration_end(iteration, np.mean(metrics['nrmse']))
            self.write_log(self.writer, 'lr_d', cur_lr_d, iteration)
            self.write_log(self.writer, val_names[0], np.mean(metrics['mse']), iteration)
            self.write_log(self.writer, val_names[1], np.mean(metrics['ssim']), iteration)
            self.write_log(self.writer, val_names[2], np.mean(metrics['psnr']), iteration)
            self.write_log(self.writer, val_names[3], np.mean(metrics['nrmse']), iteration)
            self.write_log(self.writer, val_names[4], np.mean(metrics['uqi']), iteration)
        else:
            plt.figure(figsize=(22, 6))
            validation_id = 0
            # figures equal to the number of z patches in columns
            for j in range(self.data.input_dim[2]):
                output_results = {
                    'WF Raw Input': self.data.calc_wf(imgs)[validation_id, :, :, j, 0],
                    'SR Output': self.data.norm(outputs[validation_id, :, :, j, 0]),
                    'Ground Truth': imgs_gt[validation_id, :, :, j, 0]
                }

                plt.title('Z = ' + str(j))
                for i, (label, img) in enumerate(output_results.items()):
                    # first row: input image average of angles and phases
                    # second row: resulting output
                    # third row: ground truth
                    plt.subplot(3,
                                self.data.input_dim[2],
                                j + self.data.input_dim[2] * i + 1)
                    plt.ylabel(label)
                    plt.imshow(img, cmap=plt.get_cmap('hot'))

                    plt.gca().axes.yaxis.set_ticklabels([])
                    plt.gca().axes.xaxis.set_ticklabels([])
                    plt.gca().axes.yaxis.set_ticks([])
                    plt.gca().axes.xaxis.set_ticks([])
                    plt.colorbar()

            plt.savefig(const.SAMPLE_DIR + '%d.png' % iteration)  # Save sample results
            plt.close("all")  # Close figures to avoid memory leak
