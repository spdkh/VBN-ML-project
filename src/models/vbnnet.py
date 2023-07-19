# -*- coding: utf-8 -*-
"""
    VBN-NET
    author: SPDKH
    date: 2023
"""
import sys
import datetime

import geopy
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import visualkeras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

from src.models.dnn import DNN
from src.utils.img_helper import img_comp
from src.utils import const, data_helper, norm_helper
from src.utils.architectures.regression import simple_dense
from src.utils.architectures.transfer_learning import vgg16
from src.utils.config import dnn_pars_args, dir_pars_args


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
        self.model_output = vgg16(self.model_input, 3)
        print('model input:', self.model_input)
        print('model output:', self.model_output)
        self.model = Model(self.model_input,
                           self.model_output)

    def build_model(self):
        self.model.compile(loss='mean_absolute_error',
                           optimizer='adam',
                           metrics=['mean_absolute_error'])
        print(self.model.summary())
        # tf.keras.utils.plot_model(self.model, to_file='Model.png', show_shapes=True, dpi=64, rankdir='LR')
        visualkeras.layered_view(self.model, draw_volume=False, legend=True, to_file='Model2.png')  # write to disk

    def train_epoch(self, iteration, batch_log: bool = False):
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
        with tqdm(total=self.args.batch_iter) as pbar:
            # while batch_id != 0:
            for k in range(self.args.batch_iter):
                batch_imgs = \
                    data_helper.img_batch_load(self.data.data_info['xtrain'],
                                               self.args.batch_size,
                                               batch_id)
                batch_imgs = np.asarray(list(batch_imgs.values()))

                batch_outputs \
                    = self.data.data_info['ytrain'][2 * batch_id:
                                                    2 * batch_id + self.args.batch_size]

                train_datagen = ImageDataGenerator(
                    rescale=1. / 255,
                    rotation_range=360,
                    width_shift_range=0.6,
                    height_shift_range=0.6,
                    horizontal_flip=True,
                    fill_mode='nearest')

                batch_loss = self.model.train_on_batch(batch_imgs, batch_outputs)
                loss_record.append(batch_loss)

                # print('Train with augmented data:')
                for i, (img, batch_output) in enumerate(train_datagen.flow(
                        batch_imgs,
                        y=batch_outputs,
                        batch_size=self.args.batch_size,
                        shuffle=True,
                        seed=self.args.seed,
                        ignore_class_split=True,
                )):
                    # output = self.model.predict(img)
                    # print(i, img.shape, batch_output.shape)
                    # output = norm_helper.min_max_norm(output)
                    batch_loss = self.model.train_on_batch(img, batch_output)
                    loss_record.append(batch_loss)

                    if i > self.args.n_augment:
                        break

                elapsed_time = datetime.datetime.now() - start_time
                batch_loss = np.mean(loss_record)
                if batch_log:
                    print("%d batch iteration: time: %s, batch_loss = %s" % (
                        batch_id,
                        elapsed_time,
                        batch_loss))

                self.write_log(
                    'full train loss',
                    batch_loss,
                    iteration * self.n_batches['train'] + batch_id)

                batch_id = self.batch_iterator('train')
                pbar.update()

        return np.mean(loss_record)

    def train(self):
        """
            iterate over epochs
        """
        print('Training...')

        text = ''
        index = 0
        for key, value in vars(dnn_pars_args().parse_args()).items():
            if value is not None:
                index += 1
                text += "{:<20} = {:<10}".format(key, value)
                text += " " * 5 + '|' + " " * 5

                if index % 3 == 0:
                    text += "\n"

        text = text.replace(' ', '.')
        print(text)
        self.write_log(names='DNN Params',
                       logs=text,
                       mode='')

        self.write_log(names='Directories',
                       logs=vars(dir_pars_args().parse_args()),
                       mode='')
        start_time = datetime.datetime.now()

        self.loss_record = []
        for iteration in range(self.args.iteration):
            elapsed_time = datetime.datetime.now() - start_time

            # tf.keras.utils.image_dataset_from_directory(
            #     self.data.data_info['xtrain'],
            #     labels=self.data.data_info['ytrain'],
            #     label_mode='categorical',
            #     color_mode='rgba',
            #     batch_size=self.args.batch_size,
            #     image_size=self.data.input_dim,
            #     shuffle=True,
            #     seed=self.args.seed,
            #     validation_split=None,
            #     subset=None,
            #     interpolation='bilinear',
            #     follow_links=False,
            #     crop_to_aspect_ratio=False,
            #     **kwargs
            # )

            model_loss = self.train_epoch(iteration=iteration)
            self.loss_record.append(model_loss)
            print("%d epoch: time: %s, loss = %s" % (
                iteration + 1,
                elapsed_time,
                model_loss))

            if iteration % self.args.sample_interval == 0:
                self.validate(sample=1, sample_id=iteration)

            if iteration % self.args.validate_interval == 0:
                self.validate(sample=0, sample_id=iteration)

                self.write_log(
                    'model loss',
                    np.mean(self.loss_record),
                    iteration)
                self.loss_record = []

    def validate(self, sample=0, sample_id=None):
        """
                :param sample: sample id
                :return:
                todo: review
        """

        batch_id = self.batch_iterator('val')
        err = [np.Inf]

        val_names = {'val_MAE': [],
                     'err_meter': []}

        imgs = \
            data_helper.img_batch_load(self.data.data_info['xval'],
                                       self.args.batch_size,
                                       batch_id)
        batch_output \
            = self.data.data_info['yval'][batch_id*self.args.batch_size:
                                          (batch_id + 1)*self.args.batch_size]

        outputs = self.model.predict(np.asarray(list(imgs.values())))
        # print(batch_id, 'Validating...')
        for i, ((img_name, img), output) in enumerate(zip(imgs.items(), outputs)):
            output = norm_helper.min_max_norm(output)

            img_gt = np.asarray(batch_output.iloc[i, :])

            output_m = output * (self.data.org_out_max - self.data.org_out_min) \
                       + self.data.org_out_min
            img_gt_m = img_gt * (self.data.org_out_max - self.data.org_out_min) \
                       + self.data.org_out_min

            val_names['val_MAE'].append(np.mean(np.abs(output - img_gt)))
            val_names['err_meter'].append(
                geopy.distance.geodesic((output_m['Lat'], output_m['Long']),
                                        (img_gt_m['Lat'], img_gt_m['Long'])).m)

        if sample == 0:
            self.model.save_weights(const.WEIGHTS_DIR
                                    / 'weights_gen_latest.h5')

            if min(err) > np.mean(val_names['val_MAE']):
                self.model.save_weights(const.WEIGHTS_DIR
                                        / 'weights_gen_best.h5')

            err.append(np.mean(val_names['val_MAE']))

            self.write_log('val_MAE', np.mean(val_names['val_MAE']),
                           sample_id)
            self.write_log('val Error in meters', np.mean(val_names['err_meter']),
                           sample_id)

        else:
            plt.figure()
            # figures equal to the number of z patches in columns
            img_name = img_name.split('/')[-1].split('.')[0]

            plt.title('original lat/long = ' \
                      + str(list(img_gt_m)[:-1]) \
                      + '\nPredicted lat/long =' \
                      + str(list(output_m)[:-1]))

            plt.imshow(img)
            plt.show()

            plt.gca().axes.yaxis.set_ticklabels([])
            plt.gca().axes.xaxis.set_ticklabels([])
            plt.gca().axes.yaxis.set_ticks([])
            plt.gca().axes.xaxis.set_ticks([])
            plt.xlabel('\nError =' 
                      + str(round(val_names['err_meter'][-1], 3)) + ' m')

            result_name = str(sample_id) + '_batch' + str(batch_id) + '_img' + img_name + '.png'
            plt.savefig(const.SAMPLE_DIR / result_name)  # Save sample results
            plt.close("all")  # Close figures to avoid memory leak
