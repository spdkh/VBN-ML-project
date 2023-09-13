# -*- coding: utf-8 -*-
"""
    simese net
    author: SPDKH
    date: Aug 2023
"""
import os
import psutil
import datetime
from pathlib import Path

import geopy
from tqdm import tqdm
import numpy as np
import visualkeras
import sklearn
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense
import tensorflow.keras.backend as K

from src.models.vbnnet import VBNNET
from src.utils import const, data_helper, norm_helper, geo_helper
from src.utils.architectures import transfer_learning, feature_extraction
from src.utils.architectures.transfer_learning import vgg16
from src.utils.config import dnn_pars_args, dir_pars_args


class Simese(VBNNET):
    """
        Simese Netwokr implementation
        DLL implementation
    """

    def __init__(self, args):
        """
            params:
                args: argparse object
        """
        VBNNET.__init__(self, args)

        # build the positive and negative image pairs
        print("[Simese] preparing positive and negative pairs...")
        self.pairs = {}
        for mode in ['train', 'val', 'test']:
            print('\t', mode, 'pairs...')
            self.pairs['x' + mode], self.pairs['y' + mode] = \
                self.make_pairs(self.data.data_info['x' + mode],
                                np.asarray(self.data.data_info['y' + mode]))
        self.feature_extractor = None
        self.build_model()

    def build_model(self):
        print("[Simese] building siamese network...")
        img_a = Input(shape=self.data.input_dim)
        img_b = Input(shape=self.data.input_dim)
        vgg_o = vgg16(self.model_input, 4)
        self.model_output = feature_extraction.build_siamese_model(vgg_o)
        self.feature_extractor = Model(self.model_input,
                                       self.model_output)
        feats_a = self.feature_extractor(img_a)
        feats_b = self.feature_extractor(img_b)

        # finally, construct the siamese network
        distance = Lambda(euclidean_distance)([feats_a, feats_b])
        outputs = Dense(1, activation="sigmoid")(distance)
        self.model = Model(inputs=[img_a, img_b],
                           outputs=outputs)

        # compile the model
        print("[Simese] compiling model...")
        self.model.compile(loss="binary_crossentropy",
                           optimizer="adam",
                           metrics=["accuracy"])

        print('[Simese] Model Summary:\n', self.model.summary())
        # tf.keras.utils.plot_model(self.model, to_file='Model.png',
        # show_shapes=True, dpi=64, rankdir='LR')
        # write to disk
        visualkeras.layered_view(self.model, draw_volume=False, legend=True, to_file='Model2.png')

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

        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 ** 3)  # Memory usage in GB
        print(f"['Simese']\tMemory Usage: {memory_usage} GB")

        with tqdm(total=self.args.batch_iter) as pbar:
            # while batch_id != 0:
            for _ in range(self.args.batch_iter):
                batch_pairs, batch_outputs = self.load_batch('train', batch_id)

                train_datagen = ImageDataGenerator(
                    rotation_range=360,
                    zoom_range=.5,
                    width_shift_range=0,
                    height_shift_range=0,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    preprocessing_function=norm_helper.min_max_norm)

                batch_loss = self.model.train_on_batch([batch_pairs[0, :, :, :, :],
                                                        batch_pairs[1, :, :, :, :]],
                                                        batch_outputs)

                loss_record.append(batch_loss)

                # print('Train with augmented data:')
                # for i, (img, batch_pair) in enumerate(train_datagen.flow(
                #         batch_pairs[1, :, :, :, :],
                #         y=batch_outputs,
                #         batch_size=self.args.batch_size,
                #         shuffle=True,
                #         seed=self.args.seed,
                #         ignore_class_split=True,
                # )):
                #     # output = self.model.predict(img)
                #     # print(i, img.shape, batch_output.shape)
                #     # output = norm_helper.min_max_norm(output)
                #     batch_loss = self.model.train_on_batch([batch_pairs[0, :, :, :, :],
                #                                         batch_pair],
                #                                         batch_outputs)
                #     loss_record.append(batch_loss)
                
                #     if i > self.args.n_augment:
                #         break

                elapsed_time = datetime.datetime.now() - start_time
                batch_loss = np.mean(loss_record)
                if batch_log:
                    print(batch_id, 'batch iteration: time:',
                          elapsed_time, 'batch_loss = ', batch_loss)

                self.write_log(
                    'full train loss',
                    batch_loss,
                    iteration * self.n_batches['train'] + batch_id)

                batch_id = self.batch_iterator('train')
                pbar.update()

        return np.mean(loss_record)

    def validate(self, sample=0, sample_id=None, mode='val'):
        """
                :param sample: sample id
                :return:
                todo: review
        """

        batch_id = self.batch_iterator(mode)
        err = [np.Inf]

        metrics = {'Acc': [], 'BCE': []}

        imgs, batch_output = self.load_batch(mode, batch_id)
        outputs = self.model.predict([imgs[0, :, :, :, :],
                                     imgs[1, :, :, :, :]])

        thr = 0.5
        outputs = np.where(outputs > thr, 1, 0)
        # for i, output in enumerate(outputs):
        # print(batch_output, outputs)
        metrics['Acc'].append(sklearn.metrics.accuracy_score(batch_output,
                                                             outputs))
        metrics['BCE'].append(sklearn.metrics.log_loss(batch_output,
                                                       outputs))
        if sample == 0:
            self.feature_extractor.save_weights(const.WEIGHTS_DIR
                                                / 'feater_extractor_latest.h5')
            self.model.save_weights(const.WEIGHTS_DIR
                                    / 'weights_gen_latest.h5')

            if min(err) > np.mean(metrics['BCE']):
                self.feature_extractor.save_weights(const.WEIGHTS_DIR
                                                    / 'feater_extractor_best.h5')
                self.model.save_weights(const.WEIGHTS_DIR
                                        / 'weights_gen_best.h5')

        else:
            err.append(np.mean(metrics['BCE']))

            self.write_log(mode + '_Accuracy', np.mean(metrics['Acc']),
                           sample_id)
            self.write_log(mode + '_Loss', np.mean(metrics['BCE']),
                           sample_id)

    def load_batch(self, mode, batch_id):
        """

        """
        imgs = \
            pair_batch_load(self.pairs['x' + mode],
                                        self.args.batch_size,
                                        batch_id)
        imgs = np.squeeze(np.asarray(imgs))
        imgs = imgs.transpose((1, 0, 2, 3, 4))

        batch_output \
            = self.pairs['y' + mode][batch_id * self.args.batch_size:
                                     (batch_id + 1) * self.args.batch_size]
        return imgs, batch_output

    def make_pairs(self, images, labels):
        # initialize two empty lists to hold the (image, image) pairs and
        # labels to indicate if a pair is positive or negative
        pair_images = []
        pair_labels = []

        # loop over all images
        with tqdm(total=len(images)) as pbar:

            for idx_a in range(len(images)):
                # grab the current image and label belonging to the current iteration
                cur_img = images[idx_a]
                label = self.data.norm_geo2geo(labels[idx_a])

                # randomly pick an image that belongs to the *same* class label
                idx_b = np.random.choice(range(len(images)))
                label_b = self.data.norm_geo2geo(labels[idx_b])
                # print(geo_helper.overlapped(label, label, (400, 400)))
                # quit()
                if geo_helper.overlapped(label, label_b, (400, 400)):
                    pos_img = images[idx_b]
                    while geo_helper.overlapped(label, label_b, (400, 400)):
                        idx_b = np.random.choice(range(len(images)))
                        label_b = self.data.norm_geo2geo(labels[idx_b])
                    # grab the indices for each of the class labels *not* equal to
                    # the current label and randomly pick an image corresponding
                    # to a label *not* equal to the current label
                    neg_img = images[idx_b]
                else:
                    neg_img = images[idx_b]
                    while not geo_helper.overlapped(label, label_b, (400, 400)):
                        idx_b = np.random.choice(range(len(images)))
                        label_b = self.data.norm_geo2geo(labels[idx_b])
                    # grab the indices for each of the class labels equal to
                    # the current label and randomly pick an image corresponding
                    # to a label equal to the current label
                    pos_img = images[idx_b]
                # prepare a positive pair and update the images and labels
                # lists, respectively
                pair_images.append([cur_img, pos_img])
                pair_labels.append([1])
                # prepare a negative pair of images and update our lists
                pair_images.append([cur_img, neg_img])
                pair_labels.append([0])
                pbar.update()
        # return a 2-tuple of our image pairs and labels
        return np.array(pair_images), np.array(pair_labels)


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (feats_a, feats_b) = vectors
    # compute the sum of squared distances between the vectors
    sum_squared = K.sum(K.square(feats_a - feats_b),
                        axis=1,
                        keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))


def pair_batch_load(pairs, batch_size, iteration):
    """
        Parameters
        ----------
        pairs: tuple of two str paths

        iteration: int
            batch iteration id to load the right batch
            pass batch_iterator(.) directly if loading batches
            this updates the batch id,
            then passes the updated value
            can leave 0 if
        batch_size: int
            if not loading batches,
            keep it the same as number of all samples loading

        Returns: array
            loaded batch of raw images
        -------
        """
    iteration = iteration * batch_size
    image_batch = []
    for i in range(batch_size):
        pair = pairs[i + iteration]
        image_batch.append([])
        # print('Empty batch:', image_batch[i])
        for j in range(2):
            img = data_helper.imread(pair[j])
            image_batch[i].append(img)
    return image_batch
