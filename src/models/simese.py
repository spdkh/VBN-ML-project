# -*- coding: utf-8 -*-
"""
    simese net
    author: SPDKH
    date: Aug 2023
"""
import os
import datetime
from pathlib import Path

import geopy
from tqdm import tqdm
import numpy as np
import visualkeras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

from src.models.vbnnet import VBNNET
from src.utils import const, data_helper, norm_helper
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
        print("[INFO] preparing positive and negative pairs...")
        self.pairs = {}
        for mode in ['train', 'val', 'test']:
            self.pairs['x' + mode], self.pairs['y' + mode] = \
                make_pairs(self.data_info['x' + mode],
                           self.data_info['y' + mode])

        self.build_simese()

    def build_model(self):
        print("[INFO] building siamese network...")
        img_a = Input(shape=self.data.input_dim)
        img_b = Input(shape=self.data.input_dim)
        feature_extractor = Model(self.model_input,
                                  vgg16(self.model_input, 3))
        feats_a = feature_extractor(img_a)
        feats_b = feature_extractor(img_b)

        # finally, construct the siamese network
        distance = Lambda(euclidean_distance)([feats_a, feats_b])
        outputs = Dense(1, activation="sigmoid")(distance)
        self.model = Model(inputs=[img_a, img_b],
                           outputs=outputs)
        # compile the model
        print("[INFO] compiling model...")
        self.model.compile(loss="binary_crossentropy",
                           optimizer="adam",
                           metrics=["accuracy"])

        print(self.model.summary())
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
        with tqdm(total=self.args.batch_iter) as pbar:
            # while batch_id != 0:
            for _ in range(self.args.batch_iter):
                batch_pairs = \
                    pair_batch_load(self.pairs['xtrain'],
                                    self.args.batch_size,
                                    batch_id)

                batch_outputs \
                    = self.pairs['ytrain'][self.args.batch_size * batch_id:
                                           self.args.batch_size * batch_id + self.args.batch_size]

                # train_datagen = ImageDataGenerator(
                #     rescale=1. / 255,
                #     rotation_range=360,
                #     width_shift_range=0.6,
                #     height_shift_range=0.6,
                #     horizontal_flip=True,
                #     fill_mode='nearest')

                batch_loss = self.model.train_on_batch(batch_pairs, batch_outputs)
                loss_record.append(batch_loss)

                # print('Train with augmented data:')
                # for i, (img, batch_output) in enumerate(train_datagen.flow(
                #         batch_imgs,
                #         y=batch_outputs,
                #         batch_size=self.args.batch_size,
                #         shuffle=True,
                #         seed=self.args.seed,
                #         ignore_class_split=True,
                # )):
                #     # output = self.model.predict(img)
                #     # print(i, img.shape, batch_output.shape)
                #     # output = norm_helper.min_max_norm(output)
                #     batch_loss = self.model.train_on_batch(img, batch_output)
                #     loss_record.append(batch_loss)
                #
                #     if i > self.args.n_augment:
                #         break

                elapsed_time = datetime.datetime.now() - start_time
                batch_loss = np.mean(loss_record)
                if batch_log:
                    print(f"{batch_id} batch iteration: time: "
                          f"{elapsed_time}, batch_loss = {batch_loss}")

                self.write_log(
                    'full train loss',
                    batch_loss,
                    iteration * self.n_batches['train'] + batch_id)

                batch_id = self.batch_iterator('train')
                pbar.update()

        return np.mean(loss_record)


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (feats_a, feats_b) = vectors
    # compute the sum of squared distances between the vectors
    sum_squared = K.sum(K.square(feats_a - feats_b),
                        axis=1,
                        keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))


def make_pairs(images, labels):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pair_images = []
    pair_labels = []

    # loop over all images
    for idx_a in range(len(images)):
        # grab the current image and label belonging to the current iteration
        cur_img = images[idx_a]
        label = labels[idx_a]
        # randomly pick an image that belongs to the *same* class label
        idx_b = np.random.choice(range(len(images)))
        label_b = labels[idx_b]
        if overlapped(label, label_b):
            pos_img = images[idx_b]
            while overlapped(label, label_b):
                idx_b = np.random.choice(range(len(images)))
                label_b = labels[idx_b]
            # grab the indices for each of the class labels *not* equal to
            # the current label and randomly pick an image corresponding
            # to a label *not* equal to the current label
            neg_img = images[idx_b]
        else:
            neg_img = images[idx_b]
            while not overlapped(label, label_b):
                idx_b = np.random.choice(range(len(images)))
                label_b = labels[idx_b]
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
    # return a 2-tuple of our image pairs and labels
    return np.array(pair_images), np.array(pair_labels)


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
    for i in range(iteration, batch_size + iteration):
        pair = pairs[i]
        image_batch.append([])
        for j in range(2):
            img = imread(pair[j])
            image_batch[i].append(img)
    return image_batch
