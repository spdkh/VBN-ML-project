"""
    author: SPDKH

    Data Manager
"""
from abc import ABC, abstractmethod
import os

import numpy as np
import pandas as pd
import tifffile as tiff
from matplotlib import pyplot as plt
from skimage.measure import block_reduce

from src.utils import const
from src.utils import norm_helper


class Data(ABC):
    """
        Abstract Data Manager Class
    """

    def __init__(self, args):
        self.args = args
        super().__init__()

        self.data_groups = {'train': 'training',
                            'test': 'testing',
                            'val': 'validation'}

        self.data_types = {'x': 'rawdata', 'y': 'gt'}

        # self.norm = getattr(norm_helper, self.args.norm + '_norm')
        self.data_info = {}
        self.otf_path = None

        self.input_dim = None
        self.output_dim = None

    def config(self):
        """
            Initial configuration to call in the children classes
        """
        # Load Input Sample
        input_dir = const.DATA_DIR \
                    / self.data_groups['train'] \
                    / self.data_types['x']

        self.input_dim = self.load_sample(input_dir)
        print('Input Image shape:', self.input_dim)

        # Load Output Sample
        output_dir = const.DATA_DIR \
                     / self.data_groups['train'] \
                     / self.data_types['y']

        self.output_dim = self.load_sample(output_dir)
        print('Output Image shape:', self.output_dim)

        for data_group, data_dir in self.data_groups.items():
            self.data_info[data_group] = \
                const.DATA_DIR / data_dir
            for data_type, values in self.data_types.items():
                self.data_info[data_type + data_group] = \
                    self.data_info[data_group] / values

        print(self.data_info)

    def load_sample(self, parent_path, show=0, sep='\t'):
        """
        convert any type of data to [x, y, z, ch] then return dimension
        """
        path = os.listdir(parent_path)[0]
        img_formats = ['png', 'PNG', 'JPG', 'jpg', 'tif']
        table_formats = ['xlsx', 'csv', 'txt']
        if any(ext in path for ext in img_formats):
            sample = np.transpose(tiff.imread(path), (1, 2, 0))
            sample_size = list(np.shape(sample))
            if sample_size[-1] % (self.args.n_phases * self.args.n_angles) == 0:
                sample_size[-1] = sample_size[-1] // (self.args.n_phases * self.args.n_angles)
                sample_size.append(self.args.n_phases * self.args.n_angles)
            else:
                sample_size.append(1)
            if show:
                plt.figure()
                plt.imshow(sample[:, :, 0])
                plt.show()
        elif any(ext in path for ext in table_formats):
            sample = pd.read_csv(path, sep=sep)
            print(sample)
            sample_size = [1]
        return sample_size

    def image2image_batch_load(self,
                               batch_size: int,
                               iteration: int = 0,
                               scale: int = 1,
                               mode: str = 'train'):
        """
        Parameters
        ----------
        mode: str
            options: "train" or "test" or "val"
        iteration: int
            batch iteration id to load the right batch
            pass batch_iterator(.) directly if loading batches
            this updates the batch id,
            then passes the updated value
            can leave 0 if
        batch_size: int
            if not loading batches,
            keep it the same as number of all samples loading
        scale: int = 1
            image to image translation scale factor
            ratio of gt size vs raw data size for super-resolution
            leave 1 if not doing super-resolution

        Returns: tuple
            loaded batch of raw images,
            loaded batch of ground truth
        -------
        """
        batch_images_path = os.listdir(self.data_info['x' + mode])
        gt_images_path = os.listdir(self.data_info['y' + mode])

        batch_images_path.sort()
        gt_images_path.sort()
        x_path = self.data_info['x' + mode]
        y_path = self.data_info['y' + mode]

        iteration = iteration * batch_size
        batch_images_path = batch_images_path[iteration:batch_size + iteration]
        gt_images_path = gt_images_path[iteration:batch_size + iteration]

        image_batch = []
        gt_batch = []
        for i, _ in enumerate(batch_images_path):
            cur_img = tiff.imread(x_path /
                                  batch_images_path[i])
            cur_img[cur_img < 0] = 0

            cur_gt = tiff.imread(y_path
                                 / gt_images_path[i])
            cur_gt[cur_gt < 0] = 0

            cur_img = self.norm(np.array(cur_img))
            cur_gt = self.norm(np.array(cur_gt))
            image_batch.append(cur_img)
            gt_batch.append(cur_gt)

        image_batch = np.array(image_batch)
        gt_batch = np.array(gt_batch)

        image_batch = np.reshape(image_batch,
                                 (batch_size,
                                  image_batch.shape[1] // self.input_dim[2],
                                  self.input_dim[2],
                                  self.input_dim[1],
                                  self.input_dim[0]),
                                 order='F').transpose((0, 3, 4, 2, 1))

        gt_batch = gt_batch.reshape((batch_size,
                                     self.input_dim[2],
                                     self.input_dim[1] * scale,
                                     self.input_dim[0] * scale,
                                     1),
                                    order='F').transpose((0, 2, 3, 1, 4))

        return image_batch, gt_batch
