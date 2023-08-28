"""
    author: SPDKH

    Data Manager
"""
from abc import ABC
import os

import numpy as np
import pandas as pd
import tifffile as tiff
from matplotlib import pyplot as plt

from src.utils import const


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
