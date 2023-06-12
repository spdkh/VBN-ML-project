"""
    Load VBN data generated in simulation
    author: spdkh
    date: June 2023
"""
import glob

import numpy as np
import pandas as pd
import geopy.distance
from sklearn.model_selection import train_test_split
from tensorflow import keras

from src.data.data import Data
from src.utils import norm_helper, data_helper, const


class VBN(Data):
    """
        Load VBN data
    """

    def __init__(self, args):
        """
            args
        """
        Data.__init__(self, args)

        self.data_types = {'x': 'JPG2', 'y': 'MetaData2'}

        self.config()
        #
        # coords_start = (35.110000, -89.800000)
        # coords_end =   (35.130000, -89.820000)
        #
        # dist = geopy.distance.geodesic(coords_start, coords_end).km
        # print(dist)
        # print(dist/np.sqrt(2))

    def config(self):
        """
            configuration after init parent
        """
        img_paths = data_helper.find_files(const.DATA_DIR / 'JPG2', 'jpg')
        img_paths.sort()
        print('number of images in the path:', len(img_paths))

        text_paths = data_helper.find_files(const.DATA_DIR, 'txt')
        text_paths.sort()
        print('number of texts in the path:', len(text_paths))

        meta_df = pd.read_csv(text_paths[0], sep=':', index_col=0,
                              names=[0])

        print('First metadata sample:')
        print(meta_df)

        for i, meta_data in enumerate(text_paths[1:-1]):
            df = pd.read_csv(meta_data, sep=':', index_col=0, names=[i + 1])
            meta_df = pd.concat((meta_df.loc[:, :], df.iloc[:, 0]), axis=1)

        print('All metadata:')
        print(meta_df)

        network_out = meta_df.loc['Platform_position_LatLongAlt', :]
        network_out = network_out.str.split(" ", expand=True).iloc[:, 1:-1].astype('float64')
        network_out.columns = ['Lat', 'Long', 'Alt']
        print('Network Outputs:')
        print(network_out)

        y_normalized = norm_helper.min_max_norm(network_out)
        print('Normalized outputs (y_normalized):')
        print(y_normalized)

        class_ids = keras.utils.to_categorical(range(len(network_out.columns))).tolist()
        print('class ids:', class_ids, np.shape(class_ids))

        self.data_info['xtrain'], x_test, self.data_info['ytrain'], y_test \
            = train_test_split(img_paths, y_normalized,
                               test_size=0.2,
                               random_state=self.args.seed)
        self.data_info['xval'], self.data_info['xtest'], self.data_info['yval'], self.data_info['ytest'] \
            = train_test_split(x_test, y_test,
                               test_size=0.5,
                               random_state=self.args.seed)

        self.input_dim = np.shape(data_helper.imread(self.data_info['xtrain'][0]))
        self.output_dim = np.shape(class_ids)
        print('Sample image size:', self.input_dim)
        # print('X_train size:', np.shape(self.x_train))
        # print('X_test size:', np.shape(self.x_test))
        # print('Y_train size:', np.shape(self.y_train))
        # print('Y_test size:', np.shape(self.y_test))
