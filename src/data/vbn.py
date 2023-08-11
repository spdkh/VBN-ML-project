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


    def config(self):
        """
            configuration after init parent
        """
        img_paths = data_helper.find_files(const.DATA_DIR / self.data_types['x'],
                                           'jpg')
        img_paths.sort()
        print('\nNumber of images in the path:', len(img_paths))

        text_paths = data_helper.find_files(const.DATA_DIR, 'txt')
        text_paths.sort()
        print('Number of texts in the path:', len(text_paths))

        meta_df = pd.read_csv(text_paths[0], sep=':', index_col=0,
                              names=[0])

        print('\nFirst metadata sample:')
        print(meta_df)

        meta_dfs = []
        for i, meta_data in enumerate(text_paths[:-1]):
            df = pd.read_csv(meta_data, sep=':', index_col=0, names=[i + 1])
            meta_dfs.append(df.iloc[:, 0])

        meta_df = pd.concat(meta_dfs, axis=1)

        print('All metadata:')
        print(meta_df)

        network_out = meta_df.loc['Platform_position_LatLongAlt', :]
        network_out = network_out.str.split(" ", expand=True).iloc[:, 1:-1].astype('float64')
        network_out.columns = ['Lat', 'Long', 'Alt']
        print('Network Outputs:')
        print(network_out)

        self.org_out_min = np.min(network_out, axis=0)
        self.org_out_max = np.max(network_out, axis=0)

        print('Min Lat, Long, Alt:', self.org_out_min)
        print('Max Lat, Long, Alt:', self.org_out_max)

        coords_ul = (self.org_out_min['Lat'], self.org_out_min['Long'])
        coords_ur = (self.org_out_max['Lat'], self.org_out_min['Long'])
        coords_dl = (self.org_out_min['Lat'], self.org_out_max['Long'])
        coords_dr = (self.org_out_max['Lat'], self.org_out_max['Long'])

        land_width = geopy.distance.geodesic(coords_ul, coords_ur).km
        land_height = geopy.distance.geodesic(coords_ul, coords_dl).km
        img_diagonal = geopy.distance.geodesic(coords_ul, coords_dr).km
        print('Area Diagonal Distance:', img_diagonal, ' Km')
        print('Width =', land_width, 'Km')
        print('Height =', land_height, 'Km')


        # only applicable if the images form a recangle overall
        land_area = land_width * land_height
        print('Land area = ', land_area, 'Km^2')

        # only applicable if the images forming a rectangle do not overlap
        img_area = land_area / len(network_out.index)
        print('Area covered by each image =', img_area, 'Km^2')



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

        sample_input_img = data_helper.preprocess(data_helper.imread(self.data_info['xtrain'][0]))
        self.input_dim = np.shape(sample_input_img)
        self.output_dim = 3 #np.shape(class_ids)
        print('Sample image size:', self.input_dim)
        print('X_train size:', np.shape(self.data_info['xtrain']))
        print('X_val size:', np.shape(self.data_info['xval']))
        print('X_test size:', np.shape(self.data_info['xtest']))
        print('Y_train size:', np.shape(self.data_info['ytrain']))
        print('Y_val size:', np.shape(self.data_info['yval']))
        print('Y_test size:', np.shape(self.data_info['ytest']))
