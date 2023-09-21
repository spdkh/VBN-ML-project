"""
    Load VBN data generated in simulation
    author: spdkh
    date: June 2023
"""
import os
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

from src.data.vbn import VBN
from src.utils import data_helper, const, geo_helper
from src.utils.data_helper import pretty



class Satellite(VBN):
    """
        Load VBN data
    """

    def __init__(self, args):
        """
            args
        """
        VBN.__init__(self, args)

        if not data_helper.check_folder(const.DATA_DIR):
            pretty('Downloading satellite images...')
            self.gen_data()

        self.data_types = {'x': 'images'}
        self.img_paths = data_helper.find_files(const.DATA_DIR / self.data_types['x'],
                                                'jpg')
        self.img_paths.sort()

    def config(self):
        """
            configuration after init parent
        """
        pretty('Number of images in the path:', len(self.img_paths))

        meta_dfs = []
        for path in self.img_paths:
            df_i = pd.Series(map(float, os.path.basename(path).split('.jpg')[0].split('_')))
            meta_dfs.append(df_i)

        pretty('First metadata sample:\n',meta_dfs)
        meta_df = pd.concat(meta_dfs, axis=1).transpose()
        meta_df.columns = ['columns', 'row', 'Lat', 'Long', 'Alt']

        pretty('All metadata:\n', meta_df)

        self.network_out = meta_df.loc[:, ['Lat', 'Long', 'Alt']]
        pretty('Network Outputs\n:', self.network_out)
        
        self.geo_calcs()
        self.my_train_test_split()

    def gen_data(self):
        """
            Generate Satellite data from a given big picture map
        :return:
        """
        top_left = self.args.coords[0], self.args.coords[1]
        buttom_right = self.args.coords[2], self.args.coords[3]

        map_ratio = abs(top_left[0] - buttom_right[0]) / abs(top_left[1] - buttom_right[1])
        map_w = 400
        map_h = int(map_w * map_ratio)
        map_size = [map_w, map_h]
        center_lat = (top_left[0] + buttom_right[0]) / 2
        center_lon = (top_left[1] + buttom_right[1]) / 2

        map_zoom = int(self.args.coords[-1])

        # best zooms can be from 15 to 19
        map_data = geo_helper.get_static_map_image(center_lat,
                                                   center_lon,
                                                   zoom=map_zoom,
                                                   size=map_size)

        top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon = \
            geo_helper.calculate_bounding_box((center_lat,
                                              center_lon),
                                              map_zoom,
                                              map_size=map_size)

        pretty("[INFO]"
            , "\n\tCenter (Latitude, Longitude):", center_lat, center_lon
            , "\n\tTop Left (Latitude, Longitude):", top_left_lat, top_left_lon
            , "\n\tBottom Right (Latitude, Longitude):", bottom_right_lat, bottom_right_lon)

        img = Image.open(BytesIO(map_data))
        plt.imshow(img)
        # Set y-axis (latitude) labels
        plt.yticks([0, img.size[1]], [np.round(top_left_lat, 4),
                                      np.round(bottom_right_lat, 4)])

        # Set x-axis (longitude) labels
        plt.xticks([0, img.size[0]], [np.round(top_left_lon, 4),
                                      np.round(bottom_right_lon, 4)])

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Satellite Image")
        # plt.show()
        plt.savefig(const.DATA_DIR / "map.jpg")  # Save sample results
        plt.close("all")  # Close figures to avoid memory leak

        geo_helper.gen_raster_from_map((top_left_lat, top_left_lon),
                                       (bottom_right_lat, bottom_right_lon),
                                       (400, 400),
                                       raster_zoom=18,
                                       overlap=0)

        geo_helper.gen_raster_from_map((top_left_lat, top_left_lon),
                                       (bottom_right_lat, bottom_right_lon),
                                       (400, 400),
                                       raster_zoom=19,
                                       overlap=0)
