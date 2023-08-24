"""
    Load VBN data generated in simulation
    author: spdkh
    date: June 2023
"""
import glob
import os

import numpy as np
import pandas as pd
import geopy.distance
from sklearn.model_selection import train_test_split
from tensorflow import keras
from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt

from src.data.vbn import VBN
from src.utils import norm_helper, data_helper, const, geo_helper


class Satellite(VBN):
    """
        Load VBN data
    """

    def __init__(self, args):
        """
            args
        """
        VBN.__init__(self, args)
        self.data_types = {'x': 'images'}
        self.img_paths = data_helper.find_files(const.DATA_DIR / self.data_types['x'],
                                                'jpg')
        self.img_paths.sort()
        if not data_helper.check_folder(const.DATA_DIR):
            print('Downloading satellite images...')
            self.gen_data()


    def config(self):
        """
            configuration after init parent
        """
        print('\nNumber of images in the path:', len(self.img_paths))

        # meta_df = pd.read_csv(text_paths[0], sep=':', index_col=0,
        #                       names=[0])
        #
        # print('\nFirst metadata sample:')
        # print(meta_df)

        meta_dfs = []
        for i, path in enumerate(self.img_paths):
            df = pd.Series(map(float, os.path.basename(path).split('.jpg')[0].split('_')))
            meta_dfs.append(df)

        meta_df = pd.concat(meta_dfs, axis=1).transpose()
        meta_df.columns = ['columns', 'row', 'Lat', 'Long', 'Alt']

        print('All metadata:')
        print(meta_df)

        self.network_out = meta_df.loc[:, ['Lat', 'Long', 'Alt']]
        print('Network Outputs:')
        print(self.network_out)
        self.geo_calcs()
        self.train_test_split()

    def gen_data(self):
        center_lat = self.args.coords[0]
        center_lon = self.args.coords[1]
        map_zoom = int(self.args.coords[2])

        # best zooms can be from 15 to 19
        map_data = geo_helper.get_static_map_image(center_lat,
                                                   center_lon,
                                                   zoom=map_zoom)

        top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon = \
            geo_helper.calculate_bounding_box(center_lat,
                                              center_lon,
                                              map_zoom)
        print("Center (Latitude, Longitude):", center_lat, center_lon)
        print("Top Left (Latitude, Longitude):", top_left_lat, top_left_lon)
        print("Bottom Right (Latitude, Longitude):", bottom_right_lat, bottom_right_lon)

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
        plt.show()
        plt.savefig(const.DATA_DIR / "map.jpg")  # Save sample results
        plt.close("all")  # Close figures to avoid memory leak

        self.gen_raster_from_map((top_left_lat, top_left_lon),
                                 (bottom_right_lat, bottom_right_lon),
                                 raster_zoom=18,
                                 overlap=60)

    def gen_raster_from_map(self,
                            top_left_coords: tuple,
                            buttom_right_coords: tuple,
                            raster_zoom: int = 19,
                            overlap: int = 0):
        lat_i, lon_j = top_left_coords

        tl_lat, tl_lon, br_lat, br_lon = \
            geo_helper.calculate_bounding_box(lat_i, lon_j, raster_zoom)
        raster_w = np.abs(tl_lon - br_lon)
        raster_h = np.abs(tl_lat - br_lat)

        print("First (Latitude, Longitude):", top_left_coords)
        print("Last (Latitude, Longitude):", buttom_right_coords)

        # imgs = []
        # labels = []
        i = 0
        j = 0
        data_helper.check_folder(const.DATA_DIR / 'images')
        print('Downloading Images...')
        while lat_i <= buttom_right_coords[0] + raster_h:
            j = 0
            # print('Current Lat', lat_i)
            while lon_j <= buttom_right_coords[-1] + raster_w:
                # print('Current Long: ', lon_j)
                out_name = str(i) + '_' + str(j) + '_' + str(lat_i) + '_' + str(lon_j) + '_' + str(raster_zoom) + '.jpg'

                output_dir = const.DATA_DIR / 'images' / out_name
                raster_data = geo_helper.get_static_map_image(lat_i, lon_j,
                                                              zoom=raster_zoom)

                # print("Top Left (Latitude, Longitude):", tl_lat, tl_lon)
                # print("Bottom Right (Latitude, Longitude):", br_lat, br_lon)
                # print("Width, Height (m):", raster_w, raster_h)
                lon_j += raster_w * (100 - overlap) / 100

                img = np.array(Image.open(BytesIO(raster_data)))
                # imgs.append(img)
                # labels.append([lat_i, lon_j, raster_zoom])
                j += 1

                plt.figure()
                plt.imshow(img)
                plt.show()
                plt.axis('off')
                plt.savefig(output_dir)  # Save sample results
                plt.close("all")  # Close figures to avoid memory leak

            lat_i += raster_h * (100 - overlap) / 100
            lon_j = top_left_coords[-1]

            i += 1

        print(i, j)
