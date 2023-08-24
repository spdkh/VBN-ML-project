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
from src.utils import norm_helper, data_helper, const, geo_helper


class Satellite(Data):
    """
        Load VBN data
    """

    def __init__(self, args):
        """
            args
        """
        Data.__init__(self, args)
        self.data_types = {'x': 'images', 'y': 'MetaData2'}

        if not data_helper.check_folder(const.DATA_DIR):
            self.gen_data()
        self.config()

    def config(self):
        """
            configuration after init parent
        """
        img_paths = data_helper.find_files(const.DATA_DIR / self.data_types['x'],
                                           'jpg')
        img_paths.sort()
        print('\nNumber of images in the path:', len(img_paths))

        meta_df = pd.read_csv(text_paths[0], sep=':', index_col=0,
                              names=[0])

        print('\nFirst metadata sample:')
        print(meta_df)

        meta_dfs = []
        for i, path in enumerate(img_paths):
            df = pd.DataFrame(os.path.basename(path), sep='_', index_col=0, names=[i + 1])
            meta_dfs.append(df.iloc[:, 0])

        meta_df = pd.concat(meta_dfs, axis=1)

        print('All metadata:')
        print(meta_df)
        quit()
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
        self.output_dim = 3  # np.shape(class_ids)
        print('Sample image size:', self.input_dim)
        print('X_train size:', np.shape(self.data_info['xtrain']))
        print('X_val size:', np.shape(self.data_info['xval']))
        print('X_test size:', np.shape(self.data_info['xtest']))
        print('Y_train size:', np.shape(self.data_info['ytrain']))
        print('Y_val size:', np.shape(self.data_info['yval']))
        print('Y_test size:', np.shape(self.data_info['ytest']))

    def gen_data(self):
        center_lat = self.args.coords[0]
        center_lon = self.args.coords[1]
        map_zoom = int(self.args.coords[2])

        # best zooms can be from 15 to 19
        map_data = geo_helper.get_static_map_image(center_lat, center_lon, zoom=map_zoom)

        top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon = geo_helper.calculate_bounding_box(center_lat,
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

        gen_raster_from_map((top_left_lat, top_left_lon),
                            (bottom_right_lat, bottom_right_lon),
                            raster_zoom=18,
                            overlap=60)

    def gen_raster_from_map(top_left_coords: tuple,
                            buttom_right_coords: tuple,
                            raster_zoom: int = 19,
                            overlap: int = 0):
        lat_i, lon_j = top_left_coords

        tl_lat, tl_lon, br_lat, br_lon = calculate_bounding_box(lat_i, lon_j, raster_zoom)
        raster_w = np.abs(tl_lon - br_lon)
        raster_h = np.abs(tl_lat - br_lat)

        print("First (Latitude, Longitude):", top_left_coords)
        print("Last (Latitude, Longitude):", buttom_right_coords)

        # imgs = []
        # labels = []
        i = 0
        j = 0
        while lat_i <= buttom_right_coords[0] + raster_h:
            j = 0
            # print('Current Lat', lat_i)
            while lon_j <= buttom_right_coords[-1] + raster_w:
                # print('Current Long: ', lon_j)
                out_name = str(i) + '_' + str(j) + '_' + str(lat_i) + '_' + str(lon_j) + '_' + str(zoom) + '.jpg'
                output_dir = const.DATA_DIR / 'images' / out_name
                raster_data = get_static_map_image(lat_i, lon_j, zoom=raster_zoom, api_key=api_key)

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
        return imgs, labels, i, j
