"""
    Geolocation helper functions
    author: spdkh
    date: Aug 2023
"""
import math
from io import BytesIO

import numpy as np
from PIL import Image
import requests
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.utils import const, data_helper
from src.hidden_file import api_key


def spherical_mercator_to_lat_lon(x, y):
    """
    Converts spherical mercator x, y to lat, long
    :param x:
    :param y:
    :return:
    todo: needs to be fixed
    """
    r_earth = 6378137.0  # Earth's radius in meters
    # tile_size = 256.0  # Size of a tile in Web Mercator projection

    # Reverse the scaling and shifting applied during conversion to Web Mercator
    # x = x * tile_size - r_earth
    # y = y * tile_size - r_earth

    # Convert x and y to latitude and longitude in radians
    lon_rad = math.radians(x / r_earth)
    lat_rad = math.atan(math.exp(y / r_earth))
    lat_rad = 2.0 * math.atan(math.exp(y / r_earth)) - math.pi / 2.0

    # Convert latitude and longitude to degrees
    latitude = math.degrees(lat_rad)
    longitude = math.degrees(lon_rad)

    return latitude, -(90 + longitude)


def lat_lon_to_spherical_mercator(lat, lon):
    """
        Converts lat, long to x, y in spherical mercator

    :param lat:
    :param lon:
    :return:

    todo: not accurate
    """
    r_earth = 6378137.0  # Earth's radius in meters
    # tile_size = 256.0  # Size of a tile in Web Mercator projection

    # Convert latitude and longitude to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # Perform the conversion
    x = r_earth * lon_rad
    y = r_earth * math.log(math.tan(math.pi / 4.0 + lat_rad / 2.0))

    # Scale and shift to fit within the Web Mercator coordinate system
    # x = (x + r_earth) / tile_size
    # y = (y + r_earth) / tile_size

    return x, y


_map_size = (400, 400)


def get_static_map_image(lat, lon, zoom=15, size=_map_size,
                         api_key=api_key):
    """
        Dowload google earth image of the given
        central lat, long and zoom level and desired image size
    :param lat:
    :param lon:
    :param zoom:
    :param size:
    :param api_key:
    :return:
    """
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": f"{size[0]}x{size[1]}",
        "maptype": "satellite",
        "key": api_key,
    }
    response = requests.get(base_url, params=params)
    return response.content


_C = {'x': 128, 'y': 128}
_J = 256 / 360
_L = 256 / (2 * math.pi)
def calculate_bounding_box(center, zoom, map_size):
    center_lat, center_lon = center
    size_x, size_y = map_size

    pixel_size = pow(2, -(zoom + 1))
    pw_x = size_x * pixel_size
    pw_y = size_y * pixel_size

    a = min(max(math.sin(math.radians(center[0])), -(1 - 1E-15)), 1 - 1E-15)
    cp = {
        'x': _C['x'] + center[1] * _J,
        'y': _C['y'] + 0.5 * math.log((1 + a) / (1 - a)) * -_L
    }

    return ptToLatlon({'x': cp['x'] - pw_x, 'y': cp['y'] - pw_y})\
        + ptToLatlon({'x': cp['x'] + pw_x, 'y': cp['y'] + pw_y})


def ptToLatlon(pt):
    return [
        math.degrees(2 * math.atan(math.exp((pt['y'] - _C['y']) / -_L)) - math.pi / 2),
        (pt['x'] - _C['x']) / _J
    ]


def inverse_calculate_bounding_box(top_left_lat,
                                   top_left_lon,
                                   bottom_right_lat,
                                   bottom_right_lon,
                                   map_width,
                                   map_height):
    """
    Gets top left and bottom right coordinates of image in degrees
    along with image size,
    return lat, long, of the center and zoom level
    (Inverse of the bounding box calculator function)
    :param top_left_lat: float
    :param top_left_lon: float
    :param bottom_right_lat: float
    :param bottom_right_lon: float
    :param map_width: int
    :param map_height: int
    :return: tuple(lat, lon, zoom)
    """
    center_lat = (top_left_lat + bottom_right_lat) / 2
    center_lon = (top_left_lon + bottom_right_lon) / 2

    lat_range = abs(top_left_lat - bottom_right_lat)
    lon_range = abs(top_left_lon - bottom_right_lon)

    lat_degrees_per_pixel = lat_range / map_height
    lon_degrees_per_pixel = lon_range / map_width

    zoom = int(-math.log2(max(lat_degrees_per_pixel, lon_degrees_per_pixel)))

    return center_lat, center_lon, zoom


def overlapped(label_a: tuple, label_b: tuple, img_size, overlap: int = 25):
    """
    Checking if two images with known label(lat, long, zoom) are overlapped
    more than the desired overlap amount specified
    :param label_a: tuple (lat, long, zoom) of the first image
    :param label_b: tuple (lat, lon, zoom) of the second image
    :param overlap: int [0-100] overlap percentage
    :return: bool
    """
    coords_a = calculate_bounding_box(label_a[0: 2], label_a[2], img_size)
    coords_b = calculate_bounding_box(label_b[0: 2], label_b[2], img_size)
    # print(coords_b, coords_a)
    # print(min(coords_a[0], coords_b[0] - max(coords_a[2], coords_b[2])))
    # print(min(coords_a[3], coords_b[3]) - max(coords_a[1], coords_b[1]))

    y_overlap = max(0, min(coords_a[0], coords_b[0]) - max(coords_a[2], coords_b[2]))
    x_overlap = max(0, min(coords_a[3], coords_b[3]) - max(coords_a[1], coords_b[1]))
    overlap_area = x_overlap * y_overlap
    rect1_area = abs(coords_a[2] - coords_a[0]) * abs(coords_a[3] - coords_a[1])
    rect2_area = abs(coords_b[2] - coords_b[0]) * abs(coords_b[3] - coords_b[1])
    overlap_percentage = (overlap_area / (rect1_area + rect2_area - overlap_area)) * 100
    if overlap_percentage >= overlap:
        return True
    return False


def gen_raster_from_map(top_left_coords: tuple,
                        bottom_right_coords: tuple,
                        im_size,
                        raster_zoom: int = 19,
                        overlap: int = 0):
    """
    Generates raster images from satellite data
    Saves them in the DATA_DIr const address
    :param top_left_coords:  tuple
    :param bottom_right_coords:  tuple
    :param raster_zoom: int
    :param overlap: int
    """
    lat_i, lon_j = top_left_coords

    tl_lat, tl_lon, br_lat, br_lon = \
        calculate_bounding_box(top_left_coords, raster_zoom, im_size)
    raster_w = np.abs(tl_lon - br_lon)
    raster_h = np.abs(tl_lat - br_lat)

    map_size_geo = abs(np.subtract(top_left_coords, bottom_right_coords))
    n_images_x = int(2 + (map_size_geo[1] - raster_w)\
                /(((100 - overlap) / 100) * raster_w))
    n_images_y = int(2 + (map_size_geo[0] - raster_h)\
                /(((100 - overlap) / 100) * raster_h))
    print('\n[GEO Helper]')
    print("\tFirst (Latitude, Longitude):", top_left_coords)
    print("\tLast (Latitude, Longitude):", bottom_right_coords)
    print("\tNumber of Images (X, y):", n_images_x, n_images_y)

    # imgs = []
    # labels = []
    i = 0
    j = 0
    data_helper.check_folder(const.DATA_DIR / 'images')
    print('\n[GEO Helper] Downloading Images...')
    with tqdm(total=n_images_x*n_images_y) as pbar:
        for j in range(n_images_y):
            # print('\t Current Lat', lat_i)
            for i in range(n_images_x):
                # print('\t Current Long: ', lon_j)
                out_name = str(i) + '_' \
                           + str(j) + '_' \
                           + str(lat_i) \
                           + '_' + str(lon_j) \
                           + '_' + str(raster_zoom) + '.jpg'

                output_dir = const.DATA_DIR / 'images' / out_name
                raster_data = get_static_map_image(lat_i, lon_j,
                                                   zoom=raster_zoom,
                                                   size=im_size)

                lon_j += raster_w * (100 - overlap) / 100

                try:
                    img = np.array(Image.open(BytesIO(raster_data)))
                except:
                    print('Exception happened loading image', lat_i, lon_j, raster_zoom)
                    continue

                plt.figure()
                plt.imshow(img)
                # plt.show()
                plt.axis('off')
                plt.savefig(output_dir)  # Save sample results
                plt.close("all")  # Close figures to avoid memory leak
                pbar.update()
            lat_i -= raster_h * (100 - overlap) / 100
            lon_j = top_left_coords[-1]

    print('\t Number of rows and columns:', i, j)
