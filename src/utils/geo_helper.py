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

from src.utils import const, data_helper


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
                         api_key='AIzaSyBI743mOAfTFDyuJt-cpTMAy-_58oq-vu8'):
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


def calculate_bounding_box(center_lat, center_lon, zoom, map_size=_map_size):
    """
    Knowing the central lat, long of an image, and the zoom level,
    along with the image size,
    gives you top left and bottom right corners
    :param center_lat:
    :param center_lon:
    :param zoom:
    :param map_size:
    :return:
    """
    # Constants for map dimensions at zoom level 0
    world_size = [256 * (2 ** zoom)] * 2

    # Calculate pixel coordinates of the center
    center_coords = [i / 2 for i in world_size]

    # Calculate distance from center to the edge in pixels
    edge_coords = [i / 2 for i in map_size]

    # Calculate pixel coordinates of the corners
    top_left_x = center_coords[0] - edge_coords[0]
    top_left_y = center_coords[1] - edge_coords[1]
    bottom_right_x = center_coords[0] + edge_coords[0]
    bottom_right_y = center_coords[1] + edge_coords[1]

    # Convert pixel coordinates to latitude and longitude
    top_left_lat = center_lat \
                   + (90 - 360 * math.atan(math.exp(-(top_left_y / world_size[1]
                                                      - 0.5) * 2 * math.pi)) / math.pi)
    top_left_lon = center_lon \
                   + (top_left_x / world_size[0] - 0.5) * 360

    bottom_right_lat = center_lat \
                       + (90 - 360 * math.atan(math.exp(-(bottom_right_y / world_size[1]
                                                          - 0.5) * 2 * math.pi)) / math.pi)
    bottom_right_lon = center_lon + (bottom_right_x / world_size[0] - 0.5) * 360

    return top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon


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


def overlapped(label_a: tuple, label_b: tuple, overlap: int = 25):
    """
    Checking if two images with known label(lat, long, zoom) are overlapped
    more than the desired overlap amount specified
    :param label_a: tuple (lat, long, zoom) of the first image
    :param label_b: tuple (lat, lon, zoom) of the second image
    :param overlap: int [0-100] overlap percentage
    :return: bool
    """
    coords_a = calculate_bounding_box(label_a[0], label_a[1], label_a[2])
    coords_b = calculate_bounding_box(label_b[0], label_b[1], label_b[2])

    y_overlap = max(0, min(coords_a[2], coords_b[2]) - max(coords_a[0], coords_b[0]))
    x_overlap = max(0, min(coords_a[3], coords_b[3]) - max(coords_a[1], coords_b[1]))
    overlap_area = x_overlap * y_overlap
    rect1_area = (coords_a[2] - coords_a[0]) * (coords_a[3] - coords_a[1])
    rect2_area = (coords_b[2] - coords_b[0]) * (coords_b[3] - coords_b[1])
    overlap_percentage = (overlap_area / (rect1_area + rect2_area - overlap_area)) * 100
    if overlap_percentage >= overlap:
        return True
    return False


def gen_raster_from_map(top_left_coords: tuple,
                        buttom_right_coords: tuple,
                        raster_zoom: int = 19,
                        overlap: int = 0):
    """
    Generates raster images from satellite data
    Saves them in the DATA_DIr const address
    :param top_left_coords:  tuple
    :param buttom_right_coords:  tuple
    :param raster_zoom: int
    :param overlap: int
    """
    lat_i, lon_j = top_left_coords

    tl_lat, tl_lon, br_lat, br_lon = \
        calculate_bounding_box(lat_i, lon_j, raster_zoom)
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
            out_name = str(i) + '_' \
                       + str(j) + '_' \
                       + str(lat_i) \
                       + '_' + str(lon_j) \
                       + '_' + str(raster_zoom) + '.jpg'

            output_dir = const.DATA_DIR / 'images' / out_name
            raster_data = get_static_map_image(lat_i, lon_j,
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
