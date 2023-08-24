"""
    Path helper functions
"""
import os
import glob


import geopy.point
from PIL.ExifTags import TAGS, GPSTAGS
from PIL import Image
import numpy as np
import cv2
import tifffile as tiff

from src.utils import norm_helper


def check_folder(log_dir):
    """
        check if directory does not exist,
        make it.

        params:

            log_dir: str
                directory to check
    """
    if os.path.exists(log_dir):
        print(log_dir, 'Folder Exists.')
        return True
    print('Creating Folder', log_dir)
    os.makedirs(log_dir)
    return False



def find_files(path, ext):
    """
        params:

        path: str
            parent folder
        ext: str
            file extension

        returns: list
            list of directories of all files with given extention in the traverse directory
    """

    file_paths = []
    for folder_path in os.walk(path):
        file_paths.extend(glob.glob(folder_path[0] + '/*.' + ext))
    return file_paths


def img_batch_load(imgs_paths,
                   batch_size: int,
                   iteration: int = 0):
    """
    Parameters
    ----------
    path: str

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
    # imgs_paths = os.listdir(path)

    imgs_paths.sort()

    iteration = iteration * batch_size
    imgs_paths = imgs_paths[iteration:batch_size + iteration]

    image_batch = dict()
    for i, path in enumerate(imgs_paths):
        cur_img = imread(imgs_paths[i])
        image_batch[path] = preprocess(cur_img)
    return image_batch


small = (615, 515)
one_k = (1024, 768)
two_k = (2048, 1536)


def imread(img_path, shape=small):
    img = Image.open(img_path)
    im = img.resize(shape)
    im = norm_helper.min_max_norm(np.asarray(im))
    return im


def metadata_read(img_path):
    img = Image.open(img_path)

    if 'exif' in img.info.keys():

        # build reverse dicts
        _TAGS_r = dict(((v, k) for k, v in TAGS.items()))
        _GPSTAGS_r = dict(((v, k) for k, v in GPSTAGS.items()))

        exifd = img._getexif()  # this merges gpsinfo as data rather than an offset pointer
        if "GPSInfo" in _TAGS_r.keys():
            gpsinfo = exifd[_TAGS_r["GPSInfo"]]

            lat = gpsinfo[_GPSTAGS_r['GPSLatitude']], gpsinfo[_GPSTAGS_r['GPSLatitudeRef']]
            long = gpsinfo[_GPSTAGS_r['GPSLongitude']], gpsinfo[_GPSTAGS_r['GPSLongitudeRef']]
            lat = str(lat[0][0]) + ' ' + str(lat[0][1]) + "m " + str(lat[0][1]) + 's ' + lat[1]
            long = str(long[0][0]) + ' ' + str(long[0][1]) + "m " + str(long[0][1]) + 's ' + long[1]

            meta_data = geopy.point.Point(lat + ' ' + long)

            return meta_data.format_decimal()

    print('Metadata not found!')
    return None
