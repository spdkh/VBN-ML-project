"""
    Path helper functions
"""
import os
import glob


import geopy.point
from PIL.ExifTags import TAGS, GPSTAGS
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from src.utils import norm_helper


# pylint: disable=W0212
def check_folder(log_dir):
    """
        check if directory does not exist,
        make it.

        params:

            log_dir: str
                directory to check
    """
    print('\n[DATA Helper] Checking Folder')
    if os.path.exists(log_dir):
        print('\t', log_dir, 'Folder Exists.')
        return True
    print('\tCreating Folder', log_dir)
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
            list of directories of all files with
            given extention in the traverse directory
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

    image_batch = {}
    for i, path in enumerate(imgs_paths):
        cur_img = imread(imgs_paths[i])
        image_batch[path] = cur_img.copy()
    return image_batch


small = (615, 515)
one_k = (1024, 768)
two_k = (2048, 1536)


def imread(img_path, shape=small):
    """
        Read images with specified dimension and normalization
    :param img_path: string
    :param shape: 2D tuple of int
    :return:
    """
    img = Image.open(img_path)
    img = img.resize(shape)
    img = norm_helper.min_max_norm(np.asarray(img))
    return img


def metadata_read(img_path):
    """
        Read metadata embedded in JPG file
    :param img_path:
    :return:
    """
    img = Image.open(img_path)

    if 'exif' in img.info.keys():

        # build reverse dicts
        _tags_r = dict(((i, j) for j, i in TAGS.items()))
        _gpstags_r = dict(((i, j) for j, i in GPSTAGS.items()))

        # this merges gpsinfo as data rather than an offset pointer
        exifd = img._getexif()
        if "GPSInfo" in _tags_r.keys():
            gpsinfo = exifd[_tags_r["GPSInfo"]]

            lat = gpsinfo[_gpstags_r['GPSLatitude']],\
                  gpsinfo[_gpstags_r['GPSLatitudeRef']]
            long = gpsinfo[_gpstags_r['GPSLongitude']],\
                   gpsinfo[_gpstags_r['GPSLongitudeRef']]
            lat = str(lat[0][0]) + ' ' + str(lat[0][1]) + "m " \
                  + str(lat[0][1]) + 's ' + lat[1]
            long = str(long[0][0]) + ' ' + str(long[0][1]) + "m " \
                   + str(long[0][1]) + 's ' + long[1]

            meta_data = geopy.point.Point(lat + ' ' + long)

            return meta_data.format_decimal()

    print('Metadata not found!')
    return None


def visualize_predict(img, predicted_info, output_dir, gt_info='NA', error='NA'):
    """
        Visualize predicted images
    :param img:
    :param predicted_info:
    :param output_dir:
    :param gt_info:
    :param error:
    :return:
    """
    plt.figure()
    # figures equal to the number of z patches in columns

    plt.title('original lat/long = ' \
              + gt_info \
              + '\nPredicted lat/long =' \
              + predicted_info)

    plt.imshow(img)
    plt.show()

    plt.gca().axes.yaxis.set_ticklabels([])
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticks([])
    plt.gca().axes.xaxis.set_ticks([])
    plt.xlabel('\nError ='
               + error)

    plt.savefig(output_dir)  # Save sample results
    plt.close("all")  # Close figures to avoid memory leak
