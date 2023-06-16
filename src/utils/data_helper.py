"""
    Path helper functions
"""
import os
import glob

from PIL import Image
import numpy as np
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
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


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

    image_batch = []
    for i, _ in enumerate(imgs_paths):
        cur_img = imread(imgs_paths[i])
        cur_img = norm_helper.min_max_norm(np.array(cur_img))
        image_batch.append(cur_img)

    image_batch = np.array(image_batch)

    return image_batch


one_k = (1024, 768)
two_k = (2048, 1536)


def imread(img_path, shape=one_k):
    im = Image.open(img_path)
    im.draft('RGB', shape)
    return np.asarray(im)
