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
<<<<<<< HEAD


def preprocess(img):
    org_img = np.asarray(img * 255, dtype='uint8')
    z = img.reshape((-1, 3))

    # convert to np.float32
    z = np.float32(z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret, label, center = cv2.kmeans(z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    org_img = res.reshape((img.shape))

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # for ch in range(3):
    #     img = cv2.GaussianBlur(org_img[0], (5, 5), 0)
    #
    #     img = cv2.Canny(img, 254, 255, apertureSize=5)
    #
    #     kernelSize = 5
    #     kernel = np.ones((kernelSize, kernelSize), np.uint8)
    #
    #     iterations = 1
    #     img = 255 - cv2.dilate(img, kernel, iterations=iterations)
    #     org_img[ch] = img
    org_img = simplify_image_with_hough(org_img)

    return np.asarray(org_img / 255).astype(np.float32)

def simplify_image_with_hough(image, animate=True):

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve line detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    canny_img = cv2.Canny(blurred, 254, 255, apertureSize=5)

    kernelSize = 5
    kernel = np.ones((kernelSize, kernelSize), np.uint8)

    iterations = 1
    dilate_imgs = 255 - cv2.dilate(canny_img, kernel, iterations=iterations)


    tested_angles = np.arange(-90,89.5,0.5)
    num_peaks = 10
    hough_thresh = 0.05
    fill_gap_val = 200
    min_length_val = 1000

    # h, t, r = hough_line(dilate_imgs[1], theta=tested_angles)
    # p, angles, dists = hough_line_peaks(h, t, r,num_peaks=num_peaks,threshold=math.ceil(hough_thresh*max(np.ndarray.flatten(h))))
    # lines_img_sim = probabilistic_hough_line(dilate_imgs[1], threshold=math.ceil(hough_thresh*max(np.ndarray.flatten(h))), line_length=min_length_val, line_gap=fill_gap_val, theta=t)

    # Apply Hough Transform to detect lines
    lines = cv2.HoughLinesP(dilate_imgs, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=5)

    # Create a blank canvas to draw the lines on
    line_image = np.zeros_like(image)

    # Draw the detected lines on the blank canvas
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # You can adjust the color and line thickness

    # Combine the original image with the detected lines
    return cv2.addWeighted(image, 0, line_image, 1, 1)
=======
>>>>>>> caa657d852fd7421da00f14e5e025a122a9d03c0
