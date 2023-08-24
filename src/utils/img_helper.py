"""
    caGAN project image edit helper functions

Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""

import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

from .norm_helper import *
from .data_helper import *


def choose_random_images(n, imgs_paths):
    imgs = []
    titles = []
    for i in range(n):
        rand_sample = np.random.choice(np.shape(imgs_paths)[0])
        img = imread(imgs_paths[rand_sample])
        imgs.append(img)
        print(np.shape(img))
        # The title for each image
        titles.append('\n'.join(imgs_paths[rand_sample].split('/')[-2:]))
    plot_multy(imgs, n, 1, titles)
    return imgs, titles


def plot_multy(imgs, cols, rows=1, titles=None):
    # creates a figure with subplots organized in a grid pattern with the specified number of rows and columns.
    fig, ax = plt.subplots(nrows=rows, ncols=cols,
                        figsize=(20,8),
                        subplot_kw=dict(xticks=[], yticks=[]))

    if rows == 1:
        ax = np.expand_dims(ax, 0)


    cmap = 'gray' if len(imgs[0].shape) == 2 else None

    # iterate over each row and column in the grid and display an image
    # The images to be displayed are accessed from the variable x
    i = 0
    for row in range(rows):
        for col in range(cols):
            ax[row, col].imshow(imgs[i], cmap=cmap)
            if titles is None:
                ax[row, col].set_title(str(row)+str(col))
            else:
                ax[row, col].set_title(titles[i])
            i += 1

    # show the figure
    plt.show()

