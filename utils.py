import torch
import matplotlib.colors as colors
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.autograd.variable import Variable


def from_numpy_to_var(npx, dtype='float32'):
    var = Variable(torch.from_numpy(npx.astype(dtype)))
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var


def from_tensor_to_var(tensor):
    var = Variable(tensor)
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var


def normalize_row(a):
    row_sums = a.sum(axis=1)
    return a / row_sums[:, np.newaxis]


def print_array(array):
    a = ["%.2f" % i for i in array]
    return ", ".join(a)


def write_on_image(img, text):
    """
    Make sure to write to final images - not fed into a generator.
    :param img: W x H x channel size
    :param text: string
    :return: write text on image.
    """
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (2, 7)
    fontScale = 0.25
    fontColor = (1, 1, 1)
    lineType = 0
    cv2.putText(img,
                text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)


def write_number_on_images(imgs, texts):
    """
    :param imgs: (numpy array) n x m x channel size x W x H
    :param texts: (numpy array) n x m
    :return: write texts on images.
    """
    n, m = texts.shape
    for i in range(n):
        for j in range(m):
            img = imgs[i, j]
            text = texts[i, j]
            trans_img = np.transpose(img, (1, 2, 0)).copy()
            write_on_image(trans_img, "%.2f" % text)
            imgs[i, j] = np.transpose(trans_img, (2, 0, 1))


def write_stats_from_var(log_dict, torch_var, name, idx=None):
    if idx is None:
        # log_dict['%s_mean' % name] = torch_var.data.mean()
        # log_dict['%s_std' % name] = torch_var.data.std()
        # log_dict['%s_max' % name] = torch_var.data.max()
        # log_dict['%s_min' % name] = torch_var.data.min()
        np_var = torch_var.data.cpu().numpy()
        for i in [0, 25, 50, 75, 100]:
            log_dict['%s_%d' % (name, i)] = np.percentile(np_var, i)
    else:
        assert type(idx) == int
        assert len(torch_var.size()) == 2
        write_stats_from_var(log_dict, torch_var[:, idx], '%d_%s' % (idx, name))


def plot_img(img, path, vrange=None, title=None):
    if title is not None:
        plt.title(title)
    if vrange is None:
        vrange = (0, 1)
    plt.imshow(img, vmin=vrange[0], vmax=vrange[1], cmap='gray', aspect='auto')
    plt.savefig(path)
    plt.close()