import cv2
import numpy as np
from scipy import ndimage


def bicubic(image, ratio):
    h, w, c = image.shape
    re_image = cv2.resize(image, (w * ratio, h * ratio), interpolation=cv2.INTER_CUBIC)

    return re_image


def interp23(image, ratio):
    image = np.transpose(image, (2, 0, 1))

    b, r, c = image.shape

    CDF23 = 2 * np.array(
        [0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0,
         -0.000060081482])
    d = CDF23[::-1]
    CDF23 = np.insert(CDF23, 0, d[:-1])
    BaseCoeff = CDF23

    first = 1
    for z in range(1, int(np.log2(ratio)) + 1):
        I1LRU = np.zeros((b, 2 ** z * r, 2 ** z * c))
        if first:
            I1LRU[:, 1:I1LRU.shape[1]:2, 1:I1LRU.shape[2]:2] = image
            first = 0
        else:
            I1LRU[:, 0:I1LRU.shape[1]:2, 0:I1LRU.shape[2]:2] = image

        for ii in range(0, b):
            t = I1LRU[ii, :, :]
            for j in range(0, t.shape[0]):
                t[j, :] = ndimage.correlate(t[j, :], BaseCoeff, mode='wrap')
            for k in range(0, t.shape[1]):
                t[:, k] = ndimage.correlate(t[:, k], BaseCoeff, mode='wrap')
            I1LRU[ii, :, :] = t
        image = I1LRU

    re_image = np.transpose(I1LRU, (1, 2, 0))

    return re_image
