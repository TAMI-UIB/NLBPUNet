"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
"""

import numpy as np

from .utils.upsamplings import bicubic


def Bicubic(pan, hs, **kwargs):
    M, N, c = pan.shape
    m, n, C = hs.shape

    ratio = int(np.round(M / m))

    print('get sharpening ratio: ', ratio)
    assert int(np.round(M / m)) == int(np.round(N / n))

    # jsut upsample with bicubic
    I_Bicubic = bicubic(hs, ratio)

    # adjustment
    I_Bicubic[I_Bicubic < 0] = 0
    I_Bicubic[I_Bicubic > 1] = 1

    return dict(fused=I_Bicubic)
