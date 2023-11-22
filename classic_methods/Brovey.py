import numpy as np

from .utils.upsamplings import interp23


def Brovey(pan, hs, **kwargs):
    M, N, c = pan.shape
    m, n, C = hs.shape

    ratio = int(np.round(M / m))

    print('get sharpening ratio: ', ratio)
    assert int(np.round(M / m)) == int(np.round(N / n))

    # upsample
    u_hs = interp23(hs, ratio)

    I = np.mean(u_hs, axis=-1)

    image_hr = (pan - np.mean(pan)) * (np.std(I, ddof=1) / np.std(pan, ddof=1)) + np.mean(I)
    image_hr = np.squeeze(image_hr)

    I_Brovey = []
    for i in range(C):
        temp = image_hr * u_hs[:, :, i] / (I + 1e-8)
        temp = np.expand_dims(temp, axis=-1)
        I_Brovey.append(temp)

    I_Brovey = np.concatenate(I_Brovey, axis=-1)

    # adjustment
    I_Brovey[I_Brovey < 0] = 0
    I_Brovey[I_Brovey > 1] = 1

    return dict(fused=I_Brovey)
