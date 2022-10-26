import numpy as np

class MCIE(object):
    def __init__(self, _limit = 0.02):
        self._limit = _limit
        
    def equalize(self, img):
        h, w = img.shape
        num_pixels = h * w
        # get hist
        hist, _ = np.histogram(img.flatten(), 256, [0, 256])
        limit_pixels = int(num_pixels * self._limit)
        num_overflow = np.sum(np.clip(hist - limit_pixels, a_min=0, a_max=None))
        hist = np.clip(hist, a_min=0, a_max=limit_pixels)
        # add
        hist += np.round(num_overflow / 256.0).astype(np.int)
        # get cdf
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype(np.uint8)
        return cdf

    def __call__(self, img):
        """_summary_
        Args:
            image: HxWxC
        """
        equal_img = [self.equalize(img[:, :, i])[img[:, :, i]] for i in range(3)]
        return np.stack(equal_img, axis=-1)