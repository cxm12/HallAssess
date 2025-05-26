import tifffile as tiff
from tifffile import imread
import torch.utils.data as data
import cv2
import torch
import os
import numpy as np


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    # print('minmax: ', mi, ma)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)
    
    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)
    #print('normalize_mi_ma_debug: ', mi, ma-mi)
    if clip:
        x = np.clip(x, 0, 1)
    return x


datamin, datamax = 0, 100  #


def np2Tensor(*args):
    def _np2Tensor(img):
        # print('np2Tensor img.shape', img.shape)
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        return tensor
    
    return [_np2Tensor(a) for a in args]



class Deg_SR(data.Dataset):
    def __init__(self, srpath='', lrpath=''):
        self.dir_demo = srpath
        self.dir_demoLR = lrpath
        self.rgb_range = 1

    def __getitem__(self, idx):
        lr = tiff.imread(self.dir_demoLR)
        if '.png' in self.dir_demo:
            sr = cv2.cvtColor(np.squeeze(cv2.imread(self.dir_demo)), cv2.COLOR_BGR2GRAY)
        else:
            sr = tiff.imread(self.dir_demo)
        h, w = lr.shape
        lr = lr[0:h//8*8, 0:w//8*8]
        lr = np.expand_dims(lr, -1)
        filename = os.path.basename(self.dir_demo)
        
        hrreshape = cv2.resize(sr, (lr.shape[0], lr.shape[1]), interpolation=cv2.INTER_CUBIC)
        hrreshape = normalize(hrreshape, datamin, datamax, clip=True) * self.rgb_range
        lr = normalize(lr, datamin, datamax, clip=True) * self.rgb_range
        if len(hrreshape.shape) == 2:
            hrreshape = np.expand_dims(hrreshape, -1)
        pair = (lr, hrreshape)
                
        pair_t = np2Tensor(*pair)
        return pair_t[0], pair_t[1], filename

    def __len__(self):
        return 1


class Deg_Flourescenedenoise(data.Dataset):
    def __init__(self, denoisepath='', noisepath=''):     
        self.datamin, self.datamax = 0, 100
        self.denoisepath = denoisepath
        self.noisepath = noisepath
        self.rgb_range = 1

    def __getitem__(self, idx):
        filename, fmt = os.path.splitext(os.path.basename(self.denoisepath))
        rgbN = np.float32(imread(self.noisepath))       
        rgbDe = np.float32(imread(self.denoisepath))
        
        rgbN = torch.from_numpy(np.ascontiguousarray(rgbN * self.rgb_range)).float()
        rgbDe = torch.from_numpy(np.ascontiguousarray(rgbDe * self.rgb_range)).float()
        return rgbN, rgbDe, filename
    
    def __len__(self):
        return 1


# Inheritted from CARE
class PercentileNormalizer(object):
    def __init__(self, pmin=2, pmax=99.8, do_after=True, dtype=torch.float32, **kwargs):
        if not (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100):
            raise ValueError
        self.pmin = pmin
        self.pmax = pmax
        self._do_after = do_after
        self.dtype = dtype
        self.kwargs = kwargs
    
    def before(self, img, axes):
        if len(axes) != img.ndim:
            raise ValueError
        channel = None if axes.find('C') == -1 else axes.find('C')
        axes = None if channel is None else tuple((d for d in range(img.ndim) if d != channel))
        self.mi = np.percentile(img.detach().cpu().numpy(), self.pmin, axis=axes, keepdims=True).astype(np.float32, copy=False)
        self.ma = np.percentile(img.detach().cpu().numpy(), self.pmax, axis=axes, keepdims=True).astype(np.float32, copy=False)
        return (img - self.mi) / (self.ma - self.mi + 1e-20)
    
    def after(self, img):
        if not self.do_after():
            raise ValueError
        alpha = self.ma - self.mi
        beta = self.mi
        return (alpha * img + beta).astype(np.float32, copy=False)
    
    def do_after(self):
        return self._do_after
