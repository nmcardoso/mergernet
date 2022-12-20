from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image, ImageOps

from mergernet.core.utils import load_image

DEFAULT_BANDS_SCALES = {
  'urz': {
    'u': (2, 0.0066),
    'r': (1, 0.01),
    'z': (0, 0.025),
  },
  'gri': {
    'g': (2, 0.002),
    'r': (1, 0.004),
    'i': (0, 0.005),
  },
  'grz': {
    'g': (2, 0.0066),
    'r': (1, 0.01385),
    'z': (0, 0.025),
  },
  'ugriz': {
    'g': (2, 0.0066),
    'r': (1, 0.01),
    'i': (0, 0.05),
  }
}


class ColorImage:
  @staticmethod
  def asinh_map(x, arcsinh=1.):
    """
    Apply non-linear map to input matrix. Useful to rescale telescope
    pixels for viewing.
    Args:
      x (np.array): array to have map applied
      arcsinh (np.float):
    Returns:
      (np.array) array with map applied
    """
    return np.arcsinh(x * arcsinh)

  @staticmethod
  def asinh_map2(x, arcsinh=1):
    r"""
    Apenas uma coisinha:

    ```{math}
    :label: coisas

      Y = \frac{arcsinh(\alpha X)}{\sqrt{\alpha}}
    ```{math}
    """
    return np.arcsinh(x * arcsinh) / np.sqrt(arcsinh)

  @staticmethod
  def legacy_style(
    img: Union[str, Path, np.ndarray],
    bands: str = 'grz',
    brightness: float = None,
    scales: Dict[str, Tuple[int, float]] = None,
    desaturate: bool = False,
    minmax: Tuple[float, float] = (-3, 10), # cutoff range
    nl_func = None,
    normalize: bool = True,
    save_path: Union[str, Path] = None,
  ) -> np.ndarray:
    if not isinstance(img, np.ndarray):
      img = load_image(img)

    if nl_func is None:
      nl_func = ColorImage.asinh_map

    # first number is index of that band
    # second number is scale divisor - divide pixel values by
    # scale divisor for rgb pixel value
    if scales is None:
      scales = DEFAULT_BANDS_SCALES[bands]

    #  create blank matrix to work with
    h, w, _ = img.shape
    rgb = np.zeros((h, w, 3), np.float32)

    # Copy each band matrix into the rgb image, dividing by band
    # scale divisor to increase pixel values
    for i, band in enumerate(bands):
      plane, scale = scales[band]
      rgb[:, :, plane] = (img[:, :, i] / scale).astype(np.float32)

    if brightness is not None:
      # image rescaled by single-pixel not image-pixel,
      # which means colours depend on brightness
      rgb = nl_func(rgb, arcsinh=brightness)
      nl_minmax = nl_func(np.array(minmax), arcsinh=brightness)

    # lastly, rescale image to be between min and max
    rgb = (rgb - nl_minmax[0]) / (nl_minmax[1] - nl_minmax[0])

    # optionally desaturate pixels that are dominated by a single
    # colour to avoid colourful speckled sky
    if desaturate:
      # reshape rgb from (h, w, 3) to (3, h, w)
      # rgb = np.array([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]])
      rgb = np.moveaxis(rgb, -1 , 0)

      # a is mean pixel value across all bands, (h, w) shape
      mean_matrix = rgb.mean(axis=0)

      # set pixels with 0 mean value to mean of 1. Inplace?
      np.putmask(mean_matrix, mean_matrix == 0.0, 1.0)

      # copy mean value array (h,w) into 3 bands (3, h, w)
      mean_cube = np.resize(mean_matrix, (3, h, w))

      # bcube: divide image by mean-across-bands pixel value, and again by 2.5 (why?)
      mask = (rgb / mean_cube) / 2.5

      # maximum per pixel across bands of mean-band-normalised rescaled image
      wt = np.max(mask, axis=0)

      # i.e largest relative deviation from mean
      # clip largest allowed relative deviation to one (inplace?)
      np.putmask(wt, wt > 1.0, 1.0)

      # invert relative deviations
      wt = 1 - wt

      # non-linear rescaling of relative deviations
      wt = np.sin(wt*np.pi/2.0)

      # multiply by weights in complicated fashion
      temp = rgb*wt + mean_matrix*(1-wt) + mean_matrix*(1-wt)**2 * rgb

      # reset rgb to be blank
      desatured_rgb = np.zeros((h, w, 3), np.float32)

      # fill rgb with weight-rescaled rgb
      for idx, im in enumerate((temp[0, :, :], temp[1, :, :], temp[2, :, :])):
        desatured_rgb[:, :, idx] = im

      rgb = desatured_rgb

    # set min/max to 0 and 1
    rgb = np.clip(rgb, 0., 1.)

    # set image to RGB levels (0 - 255)
    if not normalize or save_path is not None:
      rgb = (rgb * 255).astype(np.uint8)

    # save image
    if save_path is not None:
      ImageOps.flip(Image.fromarray(rgb, 'RGB')).save(save_path)

    return rgb




  @staticmethod
  def lupton_rgb(
    imgs,
    bands='grz',
    arcsinh=1.,
    mn=0.1,
    mx=100.,
    desaturate=False,
    desaturate_factor=.01,
    nl_func=None,
  ):
    """
    Create human-interpretable rgb image from multi-band pixel data
    Follow the comments of Lupton (2004) to preserve colour during rescaling
    1) linearly scale each band to have good colour spread (subjective choice)
    2) nonlinear rescale of total intensity using arcsinh
    3) linearly scale all pixel values to lie between mn and mx
    4) clip all pixel values to lie between 0 and 1
    Optionally, desaturate pixels with low signal/noise value to avoid speckled sky (partially implemented)
    Args:
        imgs (list): of 2-dim np.arrays, each with pixel data on a band # TODO refactor to one 3-dim array
        bands (str): ordered characters of bands of the 2-dim pixel arrays in imgs
        arcsinh (float): softening factor for arcsinh rescaling
        mn (float): min pixel value to set before (0, 1) clipping
        mx (float): max pixel value to set before (0, 1) clipping
        desaturate (bool): If True, reduce saturation on low S/N pixels to avoid speckled sky
        desaturate_factor (float): parameter controlling desaturation. Proportional to saturation.
    Returns:
        (np.array) of shape (H, W, 3) of pixel values for colour image
    """
    if nl_func is None:
      nl_func = ColorImage.asinh_map

    size = imgs[0].shape[1]
    grzscales = dict(
      g=(2, 0.00526),
      r=(1, 0.008),
      z=(0, 0.0135)
    )

    # set the relative intensities of each band to be approximately equal
    img = np.zeros((size, size, 3), np.float32)
    for im, band in zip(imgs, bands):
      plane, scale = grzscales.get(band, (0, 1.))
      img[:, :, plane] = (im / scale).astype(np.float32)

    I = img.mean(axis=2, keepdims=True)

    if desaturate:
      img_nanomaggies = np.zeros((size, size, 3), np.float32)
      for im, band in zip(imgs, bands):
        plane, scale = grzscales.get(band, (0, 1.))
        img_nanomaggies[:, :, plane] = im.astype(np.float32)
      img_nanomaggies_nonzero = np.clip(img_nanomaggies, 1e-9, None)
      img_ab_mag = 22.5 - 2.5 * np.log10(img_nanomaggies_nonzero)
      img_flux = np.power(10, img_ab_mag / -2.5) * 3631
      # DR1 release paper quotes 90s exposure time per band, 900s on completion
      # TODO assume 3 exposures per band per image. exptime is per ccd, nexp per tile, will be awful to add
      exposure_time_seconds = 90. * 3.
      photon_energy = 600. * 1e-9  # TODO assume 600nm mean freq. for gri bands, can improve this
      img_photons = img_flux * exposure_time_seconds / photon_energy
      img_photons_per_pixel = np.sum(img_photons, axis=2, keepdims=True)

      mean_all_bands = img.mean(axis=2, keepdims=True)
      deviation_from_mean = img - mean_all_bands
      signal_to_noise = np.sqrt(img_photons_per_pixel)
      saturation_factor = signal_to_noise * desaturate_factor
      # if that would imply INCREASING the deviation, do nothing
      saturation_factor[saturation_factor > 1] = 1.
      img = mean_all_bands + (deviation_from_mean * saturation_factor)

    rescaling = nl_func(I, arcsinh=arcsinh)/I
    rescaled_img = img * rescaling

    rescaled_img = (rescaled_img - mn) * (mx - mn)
    rescaled_img = (rescaled_img - mn) * (mx - mn)

    return np.clip(rescaled_img, 0., 1.)





if __name__ == '__main__':
  fits_path = '/home/natan/repos/mergernet/data/images/J000030.87-011246.8.fits'
  rgb_path = '/home/natan/repos/mergernet/data/images/J000/J000030.87-011246.8.png'
  output_path = '/home/natan/repos/mergernet/data/images/out.png'


  fits_img = load_image(fits_path)
  rgb_img = load_image(rgb_path)

  print('fits shape', fits_img.shape)
  print('rgb shape', rgb_img.shape)


  kwargs = {
    'scales': {
      'g': (2, 0.008),
      'r': (1, 0.014),
      'z': (0, 0.019)
    },
    'bands': 'grz',
    'minmax': (-0.5, 300),
    'brightness': 1.3,
    'desaturate': True,
    'nl_func': ColorImage.asinh_map2
  }

  # kwargs = {
  #   'arcsinh': .3,
  #   'mn': 0,
  #   'mx': .4,
  # }
  fits_color = ColorImage.legacy_style(fits_img, **kwargs)

  print('fits_color shape', fits_color.shape)

  fits_color_int = (fits_color * 255).astype(np.uint8)
  im = Image.fromarray(fits_color_int, 'RGB')
  im = ImageOps.flip(im)
  im.save(output_path)
  # plt.imshow(fits_color)
  # plt.savefig()


  print('final fits shape', fits_color.shape)
