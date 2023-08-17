from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tqdm import tqdm

from mergernet.core.utils import (extract_iauname_from_path, load_image,
                                  save_image)

# first number is index of that band
# second number is scale divisor - divide pixel values by
# scale divisor for rgb pixel value
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


def asinh_map(x: np.ndarray, gain: float = 1.) -> np.ndarray:
  r"""
  Apply the following non-linear map to input matrix. Useful to
  rescale telescope pixels for viewing.

  .. math::
    Y = \textrm{arcsinh}(\alpha X)

  where :math:`\alpha` is the gain, :math:`X` is the input image and
  :math:`Y` is the transformed image.

  Parameters
  ----------
  x: numpy.ndarray
    image to have map applied
  gain: float, optional
    gain applied to input image

  Returns
  -------
  numpy.ndarray
    transformed image
  """
  return np.arcsinh(x * gain)



def asinh_map2(x: np.ndarray, gain: float = 1) -> np.ndarray:
  r"""
  Apply the following non-linear map to input matrix. Useful to
  rescale telescope pixels for viewing.

  .. math::
    Y = \frac{\textrm{arcsinh}(\alpha X)}{\sqrt{\alpha}}

  where :math:`\alpha` is the gain, :math:`X` is the input image and
  :math:`Y` is the transformed image.

  Parameters
  ----------
  x: numpy.ndarray
    image to have map applied
  gain: float, optional
    gain applied to input image

  Returns
  -------
  numpy.ndarray
    transformed image
  """
  return np.arcsinh(x * gain) / np.sqrt(gain)



class ImageTransform(ABC):
  @abstractmethod
  def transform(self, image: np.ndarray) -> np.ndarray:
    pass


  def on_batch_end(self):
    pass


  def batch_transform(
    self,
    images: List[Path],
    save_paths: List[Path] = None,
    silent: bool = False,
  ):
    errors = []

    if save_paths is None:
      for image_path in tqdm(images, total=len(images), unit='file'):
        try:
          self.transform(load_image(image_path))
        except KeyboardInterrupt:
          break
        except:
          errors.append(extract_iauname_from_path(image_path))
    else:
      for image_path, save_path in tqdm(
        zip(images, save_paths), total=len(images), unit='file'
      ):
        try:
          transformed_image = self.transform(load_image(image_path))
          save_image(transformed_image, save_path)
        except KeyboardInterrupt:
          break
        except:
          errors.append(extract_iauname_from_path(image_path))

    self.on_batch_end()

    if not silent and len(errors) > 0:
      print('Failed images:')
      for error in errors:
        print(error)

    return errors




class ImagePipeline(ImageTransform):
  def __init__(self, transforms: List[ImageTransform] = []):
    self._transforms = transforms


  def transform(self, image: np.ndarray) -> np.ndarray:
    i = image.copy()
    for transform in self._transforms:
      i = transform.transform(i)
    return i


  def on_batch_end(self):
    for transform in self._transforms:
      transform.on_batch_end()



class ChannelAverage(ImageTransform):
  def __init__(self, return_int=False, normalize=False):
    self.return_int = return_int
    self.normalize = normalize


  def transform(self, image: np.ndarray) -> np.ndarray:
    channel_index = np.argmin(image.shape)
    mean = np.mean(image, axis=channel_index)
    if self.normalize:
      mean = mean / np.max(mean)
    if self.return_int:
      mean = (mean * 255).astype(np.uint8)
    return mean



class LegacyRGB(ImageTransform):
  def __init__(
    self,
    bands: str = 'grz',
    brightness: float = None,
    scales: Dict[str, Tuple[int, float]] = None,
    desaturate: bool = False,
    minmax: Tuple[float, float] = (-3, 10), # cutoff range
    nl_func: str = 'asinh',
    rgb_output: bool = True,
  ):
    self.bands = bands
    self.brightness = brightness
    self.scales = scales or DEFAULT_BANDS_SCALES[self.bands]
    self.desaturate = desaturate
    self.minmax = minmax
    self.nl_func = asinh_map if nl_func == 'asinh' else asinh_map2
    self.rgb_output = rgb_output


  def transform(self, img: np.ndarray) -> np.ndarray:
    #  create blank matrix to work with
    h, w, _ = img.shape
    rgb = np.zeros((h, w, 3), np.float32)

    # Copy each band matrix into the rgb image, dividing by band
    # scale divisor to increase pixel values
    for i, band in enumerate(self.bands):
      plane, scale = self.scales[band]
      rgb[:, :, plane] = (img[:, :, i] / scale).astype(np.float32)

    if self.brightness is not None:
      # image rescaled by single-pixel not image-pixel,
      # which means colours depend on brightness
      rgb = self.nl_func(rgb, gain=self.brightness)
      nl_minmax = self.nl_func(np.array(self.minmax), gain=self.brightness)

    # lastly, rescale image to be between min and max
    rgb = (rgb - nl_minmax[0]) / (nl_minmax[1] - nl_minmax[0])

    # optionally desaturate pixels that are dominated by a single
    # colour to avoid colourful speckled sky
    if self.desaturate:
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
    if self.rgb_output:
      rgb = (rgb * 255).astype(np.uint8)

    # save image
    # if save_path is not None:
    #   ImageOps.flip(Image.fromarray(rgb, 'RGB')).save(save_path)

    return rgb



class LuptonRGB(ImageTransform):
  def __init__(
    self,
    bands='grz',
    arcsinh=1.,
    mn=0.1,
    mx=100.,
    desaturate=False,
    desaturate_factor=.01,
    nl_func: str = 'asinh',
  ):
    self.bands = bands
    self.arcsinh = arcsinh
    self.mn = mn
    self.mx = mx
    self.desaturate = desaturate
    self.desaturate_factor = desaturate_factor
    self.nl_func = asinh_map if nl_func == 'asinh' else asinh_map2


  def transform(self, imgs:  np.ndarray) -> np.ndarray:
    """
    Create human-interpretable rgb image from multi-band pixel data
    Follow the comments of Lupton (2004) to preserve colour during rescaling
    1) linearly scale each band to have good colour spread (subjective choice)
    2) nonlinear rescale of total intensity using arcsinh
    3) linearly scale all pixel values to lie between mn and mx
    4) clip all pixel values to lie between 0 and 1
    Optionally, desaturate pixels with low signal/noise value to avoid speckled sky (partially implemented)

    Parameters
    ----------
    imgs: numpy.ndarray
      an array with shape (h, w, c) which represents the image,
      each with pixel data on a band
    bands: str, optional
      ordered characters of bands of the 2-dim pixel arrays in imgs
    arcsinh: float, optional
      softening factor for arcsinh rescaling
    mn: float, optional
      min pixel value to set before (0, 1) clipping
    mx: float, optional
      max pixel value to set before (0, 1) clipping
    desaturate: bool, optional
      If True, reduce saturation on low S/N pixels to avoid speckled sky
    desaturate_factor: float, optional
      parameter controlling desaturation. Proportional to saturation.

    Returns
    -------
    numpy.ndarray
      aray of shape (H, W, 3) of pixel values for colour image
    """
    size = imgs[0].shape[1]
    grzscales = dict(
      g=(2, 0.00526),
      r=(1, 0.008),
      z=(0, 0.0135)
    )

    # set the relative intensities of each band to be approximately equal
    img = np.zeros((size, size, 3), np.float32)
    for im, band in zip(imgs, self.bands):
      plane, scale = grzscales.get(band, (0, 1.))
      img[:, :, plane] = (im / scale).astype(np.float32)

    I = img.mean(axis=2, keepdims=True)

    if self.desaturate:
      img_nanomaggies = np.zeros((size, size, 3), np.float32)

      for im, band in zip(imgs, self.bands):
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
      saturation_factor = signal_to_noise * self.desaturate_factor

      # if that would imply INCREASING the deviation, do nothing
      saturation_factor[saturation_factor > 1] = 1.
      img = mean_all_bands + (deviation_from_mean * saturation_factor)

    rescaling = self.nl_func(I, gain=self.arcsinh) / I
    rescaled_img = img * rescaling

    rescaled_img = (rescaled_img - self.mn) * (self.mx - self.mn)
    rescaled_img = (rescaled_img - self.mn) * (self.mx - self.mn)

    return np.clip(rescaled_img, 0., 1.)



class Crop(ImageTransform):
  def __init__(self, size: int):
    self.size = size


  def transform(self, image: np.ndarray) -> np.ndarray:
    img_size = image.shape[1]
    start = (img_size - self.size) // 2
    end = start + self.size

    if len(image.shape) == 2:
      return image[start:end, start:end]
    else:
      return image[start:end, start:end, :]



class TensorToImage(ImageTransform):
  def __init__(self, save_paths: List[Union[str, Path]]):
    self.save_paths = [Path(p) for p in save_paths]
    self.current_index = 0


  def transform(self, image: np.ndarray):
    save_image(image, self.save_paths[self.current_index])
    self.current_index += 1



class TensorToShards(ImageTransform):
  # https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
  def __init__(
    self,
    save_path: Union[str, Path],
    examples_per_shard: int = 1024,
  ):
    self.save_path = Path(save_path)
    self.examples_per_shard = examples_per_shard
    self._writer = None
    self._examples_count = 0
    self._shard_count = 0
    self.save_path.mkdir(parents=True, exist_ok=True)


  def transform(self, image: np.ndarray):
    self._create_shard()

    data = {'X': self._bytes_feature(self._serialize_array(image))}
    example = tf.train.Example(features=tf.train.Features(feature=data))

    self._writer.write(example.SerializeToString())
    self._examples_count += 1


  def on_batch_end(self):
    self._writer.close()
    self._shard_count += 1


  def _create_shard(self):
    if self._examples_count >= self.examples_per_shard:
      self._writer.close()
      self._writer = None
      self._shard_count += 1
      self._examples_count = 0

    if self._writer is None:
      path = self.save_path / f'shard-{self._shard_count:05d}.tfrecords'
      self._writer = tf.io.TFRecordWriter(str(path.resolve()))


  def _bytes_feature(self, value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
      value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


  def _serialize_array(self, array):
    return tf.io.serialize_tensor(array)




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
    'nl_func': 'asinh'
  }

  # kwargs = {
  #   'arcsinh': .3,
  #   'mn': 0,
  #   'mx': .4,
  # }
  lrgb = LegacyRGB(**kwargs)
  fits_color = lrgb.transform(fits_img)

  print('fits_color shape', fits_color.shape)

  fits_color_int = (fits_color * 255).astype(np.uint8)
  im = Image.fromarray(fits_color_int, 'RGB')
  im = ImageOps.flip(im)
  im.save(output_path)
  # plt.imshow(fits_color)
  # plt.savefig()


  print('final fits shape', fits_color.shape)
