from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Union

import requests
from astropy.io import fits

from mergernet.core.constants import RANDOM_SEED
from mergernet.core.utils import compress_fits, iauname, iauname_relative_path
from mergernet.services.imaging import ImagingService
from mergernet.services.utils import (append_query_params, download_file,
                                      parallel_function_executor)

LEGACY_RGB_URL = 'https://www.legacysurvey.org/viewer/jpeg-cutout'
LEGACY_RGB_URL_DEV = 'https://www.legacysurvey.org/viewer-dev/jpeg-cutout'
LEGACY_FITS_URL = 'https://www.legacysurvey.org/viewer/fits-cutout'
LEGACY_FITS_URL_DEV = 'https://www.legacysurvey.org/viewer-dev/fits-cutout'



class LegacyService(ImagingService):
  def __init__(
    self,
    replace: bool = False,
    width: float = 256,
    height: float = 256,
    pixscale: float = 0.27,
    bands: str = 'grz',
    layer: str = 'ls-dr9',
    use_dev: bool = False,
    fmt: str = 'jpg',
    compress_fits: bool = False,
    compress_type: str = 'HCOMPRESS_1',
    hcomp_scale: int = 3,
    quantize_level: int = 10,
    quantize_method: int = -1,
    workers: int = 3,
  ):
    """
    Parameters
    ----------
    replace: bool, optional
      Replace file if exists in ``save_path`` location
    width: float, optional
      Stamp width.
    height: float, optional
      Stamp height.
    pixscale: float, optional
      Pixel scale of the sky.
    bands: str, optional
      Image bands
    layer: str, optional
      Legacy Survey image layer.
    use_dev: bool, optional
      Use the dev env of Legacy Cutout API
    fmt: str, optional
      File format. One of: ``jpg`` or ``fits``
    compress_fits: bool, optional
      Compress the downloaded fits stamp to ``fits.fz``
    workers: int, optional
      Maximum spawned threads when `batch_cutout` is called
    """
    super().__init__(fmt)
    self.replace = replace
    self.width = width
    self.height = height
    self.pixscale = pixscale
    self.bands = bands
    self.layer = layer
    self.use_dev = use_dev
    self.workers = workers
    self.compress_fits = compress_fits
    self.compress_type = compress_type
    self.hcomp_scale = hcomp_scale
    self.quantize_level = quantize_level
    self.quantize_method = quantize_method
    self.http_client = requests.Session()


  def cutout(
    self,
    ra: float,
    dec: float,
    save_path: Path = None,
    base_path: Union[str, Path] = '',
  ) -> None:
    """
    Downloads a single Legacy Survey object RGB stamp defined by RA and DEC.

    Parameters
    ----------
    ra: float
      Right ascension of the object.
    dec: float
      Declination of the object.
    save_path: Path, optional
      Path where downloaded file will be stored.
    base_path: str, Path, optional
      The path that will be appended at beggining of every paths if ``save_path``
      is ``None``.
    """
    if self.image_format == 'jpg':
      url = LEGACY_RGB_URL_DEV if self.use_dev else LEGACY_RGB_URL
    else:
      url = LEGACY_FITS_URL_DEV if self.use_dev else LEGACY_FITS_URL

    if save_path is None:
      save_path = iauname_relative_path(
        iaunames=iauname(ra=ra, dec=dec),
        prefix=Path(base_path),
        suffix=f'.{self.image_format}'
      )

    image_url = append_query_params(url, {
      'ra': ra,
      'dec': dec,
      'width': self.width,
      'height': self.height,
      'pixscale': self.pixscale,
      'bands': self.bands,
      'layer': self.layer
    })

    content = download_file(
      url=image_url,
      save_path=save_path,
      http_client=self.http_client,
      replace=self.replace,
      return_bytes=self.compress_fits,
    )

    if self.compress_fits:
      compress_fits(
        file=BytesIO(content),
        compress_type=self.compress_type,
        hcomp_scale=self.hcomp_scale,
        quantize_level=self.quantize_level,
        quantize_method=self.quantize_method,
        ext=0,
        save_path=save_path,
        replace=self.replace,
      )


  def batch_cutout(
    self,
    ra: List[float],
    dec: List[float],
    save_path: List[Path] = None,
    base_path: Union[str, Path] = '',
  ) -> Tuple[List[Path], List[Path]]:
    """
    Downloads a list of objects defined by RA and DEC coordinates.

    The ``ra``, ``dec`` and ``save_path`` lists are mandatory and
    must have same length.

    Parameters
    ----------
    ra: List[float]
      The list of RA coordinates of the desired objects.
    dec: List[float]
      The list of DEC coordinates of the desired objects.
    save_path: List[Path], optional
      The list of path where files should be saved.
    base_path: str, Path, optional
      The path that will be appended at beggining of every paths if ``save_path``
      is ``None``.
    """
    if save_path is None:
      save_path = iauname_relative_path(
        iaunames=iauname(ra=ra, dec=dec),
        prefix=Path(base_path),
        suffix=f'.{self.image_format}'
      )

    params = [
      {
        'ra': _ra,
        'dec': _dec,
        'save_path': _save_path,
      }
      for _ra, _dec, _save_path in zip(ra, dec, save_path)
    ]

    parallel_function_executor(
      self.cutout,
      params=params,
      workers=self.workers,
      unit='file'
    )

    success = [p for p in save_path if p.exists()]
    error = [p for p in save_path if not p.exists()]
    return success, error



if __name__ == '__main__':
  ls = LegacyService(workers=6)
  ls.batch_download_legacy_rgb(
    ra=[185.1458 + dx/2 for dx in range(20)],
    dec=[12.8624 + dy/2 for dy in range(20)],
    save_path=[Path(f'test/{i}.jpg') for i in range(20)]
  )
