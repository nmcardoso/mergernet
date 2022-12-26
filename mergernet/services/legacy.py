from pathlib import Path
from typing import List, Tuple, Union

import requests

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
    self.http_client = requests.Session()


  def cutout(
    self,
    ra: float,
    dec: float,
    save_path: Path,
  ) -> None:
    """
    Downloads a single Legacy Survey object RGB stamp defined by RA and DEC.

    Parameters
    ----------
    ra: float
      Right ascension of the object.
    dec: float
      Declination of the object.
    save_path: Path
      Path where downloaded file will be stored.
    """
    if self.image_format == 'jpg':
      url = LEGACY_RGB_URL_DEV if self.use_dev else LEGACY_RGB_URL
    else:
      url = LEGACY_FITS_URL_DEV if self.use_dev else LEGACY_FITS_URL

    image_url = append_query_params(url, {
      'ra': ra,
      'dec': dec,
      'width': self.width,
      'height': self.height,
      'pixscale': self.pixscale,
      'bands': self.bands,
      'layer': self.layer
    })

    download_file(
      url=image_url,
      save_path=save_path,
      http_client=self.http_client,
      replace=self.replace
    )



  def batch_cutout(
    self,
    ra: List[float],
    dec: List[float],
    save_path: List[Path],
  ) -> Tuple[List[Path], List[Path]]:
    """
    Downloads a list of objects defined by RA and DEC coordinates.

    The ``ra``, ``dec`` and ``save_path`` lists are mandatory and
    must have same length.

    Parameters
    ----------
    ra: list of float
      The list of RA coordinates of the desired objects.
    dec: list of float
      The list of DEC coordinates of the desired objects.
    save_path: list of Path
      The list of path where files should be saved.
    """
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
      unit=' files'
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
