"""Download Module: pure functions to retrieve data from web resources.

This module define pure utilitary functions to retrieve data from astronomical
web services, like `Legacy Survey`, `SDSS` and `S-PLUS`.
"""

from pathlib import Path
import concurrent.futures
from typing import Union, List

from tqdm import tqdm

from mergernet.services.utils import append_query_params, download_file




LEGACY_RGB_URL: str = 'https://www.legacysurvey.org/viewer/jpeg-cutout'
SLOAN_RGB_URL: str = ''




class LegacyService:
  def download_rgb(
    self,
    ra: float,
    dec: float,
    save_path: Path,
    width: float = 256,
    height: float = 256,
    pixscale: float = 0.27,
    bands: str = 'grz',
    layer: str = 'ls-dr9'
  ) -> None:
    """Downloads a single Legacy Survey object RGB stamp defined by RA and DEC.

    Parameters
    ----------
    ra: float
      Right ascension of the object.
    dec: float
      Declination of the object.
    save_path: pathlib.Path
      Path where downloaded file will be stored.
    width: float
      Stamp width.
    height: float
      Stamp height.
    pixscale: float
      Pixel scale of the sky.
    bands: str
      Image bands
    layer: str
      Legacy Survey image layer.
    """

    image_url = append_query_params(LEGACY_RGB_URL, {
      'ra': ra,
      'dec': dec,
      'width': width,
      'height': height,
      'pixscale': pixscale,
      'bands': bands,
      'layer': layer
    })
    download_file(image_url, save_path)



  def batch_download_rgb(
    self,
    ra: List[float],
    dec: List[float],
    save_path: List[Path],
    workers: Union[int, None] = None,
    **kwargs
  ) -> None:
    """Downloads a list of objects defined by RA and DEC coordinates.

    `ra`, `dec` and `save_path` lists are mandatory and must have same length.

    Parameters
    ----------
    ra: list(float)
      The list of RA coordinates of the desired objects.
    dec: list(float)
      The list of DEC coordinates of the desired objects.
    save_path: list(Path)
      The list of path where files should be saved.
    workers: int, optional
      Maximum spawned threads.
    kwargs: optional
      Same args as `download_legacy_rgb` function.
    """

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
      futures = []
      for i in range(len(ra)):
        futures.append(executor.submit(
          self.download_rgb,
          ra=ra[i],
          dec=dec[i],
          save_path=save_path[i],
          **kwargs
        ))
      for _ in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), unit=' file'):
        pass



