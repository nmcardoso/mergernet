from pathlib import Path
from typing import List, Tuple, Union

import requests

from mergernet.services.utils import (append_query_params, download_file,
                                      parallel_function_executor)




class LegacyService:
  def __init__(self):
    self.http_client = requests.Session()


  def download_rgb(
    self,
    ra: float,
    dec: float,
    save_path: Path,
    replace: bool = False,
    width: float = 256,
    height: float = 256,
    pixscale: float = 0.27,
    bands: str = 'grz',
    layer: str = 'ls-dr9'
  ) -> None:
    """
    Downloads a single Legacy Survey object RGB stamp defined by RA and DEC.

    Parameters
    ----------
    ra: float
      Right ascension of the object.
    dec: float
      Declination of the object.
    save_path: pathlib.Path
      Path where downloaded file will be stored.
    replace: bool (optional)
      Replace file if exists in ``save_path`` location
    width: float (optional)
      Stamp width.
    height: float (optional)
      Stamp height.
    pixscale: float (optional)
      Pixel scale of the sky.
    bands: str (optional)
      Image bands
    layer: str (optional)
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

    download_file(
      url=image_url,
      save_path=save_path,
      http_client=self.http_client,
      replace=replace
    )



  def batch_download_rgb(
    self,
    ra: List[float],
    dec: List[float],
    save_path: List[Path],
    workers: Union[int, None] = None,
    replace: bool = False,
    **kwargs
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
    workers: int, optional
      Maximum spawned threads.
    kwargs: optional
      Same args as ``download_legacy_rgb`` function.
    """

    urls = [
      append_query_params(LEGACY_RGB_URL, {'ra': _ra, 'dec': _dec, **kwargs})
      for _ra, _dec in zip(ra, dec)
    ]
    batch_download_file(
      urls=urls,
      save_path=save_path,
      workers=workers,
      replace=replace
    )



if __name__ == '__main__':
  ls = LegacyService()
  ls.batch_download_legacy_rgb(
    ra=[185.1458 + dx/2 for dx in range(20)],
    dec=[12.8624 + dy/2 for dy in range(20)],
    save_path=[Path(f'broca/{i}.jpg') for i in range(20)],
    workers=6
  )
  # ls.download_legacy_rgb(185.1458, 12.8624, Path('stamps/broca.jpg'))
  # ls._download_file('https://natanael.net', Path('broca/broca.html'))
  # print(ls._append_query_params('http://natanael.net?key0=val0', {'key1': 'val1', 'key2': 'val2'}))
