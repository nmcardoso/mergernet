from pathlib import Path
from typing import List, Sequence, Tuple, Union
import concurrent.futures

import numpy as np
import tqdm

from mergernet.core.utils import array_fallback
from mergernet.services.utils import append_query_params, batch_download_file, download_file


SDSS_RGB_URL = 'http://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg'



class SloanService:
  def download_rgb(
    self,
    ra: float,
    dec: float,
    save_path: Path,
    width: int = 256,
    height: int = 256,
    scale: float = 0.55,
    opt: str = ''
  ):
    image_url = append_query_params(SDSS_RGB_URL, {
      'ra': ra,
      'dec': dec,
      'width': width,
      'height': height,
      'scale': scale,
      'opt': opt
    })
    download_file(image_url, save_path)


  def batch_download_rgb(
    self,
    ra: List[float],
    dec: List[float],
    save_path: List[Union[str, Path]],
    workers: int = None,
    replace: bool = False,
    **kwargs
  ):
    urls = [
      append_query_params(SDSS_RGB_URL, {'ra': _ra, 'dec': _dec, **kwargs})
      for _ra, _dec in zip(ra, dec)
    ]
    batch_download_file(
      urls=urls,
      save_path=save_path,
      workers=workers,
      replace=replace
    )


  def get_dr_prefix(self, dr17: Sequence, dr7: Sequence):
    flag = False
    fnames = []

    for i in range(len(dr17)):
      if dr17[i] == 0:
        if dr7[i] != 0:
          fnames.append(f'dr7_{dr7[i]}')
        else:
          flag = True
      else:
        fnames.append(f'dr17_{dr17[i]}')

    return fnames, flag
