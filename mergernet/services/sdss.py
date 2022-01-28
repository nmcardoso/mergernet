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


  def get_image_filename(
    self,
    dr8objid: np.ndarray,
    dr7objid: np.ndarray,
    extension: str = '.jpg',
    basepath: Union[str, Path] = None
  ) -> Tuple[list, bool]:
    fnames, flag = array_fallback(arrays=(dr7objid, dr8objid), prefix=('dr7', 'dr8'))
    if basepath:
      fnames = [Path(basepath) / f'{f}{extension}' for f in fnames]
    else:
      fnames = [Path(f'{f}{extension}') for f in fnames]
    return fnames, flag
