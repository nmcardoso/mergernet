import concurrent.futures
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import tqdm

from mergernet.services.utils import append_query_params, download_file


class ImagingService(ABC):
  def __init__(self, image_format):
    self.image_format = image_format


  @abstractmethod
  def cutout(self, ra: float, dec: float, save_path: Path):
    pass


  @abstractmethod
  def batch_cutout(self, ra: List[float], dec: List[float], save_path: List[Path]):
    pass


  def _download_rgb(
    self,
    ra: float,
    dec: float,
    save_path: Path,
    width: int = 256,
    height: int = 256,
    scale: float = 0.55,
    opt: str = ''
  ):
    image_url = append_query_params(self.imaging_url, {
      'ra': ra,
      'dec': dec,
      'width': width,
      'height': height,
      'scale': scale,
      'opt': opt
    })
    download_file(image_url, save_path)


  def _batch_download_rgb(
    self,
    ra: List[float],
    dec: List[float],
    save_path: List[Path],
    workers: int = None,
    **kwargs
  ):
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

      for _ in tqdm.tqdm(
        concurrent.futures.as_completed(futures),
        total=len(futures),
        unit=' file'
      ):
        pass
