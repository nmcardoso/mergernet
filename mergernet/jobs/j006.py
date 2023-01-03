"""
[DATA] Decals 1M-dataset fits compression
"""

import pandas as pd
from tqdm import tqdm

from mergernet.core.constants import DATA_ROOT
from mergernet.core.experiment import Experiment
from mergernet.core.utils import compress_fits, iauname, iauname_relative_path
from mergernet.services.legacy import LegacyService


class Job(Experiment):
  """
  Compress fits stamps of decals 1M-dataset
  """

  def __init__(self):
    super().__init__()
    self.exp_id = 6
    self.log_wandb = False
    self.restart = False


  def call(self):
    Experiment.download_file_gd('decals_pca10.csv', exp_id=3)

    df = pd.read_csv(Experiment.local_exp_path / 'decals_pca10.csv')[:300_000]

    iaunames = iauname(ra=df.ra.values, dec=df.dec.values)

    descompressed_paths = iauname_relative_path(
      iaunames=iaunames,
      prefix=DATA_ROOT / 'images' / 'decals_0.364_fits',
      suffix='.fits'
    )

    compressed_paths = iauname_relative_path(
      iaunames=iaunames,
      prefix=DATA_ROOT / 'images' / 'decals_0.364_fits_fz',
      suffix='.fits.fz'
    )

    for path, comp_path in tqdm(zip(descompressed_paths, compressed_paths)):
      compress_fits(
        file=path,
        compress_type='HCOMPRESS_1',
        hcomp_scale=3,
        quantize_level=10,
        quantize_method=-1,
        ext=1,
        save_path=comp_path,
        replace=True,
      )



if __name__ == '__main__':
  Job().run()
