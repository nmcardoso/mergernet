import pandas as pd

from mergernet.core.constants import DATA_ROOT
from mergernet.core.experiment import Experiment
from mergernet.core.utils import iauname_relative_path
from mergernet.services.legacy import LegacyService


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 5
    self.log_wandb = False
    self.restart = False


  def call(self):
    Experiment.download_file_gd('decals_pca10.csv', exp_id=3)

    df = pd.read_csv(Experiment.local_exp_path / 'decals_pca10.csv')

    paths = iauname_relative_path(
      ra=df.ra.values,
      dec=df.dec.values,
      prefix=DATA_ROOT / 'images' / 'decals_0.364_fits_fz',
      suffix='.fits.fz'
    )

    ls = LegacyService(
      fmt='fits',
      width=300,
      height=300,
      pixscale=0.364,
      bands='grz',
      workers=7,
      compress_fits=True,
    )

    ls.batch_cutout(
      df.ra.values,
      df.dec.values,
      paths
    )


if __name__ == '__main__':
  Job().run()
