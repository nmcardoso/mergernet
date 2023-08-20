import pandas as pd

from mergernet.core.constants import DATA_ROOT
from mergernet.core.experiment import Experiment
from mergernet.core.utils import iauname_path
from mergernet.services.legacy import LegacyService


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 24
    self.log_wandb = False
    self.restart = False


  def call(self):
    df = pd.read_csv(DATA_ROOT / 'ls10_train.csv')

    paths = iauname_path(
      iaunames=df.iauname,
      prefix=DATA_ROOT / 'images' / 'ls10_train_224_fits_fz',
      suffix='.fits.fz'
    )

    ls = LegacyService(
      fmt='fits',
      width=224,
      height=224,
      pixscale=0.5,
      bands='grz',
      workers=7,
      compress_fits=True,
      replace=True,
    )

    ls.batch_cutout(
      ra=df.ra.values,
      dec=df.dec.values,
      save_path=paths,
      mag_r=df.mag_r.values
    )


if __name__ == '__main__':
  Job().run()
