import pandas as pd

from mergernet.core.constants import DATA_ROOT
from mergernet.core.experiment import Experiment
from mergernet.core.utils import iauname_path
from mergernet.services.legacy import LegacyService


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 21
    self.log_wandb = False
    self.restart = False


  def call(self):
    df = pd.read_parquet(DATA_ROOT / 'ls10s_blind.parquet')[44_000:]

    paths = iauname_path(
      ra=df.ra.values,
      dec=df.dec.values,
      prefix=DATA_ROOT / 'images' / 'ls10s_blind_fits_fz',
      suffix='.fits.fz'
    )

    ls = LegacyService(
      fmt='fits',
      width=224,
      height=224,
      pixscale=0.4,
      bands='grz',
      workers=7,
      compress_fits=True,
    )

    ls.batch_cutout(
      ra=df.ra.values,
      dec=df.dec.values,
      save_path=paths,
      mag_r=df.mag_r.values
    )


if __name__ == '__main__':
  Job().run()
