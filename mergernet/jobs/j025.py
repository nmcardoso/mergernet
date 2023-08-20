from itertools import compress
from time import sleep

import pandas as pd

from mergernet.core.constants import DATA_ROOT
from mergernet.core.experiment import Experiment
from mergernet.core.utils import iauname, iauname_path
from mergernet.data.image import (ChannelAverage, Crop, ImagePipeline,
                                  LegacyRGB, TensorToImage, TensorToShards)
from mergernet.services.legacy import LegacyService


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 25
    self.log_wandb = False
    self.restart = False


  def call(self):
    df = pd.read_csv(DATA_ROOT / 'bin_ds.csv')

    png_path = DATA_ROOT / 'images' / 'ls10_train_224_png'
    fits_path = DATA_ROOT / 'images' / 'ls10_train_224_fits_fz'

    iaunames = df.iauname.values

    paths = iauname_path(
      iaunames=iaunames,
      prefix=fits_path,
      suffix='.fits.fz'
    )

    save_paths = iauname_path(
      iaunames=iaunames,
      prefix=png_path,
      suffix='.png'
    )

    rgb_transform = LegacyRGB(
      bands='grz',
      scales={
        'g': (2, 0.008),
        'r': (1, 0.014),
        'z': (0, 0.019)
      },
      minmax=(-0.5, 300),
      brightness=1.3,
      desaturate=True,
      nl_func='asinh2',
      rgb_output=True,
    )

    while True:
      mask = [not p.exists() for p in save_paths]
      paths = list(compress(paths, mask))
      save_paths = list(compress(save_paths, mask))

      img_transform = TensorToImage(save_paths=save_paths)

      pipe = ImagePipeline([rgb_transform, img_transform])

      errors = pipe.batch_transform(images=paths, silent=True)

      if len(errors) > 0: sleep(60)
      else: break





if __name__ == '__main__':
  Job().run()
