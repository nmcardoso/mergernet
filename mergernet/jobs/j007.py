from itertools import compress

import pandas as pd

from mergernet.core.constants import DATA_ROOT
from mergernet.core.experiment import Experiment
from mergernet.core.utils import iauname, iauname_relative_path
from mergernet.data.image import (ChannelAverage, Crop, ImagePipeline,
                                  LegacyRGB, TensorToImage, TensorToShards)
from mergernet.services.legacy import LegacyService


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 7
    self.log_wandb = False
    self.restart = False


  def call(self):
    Experiment.download_file_gd('decals_pca10.csv', exp_id=3)

    df = pd.read_csv(Experiment.local_exp_path / 'decals_pca10.csv')[700_000:]

    png_path = DATA_ROOT / 'images' / 'decals_0.364_png'
    fits_path = DATA_ROOT / 'images' / 'decals_0.364_fits_fz'

    iaunames = iauname(ra=df.ra.values, dec=df.dec.values)

    paths = iauname_relative_path(
      iaunames=iaunames,
      prefix=fits_path,
      suffix='.fits.fz'
    )

    save_paths = iauname_relative_path(
      iaunames=iaunames,
      prefix=png_path,
      suffix='.png'
    )

    mask = [not p.exists() for p in save_paths]
    paths = list(compress(paths, mask))
    save_paths = list(compress(save_paths, mask))

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

    avg_transform = ChannelAverage(return_int=True, normalize=True)

    crop_transform = Crop(224)

    shards_transform = TensorToShards(
      save_path=DATA_ROOT / 'images' / 'shards_test',
      examples_per_shard=20
    )

    img_transform = TensorToImage(save_paths=save_paths)

    pipe = ImagePipeline([crop_transform, rgb_transform, img_transform])

    pipe.batch_transform(images=paths)





if __name__ == '__main__':
  Job().run()
