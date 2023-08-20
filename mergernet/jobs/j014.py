import numpy as np
import pandas as pd
import pymilvus as pm
from tqdm import tqdm

from mergernet.core.experiment import Experiment
from mergernet.core.utils import iauname, iauname_path
from mergernet.data.dataset import Dataset
from mergernet.data.dataset_config import DatasetConfig
from mergernet.data.image import Crop, ImagePipeline, LegacyRGB, TensorToImage
from mergernet.estimators.decals import ZoobotEstimator
from mergernet.services.legacy import LegacyService


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 14
    self.log_wandb = False
    self.restart = False

  def call(self):
    """
    Good mergers dataset
    """
    self.download_file_gd('merger_only.csv', shared=True)

    df = pd.read_csv(self.local_exp_path / 'merger_only.csv')

    ls = LegacyService(
      fmt='fits',
      width=224,
      height=224,
      pixscale=0.364,
      bands='grz',
      workers=7,
      compress_fits=True,
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

    positions = [(ra, dec) for ra, dec in zip(df.ra.values, df.dec.values)]

    ds_config = DatasetConfig(
      name='exp_14_mergers_only',
      images_path='exp_14_ds',
      image_extension='png',
      image_shape=(224, 224, 3),
      positions=positions,
      image_transform=rgb_transform,
      image_service=ls,
    )

    ds = Dataset(ds_config)

    model = ZoobotEstimator(
      hp=None,
      dataset=ds,
      config=ZoobotEstimator.registry.ZOOBOT_GREYSCALE,
      crop_size=224,
      resize_size=224,
    )

    features_table_name = 'merger_only_features.parquet'

    model.cnn_representations(
      features_table_name,
      include_cols={'ra': df.ra.values, 'dec': df.dec.values}
    )
    self.upload_file_gd(features_table_name)

    self.autoclean()



if __name__ == '__main__':
  Job().run()
