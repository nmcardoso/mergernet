import secrets
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pymilvus as pm
from tqdm import tqdm

from mergernet.core.constants import DATA_ROOT
from mergernet.core.experiment import Experiment
from mergernet.core.utils import Timming, iauname, iauname_path
from mergernet.data.dataset import Dataset
from mergernet.data.dataset_config import DatasetConfig
from mergernet.data.image import Crop, ImagePipeline, LegacyRGB, TensorToImage
from mergernet.data.match import CrossMatch, XTable
from mergernet.data.sanitization import DatasetSanitization
from mergernet.estimators.decals import ZoobotEstimator
from mergernet.estimators.similarity import MilvusClusterSimilarity
from mergernet.services.legacy import LegacyService


def normalize_cnn_features(features):
  return features / np.linalg.norm(features, axis=1)[:, np.newaxis]



class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 18
    self.log_wandb = False
    self.restart = False

  def call(self):
    """
    Download stamps of Legacy Survey DR10 South (tractor_s table)
    15 <= mag_r <= 17
    """
    table_path = DATA_ROOT / 'ls10_s_r14-19.parquet'
    df = pd.read_parquet(table_path)

    if 'iauname' not in df.columns.values:
      df.insert(0, 'iauname', iauname(df.ra.values, df.dec.values))
      df.to_parquet(table_path, index=False)
      df = pd.read_parquet(table_path)

    df = df[
      df.mag_r.between(15, 17)
      & (df.nobs_g >= 1) & (df.nobs_r >= 1) & (df.nobs_z >= 1)
    ]
    df = df[320_000:400_000]

    paths = iauname_path(
      iaunames=df.iauname.values,
      prefix=DATA_ROOT / 'images' / 'ls_dr10_south_0.364_fits_fz',
      suffix='.fits.fz'
    )

    ls = LegacyService(
      fmt='fits',
      width=300,
      height=300,
      pixscale=0.364,
      bands='grz',
      workers=1,
      compress_fits=True,
      layer='ls-dr10'
    )

    ls.batch_cutout(
      df.ra.values,
      df.dec.values,
      paths
    )

    self.autoclean()



if __name__ == '__main__':
  Job().run()
