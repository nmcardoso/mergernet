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
from mergernet.core.utils import (Timming, iauname, iauname_path, load_table,
                                  save_table)
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
    self.exp_id = 19
    self.log_wandb = False
    self.restart = False

  def call(self):
    """
    Download stamps of Legacy Survey DR10 South (tractor_s table)
    15 <= mag_r <= 17
    """
    # table_path = '/home/natan/Downloads/star_gal_pairs_r15-17_spec.csv'
    table_path = DATA_ROOT / 'merger_only2.csv'
    df = load_table(table_path, default=False)

    if 'iauname' not in df.columns.values:
      df.insert(0, 'iauname', iauname(df.ra.values, df.dec.values))
      save_table(df, table_path, default=False)
      df = load_table(table_path, default=False)

    paths = iauname_path(
      iaunames=df.iauname.values,
      prefix=DATA_ROOT / 'images' / 'vis',
      suffix='.jpg',
      flat=True
    )

    ls = LegacyService(
      fmt='jpg',
      width=300,
      height=300,
      pixscale=0.364,
      bands='grz',
      workers=1,
      layer='ls-dr9'
    )

    ls.batch_cutout(
      df.ra.values,
      df.dec.values,
      paths
    )

    self.autoclean()



if __name__ == '__main__':
  Job().run()
