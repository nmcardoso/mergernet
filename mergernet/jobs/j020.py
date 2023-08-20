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
    self.exp_id = 20
    self.log_wandb = False
    self.restart = False

  def call(self):
    base_path = DATA_ROOT / 'gal_pairs'

    table_pairs = [
      (
        base_path / 'gal_pairs_r15-17_p1.csv',
        base_path / 'gal_pairs_r15-17_p1+insp.csv'
      ),
      (
        base_path / 'gal_pairs_r17-19_p1.csv',
        base_path / 'gal_pairs_r17-19_p1+insp.csv'
      ),
      (
        base_path / 'gal_pairs_r17-19_p2.csv',
        base_path / 'gal_pairs_r17-19_p2+insp.csv'
      ),
    ]

    # table_pairs = [
    #   (
    #     DATA_ROOT / 'gal_pairs' /
    #   )
    # ]

    df = None

    for pair in table_pairs:
      xmatch = CrossMatch()
      tb_full = XTable(ra='ra', dec='dec', path=pair[0])
      tb_insp = XTable(ra='ra', dec='dec', path=pair[1], columns=['XTableClass'])
      result = xmatch.crossmatch(tb_full, tb_insp, radius=1)

      if df is None:
        df = result.table
      else:
        df = pd.concat((df, result.table), ignore_index=True)

    save_table(df, base_path / 'gal_pairs+insp.csv')

    self.autoclean()



if __name__ == '__main__':
  Job().run()
