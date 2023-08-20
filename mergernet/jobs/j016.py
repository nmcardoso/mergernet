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
    self.exp_id = 16
    self.log_wandb = False
    self.restart = False

  def call(self):
    """
    Good mergers dataset
    """
    self.download_file_gd('mergers_5k_similarity.parquet', 15)
    self.download_file_gd('merger_only_features.parquet', 14)

    sim_df = pd.read_parquet(self.local_exp_path / 'mergers_5k_similarity.parquet')
    pos_df = pd.read_parquet(self.local_exp_path / 'merger_only_features.parquet')[['iauname', 'ra', 'dec']]

    sim_tb = XTable(ra='ra', dec='dec', df=sim_df)
    pos_tb = XTable(ra='ra', dec='dec', df=pos_df)

    xmatch = CrossMatch()
    result = xmatch.crossmatch(sim_tb, pos_tb, radius=60)

    repeated_iauname = result.table['iauname'].values

    unique_df = sim_df[~sim_df['iauname'].isin(repeated_iauname)]

    sample = unique_df[unique_df['count'] > 10]

    print(len(sample))

    paths = iauname_path(
      sample['iauname'].values,
      prefix=DATA_ROOT / 'images' / 'decals_0.364_png',
      suffix='.png'
    )

    tmp = Path(tempfile.gettempdir()) / f'mn_sample_{secrets.token_hex(3)}'
    tmp.mkdir(parents=True)

    for path in paths:
      shutil.copy(path, tmp / path.name)

    print(f'Sample created at {str(tmp)}')

    self.autoclean()



if __name__ == '__main__':
  Job().run()
