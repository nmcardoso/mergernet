import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from mergernet.core.experiment import Experiment, backup_model
from mergernet.core.hp import HP, HyperParameterSet
from mergernet.core.utils import iauname
from mergernet.data.dataset import Dataset, DatasetConfig
from mergernet.data.image import LegacyRGB
from mergernet.estimators.decals import ZoobotEstimator
from mergernet.model.automl import optuna_train
from mergernet.model.baseline import finetune_train
from mergernet.services.legacy import LegacyService

L = logging.getLogger(__name__)


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 2
    self.log_wandb = False
    self.restart = False

  def call(self):
    objects_positions = np.array([
      (229.30982, 21.58542),
      (55.0375, -35.62444),
      (55.7583, -36.27330),
      (57.9204, -38.45220),
      (302.5614714, -56.6415254),
      (10.3762293, -9.2627726),
      (318.44754, 2.4765272),
      (356.1110805, 9.1151493),
      (353.7240193, 27.3522417),
      # (55.0375, -35.62444),
      (62.505, -31.2583),
      (62.752, -31.407),
      (57.1255, -39.42502),
      (52.8777, -30.2123),
      (51.9916, -37.1494),
    ])

    leg_service = LegacyService(
      width=224,
      height=224,
      pixscale=0.364,
      workers=3,
      fmt='fits',
    )

    leg_rgb = LegacyRGB(
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
    )

    config = DatasetConfig(
      name='exp_2_dataset',
      images_path='exp_2_ds',
      image_extension='png',
      image_shape=(224, 224, 3),
      positions=objects_positions,
      image_transform=leg_rgb,
      image_service=leg_service,
    )
    ds = Dataset(config=config)

    model = ZoobotEstimator(
      hp=None,
      dataset=ds,
      config=ZoobotEstimator.registry.ZOOBOT_GREYSCALE,
      crop_size=224,
      resize_size=224,
    )

    model.cnn_representations(filename='representations.csv')
    model.plot('model.png')

    Experiment.upload_file_gd('representations.csv')
    Experiment.upload_file_gd('model.png')

    df = ds.get_table()

    df = pd.DataFrame({
      'ra': df['ra'].to_numpy(),
      'dec': df['dec'].to_numpy(),
      'iauname': df['iauname'].to_numpy(),
    })

    repr_df = pd.read_csv(Experiment.local_exp_path / 'representations.csv')
    repr_df = pd.merge(df, repr_df, 'inner', 'iauname')
    feat_cols = [col for col in repr_df.columns.values if col.startswith('feat_')]
    repr_feat = repr_df[feat_cols].to_numpy()

    Experiment.download_file_gd('cnn_features_decals.parquet', shared=True)
    decals_feat_df = pd.read_parquet(Experiment.local_exp_path / 'cnn_features_decals.parquet')
    feat_cols = [col for col in decals_feat_df.columns.values if col.startswith('feat_')]
    decals_feat = decals_feat_df[feat_cols].to_numpy()

    all_feat = np.concatenate((decals_feat, repr_feat))

    pca_df = model.pca(all_feat[-1000:], 10)
    pca_df = pca_df.iloc[-len(repr_df):]
    pca_df = pd.concat(
      (
        repr_df[['ra', 'dec', 'iauname']].reset_index(drop=True),
        pca_df.reset_index(drop=True)
      ),
      axis=1
    )
    pca_df = pca_df.drop_duplicates('iauname')

    Experiment.upload_file_gd('representations_pca.csv', pca_df)

if __name__ == '__main__':
  j = Job()
  j.run()
