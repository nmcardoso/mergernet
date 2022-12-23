import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from mergernet.core.experiment import Experiment, backup_model
from mergernet.core.hp import HP, HyperParameterSet
from mergernet.core.utils import iauname
from mergernet.data.dataset import Dataset, DatasetConfig
from mergernet.data.image import ColorImage
from mergernet.estimators.decals import ZoobotEstimator
from mergernet.model.automl import optuna_train
from mergernet.model.baseline import finetune_train
from mergernet.services.legacy import LegacyService

L = logging.getLogger(__name__)


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 6
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
      (55.0375, -35.62444),
      (62.505, -31.2583),
      (62.752, -31.407),
      (57.1255, -39.42502),
      (52.8777, -30.2123),
      (51.9916, -37.1494),
    ])

    ra = objects_positions[:, 0]
    dec = objects_positions[:, 1]
    iaunames = iauname(ra, dec)

    dataset_path = Path(Experiment.local_shared_path) / 'exp_6_ds'

    fits_paths = [
      dataset_path / (iauname + '.fits')
      for iauname in iaunames
    ]

    png_paths = [
      dataset_path / (iauname + '.png')
      for iauname in iaunames
    ]

    ls = LegacyService()
    ls.batch_cutout(
      ra=ra,
      dec=dec,
      save_path=fits_paths,
      width=224,
      height=224,
      pixscale=0.364,
      workers=3,
      fmt='fits',
    )

    df = pd.DataFrame({
      'ra': ra,
      'dec': dec,
      'iauname': iaunames,
      'fits_paths': fits_paths,
      'png_paths': png_paths
    })
    not_downloaded_iaunames = [i.stem for i in fits_paths if not i.exists()]
    df = df[~df.iauname.isin(not_downloaded_iaunames)]
    L.info(f'Not dowloaded files: {len(not_downloaded_iaunames)}')

    ColorImage.batch_legacy_rgb(
      images=df.fits_paths.to_numpy(),
      save_paths=df.png_paths.to_numpy(),
      bands='grz',
      scales={
        'g': (2, 0.008),
        'r': (1, 0.014),
        'z': (0, 0.019)
      },
      minmax=(-0.5, 300),
      brightness=1.3,
      desaturate=True,
      nl_func=ColorImage.asinh_map2
    )

    config = DatasetConfig(
      name='exp_6_dataset',
      images_path='exp_6_ds',
      X_column_suffix='.png',
      image_shape=(224, 224, 3),
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


    repr_df = pd.read_csv(Path(Experiment.local_exp_path) / 'representations.csv')
    repr_df = pd.merge(df[['ra', 'dec', 'iauname']], repr_df, 'inner', 'iauname')
    feat_cols = [col for col in repr_df.columns.values if col.startswith('feat_')]
    repr_feat = repr_df[feat_cols].to_numpy()

    Experiment.download_file_gd('cnn_features_decals.parquet', shared=True)
    decals_feat_df = pd.read_parquet(Path(Experiment.local_exp_path) / 'cnn_features_decals.parquet')
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

    Experiment.upload_file_gd('representations_pca.csv', pca_df)

if __name__ == '__main__':
  j = Job()
  j.run()
