from pathlib import Path

import numpy as np

from mergernet.core.experiment import Experiment, backup_model
from mergernet.core.hp import HP, HyperParameterSet
from mergernet.core.utils import iauname
from mergernet.data.dataset import Dataset, DatasetConfig
from mergernet.data.image import ColorImage
from mergernet.estimators.decals import ZoobotEstimator
from mergernet.model.automl import optuna_train
from mergernet.model.baseline import finetune_train
from mergernet.services.legacy import LegacyService


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 6
    self.log_wandb = False
    self.restart = True

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

    print(objects_positions[:, 0])

    ra = objects_positions[:, 0]
    dec = objects_positions[:, 1]

    iaunames = iauname(ra, dec)

    DATASET_PATH = Path(Experiment.local_exp_path) / 'exp_6_ds'

    input_paths = [
      DATASET_PATH / (iauname + '.fits')
      for iauname in iaunames
    ]

    output_paths = [
      DATASET_PATH / (iauname + '.png')
      for iauname in iaunames
    ]

    ls = LegacyService()
    ls.batch_cutout(
      ra=ra,
      dec=dec,
      save_path=output_paths,
      width=224,
      height=224,
      pixscale=0.364,
      workers=3,
      fmt='fits',
    )

    ColorImage.batch_legacy_rgb(
      images=input_paths,
      save_paths=output_paths,
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
      config=ZoobotEstimator.registry.ZOOBOT_GREY,
      crop_size=224,
      resize_size=224,
    )
    model.predict(output_path=Path(self.local_exp_path) / 'predictions.h5')

    Experiment.upload_file_gd('predictions.h5')


if __name__ == '__main__':
  j = Job()
  j.run()
