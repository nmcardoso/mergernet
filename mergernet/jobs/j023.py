from itertools import compress

import pandas as pd

from mergernet.core.constants import DATA_ROOT
from mergernet.core.experiment import Experiment, backup_model
from mergernet.core.hp import HP, HyperParameterSet
from mergernet.core.utils import iauname, iauname_path
from mergernet.data.dataset import Dataset
from mergernet.data.image import (ChannelAverage, Crop, ImagePipeline,
                                  LegacyRGB, TensorToImage, TensorToShards)
from mergernet.estimators.automl import OptunaEstimator
from mergernet.estimators.parametric import ParametricEstimator
from mergernet.services.legacy import LegacyService


class Job(Experiment):
  """Base model"""
  def __init__(self):
    super().__init__()
    self.exp_id = 23
    self.log_wandb = True
    self.restart = True


  def call(self):
    hps = HyperParameterSet(
      HP.const('architecture', 'convnext_tiny'),
      HP.const('pretrained_weights', 'imagenet'),
      HP.const('tl_epochs', 7),
      HP.const('epochs', 20),
      HP.const('batch_size', 64),
      HP.num('dense_1_units', low=64, high=1024, step=64, dtype=int),
      HP.num('dropout_1_rate', low=0.2, high=0.5),
      HP.num('dense_2_units', low=64, high=1024, step=64, dtype=int),
      HP.num('dropout_2_rate', low=0.2, high=0.5),
      HP.num('opt_lr', low=1e-5, high=1e-3, log=True)
    )
    ds = Dataset(config=Dataset.registry.LS10_TRAIN_224_PNG)

    model = ParametricEstimator(hp=hps, dataset=ds)

    optuna_model = OptunaEstimator(estimator=model, n_trials=5)

    optuna_model.train()

    Experiment.upload_file_gd('model.h5', optuna_model.tf_model)


if __name__ == '__main__':
  Job().run()
