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
    self.exp_id = 26
    self.log_wandb = True
    self.restart = False


  def call(self):
    hps = HyperParameterSet(
      HP.const('architecture', 'efficientnetv2b0'),
      HP.const('pretrained_weights', 'imagenet'),
      HP.const('batch_size', 64),
      # HP.const('metrics', ['tpr']),
      HP.const('positive_class_id', 1),
      HP.const('negative_class_id', 0),
      HP.const('epochs', 35),
      HP.const('tl_epochs', 12),
      HP.const('t1_opt', 'adamw'),
      HP.num('t1_lr', low=2e-4, high=5e-3, log=True),
      HP.const('optimizer', 'adamw'),
      HP.const('lr_decay', 'cosine'),
      HP.num('lr_decay_steps', low=0.5, high=0.9),
      HP.num('lr_decay_alpha', low=0.1, high=1.0),
      HP.num('opt_lr', low=1e-5, high=1e-3, log=True),
      HP.num('weight_decay', low=1e-4, high=1e-1, log=True),
      HP.num('label_smoothing', low=0, high=0.1),
      HP.num('dense_1_units', low=32, high=1024, step=1, dtype=int),
      HP.num('dropout_1_rate', low=0.1, high=0.5),
      HP.num('dense_2_units', low=32, high=1024, step=1, dtype=int),
      HP.num('dropout_2_rate', low=0.1, high=0.5),
    )
    ds = Dataset(config=Dataset.registry.LS10_TRAIN_224_PNG)

    model = ParametricEstimator(hp=hps, dataset=ds)

    optuna_model = OptunaEstimator(estimator=model, n_trials=20, resume=True)

    optuna_model.train()

    Experiment.upload_file_gd('model.h5', optuna_model.tf_model)


if __name__ == '__main__':
  Job().run()
