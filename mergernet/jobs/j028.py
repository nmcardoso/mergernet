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
    self.exp_id = 28
    self.log_wandb = True
    self.restart = False


  def call(self):
    hps = HyperParameterSet(
      # network architecture
      HP.const('architecture', 'efficientnetv2b0'),
      HP.const('pretrained_weights', 'imagenet'),
      HP.const('batch_size', 128),
      HP.const('dense_1_units', 400),
      HP.const('batch_norm_1', True),
      HP.const('dropout_1_rate', 0.32),
      HP.const('features_reduction', 'avg_pooling'),
      # metrics and validation
      HP.const('metrics', ['f1', 'recall', 'roc']),
      HP.const('positive_class_id', 1),
      HP.const('negative_class_id', 0),
      # t1 setup
      HP.const('t1_epochs', 15),
      HP.const('t1_opt', 'adamw'),
      HP.const('t1_lr', 5e-4),
      # main train setup
      HP.const('epochs', 800),
      HP.const('optimizer', 'adamw'),
      HP.const('lr_decay', 'cosine_restarts'),
      HP.const('lr_decay_steps', 6),
      HP.const('lr_decay_t', 1.5),
      HP.const('lr_decay_m', 1.0),
      HP.const('lr_decay_alpha', 0.1),
      HP.const('opt_lr', 2e-4),
      HP.const('weight_decay', 0.08),
      HP.const('label_smoothing', 0.07),
    )
    ds = Dataset(config=Dataset.registry.LS10_TRAIN_224_PNG)

    model = ParametricEstimator(hp=hps, dataset=ds)
    model.train(run_name='with_t1_softmax_logits')

    Experiment.upload_file_gd('model_softmax_logits.h5', model.tf_model)


if __name__ == '__main__':
  Job().run()
