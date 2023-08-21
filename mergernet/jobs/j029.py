from itertools import compress

import pandas as pd

from mergernet.core.constants import DATA_ROOT
from mergernet.core.experiment import Experiment, backup_model
from mergernet.core.hp import HP, HyperParameterSet
from mergernet.core.utils import iauname, iauname_path, load_table
from mergernet.data.dataset import Dataset
from mergernet.data.image import (ChannelAverage, Crop, ImagePipeline,
                                  LegacyRGB, TensorToImage, TensorToShards)
from mergernet.estimators.automl import OptunaEstimator
from mergernet.estimators.parametric import ParametricEstimator
from mergernet.model.inference import Predictor
from mergernet.model.utils import load_model
from mergernet.services.legacy import LegacyService


class Job(Experiment):
  """Base model"""
  def __init__(self):
    super().__init__()
    self.exp_id = 29
    self.log_wandb = True
    self.restart = False


  def call(self):
    model = load_model('model.h5', exp_id=28)
    ds = Dataset(Dataset.registry.LS10S_BLIND_PNG)
    p = Predictor(model=model, dataset=ds)
    p.predict()
    p.upload()


if __name__ == '__main__':
  Job().run()
