"""
Evaluate the best model of experiment 3 in BLIND_SPLUS_GAL80_LS10_RGB_128
dataset
"""

from mergernet.core.experiment import Experiment, backup_model
from mergernet.core.hp import HP, HyperParameterSet
from mergernet.data.dataset import Dataset
from mergernet.model.inference import Predictor
from mergernet.model.utils import load_model


class Job(Experiment):
  """
  Evaluate the best model of experiment 3 in BLIND_SPLUS_GAL80_LS10_RGB_128
  dataset

  Dataset
  -------
    - Train: BIN_LEGACY_NORTH_RGB_128
    - Predictions: BLIND_SPLUS_GAL80_LS10_RGB_128
  """
  def __init__(self):
    super().__init__()
    self.exp_id = 4
    self.log_wandb = True
    self.restart = True

  def call(self):
    ds = Dataset(config=Dataset.registry.BLIND_SPLUS_GAL80_LS10_RGB_128)
    model = load_model('model.h5', 3, '6bc82d80')
    p = Predictor(model, ds)
    p.predict()
    p.upload(label_map=Dataset.registry.BIN_LEGACY_NORTH_RGB_128.labels)



if __name__ == '__main__':
  Job().run()
