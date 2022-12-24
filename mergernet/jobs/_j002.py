"""
Test the finetune train with constant hyperparameters of binary
"""

from mergernet.core.experiment import Experiment, backup_model
from mergernet.core.hp import HP, HyperParameterSet
from mergernet.data.dataset import Dataset
from mergernet.model.baseline import finetune_train


class Job(Experiment):
  """
  Test the finetune train with constant hyperparameters of binary
  classification (merger/non-merger) on Legacy North with RGB images.
  This experiment uses the new hyperparameter API

  Dataset
  -------
    - BIN_LEGACY_NORTH_RGB_128: binary dataset merger/non-merger with
      128x128 RGB stamps of Legacy Survey with 0.55 pixscale

  Train
  -----
    - Training baseline model with finetune
  """
  def __init__(self):
    super().__init__()
    self.exp_id = 2
    self.log_wandb = True
    self.restart = False

  def call(self):
    hps = HyperParameterSet(
      HP.const('architecture', 'resnet50'),
      HP.const('pretrained_weights', 'imagenet'),
      HP.const('epochs', 12),
      HP.const('batch_size', 64),
      HP.const('dense_1_units', 128),
      HP.const('dropout_1_rate', 0.4),
      HP.const('dense_2_units', 64),
      HP.const('dropout_2_rate', 0.4),
      HP.const('opt_lr', 1e-4)
    )
    ds = Dataset(config=Dataset.registry.BIN_LEGACY_NORTH_RGB_128)
    model = finetune_train(ds, hps)
    backup_model(model, ds)



if __name__ == '__main__':
  Job().run()
