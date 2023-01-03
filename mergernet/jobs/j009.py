"""
Optuna finetune train with for binary classification (merger/non-merger)
on Legacy North with RGB images.
"""

from mergernet.core.experiment import Experiment, backup_model
from mergernet.core.hp import HP, HyperParameterSet
from mergernet.data.dataset import Dataset
from mergernet.estimators.automl import OptunaEstimator
from mergernet.estimators.parametric import ParametricEstimator


class Job(Experiment):
  """
  Optuna finetune train with for binary classification (merger/non-merger)
  on Legacy North with RGB images.

  Dataset
  -------
  - BIN_LEGACY_NORTH_RGB_128

  Train
  -----
  - Using Optuna automl
  - Using ResNet50v2
  - Using parametric model with finetune
  """
  def __init__(self):
    super().__init__()
    self.exp_id = 9
    self.log_wandb = True
    self.restart = True

  def call(self):
    hps = HyperParameterSet(
      HP.const('architecture', 'resnet50v2'),
      HP.const('pretrained_weights', 'imagenet'),
      HP.const('tl_epochs', 2),
      HP.const('epochs', 3),
      HP.const('batch_size', 64),
      HP.num('dense_1_units', low=64, high=1024, step=64, dtype=int),
      HP.num('dropout_1_rate', low=0.2, high=0.5),
      HP.num('dense_2_units', low=64, high=1024, step=64, dtype=int),
      HP.num('dropout_2_rate', low=0.2, high=0.5),
      HP.num('opt_lr', low=1e-5, high=1e-3, log=True)
    )
    ds = Dataset(config=Dataset.registry.BIN_LEGACY_NORTH_RGB_128)

    model = ParametricEstimator(hp=hps, dataset=ds)

    optuna_model = OptunaEstimator(estimator=model, n_trials=12)

    optuna_model.train()

    Experiment.upload_file_gd('model.h5', optuna_model.tf_model)



if __name__ == '__main__':
  Job().run()
