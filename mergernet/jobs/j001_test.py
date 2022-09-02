from mergernet.core.experiment import Experiment, experiment_run
from mergernet.core.hp import HyperParameterSet
from mergernet.data.dataset import Dataset
from mergernet.model.automl import optuna_train
from mergernet.model.baseline import finetune_train

hp = [
  {
    'name': 'architecture',
    'type': 'constant',
    'value': 'resnet50'
  },
  {
    'name': 'pretrained_weights',
    'type': 'constant',
    'value': 'imagenet'
  },
  {
    'name': 'batch_size',
    'type': 'constant',
    'value': 64
  },
  {
    'name': 'dense_1_units',
    'type': 'int',
    'low': 64,
    'high': 1024,
    'step': 64
  },
  {
    'name': 'dropout_1_rate',
    'type': 'uniform',
    'low': 0.2,
    'high': 0.5
  },
  {
    'name': 'dense_2_units',
    'type': 'int',
    'low': 64,
    'high': 1024,
    'step': 64
  },
  {
    'name': 'dropout_2_rate',
    'type': 'uniform',
    'low': 0.2,
    'high': 0.5
  },
  {
    'name': 'opt_lr',
    'type': 'loguniform',
    'low': 1.0e-5,
    'high': 1.0e-3
  }
]


@experiment_run(1)
def run():
  """
  Experiment created for test purposes. This experiments tests the
  ``Artifacts API``
  """
  ds = Dataset(config=Dataset.registry.BIN_SDSS_128)
  optuna_train(
    train_func=finetune_train,
    dataset=ds,
    hp=HyperParameterSet(hp),
    n_trials=30
  )



if __name__ == '__main__':
  run()
