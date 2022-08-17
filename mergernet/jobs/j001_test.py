from mergernet.core.entity import HyperParameterSet
from mergernet.core.dataset import Dataset
from mergernet.core.experiment import Experiment, experiment_run
from mergernet.model.study import HyperModel


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
    'name': 'learning_rate',
    'type': 'loguniform',
    'low': 1.0e-5,
    'high': 1.0e-3
  }
]


@experiment_run(1)
def run():
  NAME = 'j001_test'

  ds = Dataset(config=Dataset.registry.BIN_SDSS_128)
  model = HyperModel(dataset=ds, name=NAME, epochs=10)
  model.hypertrain(
    n_trials=1,
    hyperparameters=HyperParameterSet(hp),
    pruner='hyperband',
    objective_metric='val_loss',
    objective_direction='minimize',
    resume=False,
    save_model=True
  )
