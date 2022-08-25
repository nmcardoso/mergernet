from mergernet.core.entity import HyperParameterSet
from mergernet.core.dataset import Dataset
from mergernet.core.experiment import backup_model, experiment_run
from mergernet.model.train import finetune_train


hps = [
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
    'type': 'constant',
    'value': 128
  },
  {
    'name': 'dropout_1_rate',
    'type': 'constant',
    'value': 0.4
  },
  {
    'name': 'dense_2_units',
    'type': 'constant',
    'value': 64
  },
  {
    'name': 'dropout_2_rate',
    'type': 'constant',
    'value': 0.4
  },
  {
    'name': 'opt_lr',
    'type': 'constant',
    'value': 1e-4
  }
]


@experiment_run(2)
def run():
  """
  Test the finetune train with constant hyperparameters
  """
  ds = Dataset(config=Dataset.registry.BIN_SDSS_128)
  model = finetune_train(ds, HyperParameterSet(hps))
  backup_model(model, ds)



if __name__ == '__main__':
  run()
