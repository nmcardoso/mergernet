from mergernet.core.experiment import backup_model, experiment_run
from mergernet.core.hp import HP, HyperParameterSet
from mergernet.data.dataset import Dataset
from mergernet.model.automl import optuna_train
from mergernet.model.baseline import finetune_train

hps = HyperParameterSet(
  HP.const('architecture', 'resnet50'),
  HP.const('pretrained_weights', 'imagenet'),
  HP.const('epochs', 20),
  HP.const('batch_size', 64),
  HP.num('dense_1_units', low=64, high=1024, step=64, dtype=int),
  HP.num('dropout_1_rate', low=0.2, hight=0.5),
  HP.num('dense_2_units', low=64, high=1024, step=64, dtype=int),
  HP.num('dropout_1_rate', low=0.2, hight=0.5),
  HP.num('opt_lr', low=1e-5, high=1e-3, log=True)
)


@experiment_run(1)
def run():
  """
  Experiment created for test purposes. This experiments tests the
  ``Artifacts API``
  """
  ds = Dataset(config=Dataset.registry.BIN_LEGACY_NORTH_RGB_128)
  model = optuna_train(
    train_func=finetune_train,
    dataset=ds,
    hp=hps,
    n_trials=1
  )
  backup_model(model, ds)



if __name__ == '__main__':
  run()
