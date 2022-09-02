import logging
from pathlib import Path
from types import FunctionType

import optuna

from mergernet.core.constants import RANDOM_SEED
from mergernet.core.experiment import Experiment
from mergernet.core.hp import HyperParameterSet
from mergernet.core.utils import Timming
from mergernet.data.dataset import Dataset

L = logging.getLogger(__name__)
optuna.logging.disable_default_handler()
optuna.logging.enable_propagation()



def _objective_factory(
  train_func: FunctionType,
  dataset: Dataset,
  hp: HyperParameterSet
) -> FunctionType:
  def objective(trial: optuna.trial.FrozenTrial) -> float:
    # run train function
    hp.set_trial(trial)
    model = train_func(dataset=dataset, hp=hp)

    # generating optuna value to optimize (val_accuracy)
    h = model.history.history
    objective_value = h['val_loss'][-1]

    return objective_value
  return objective



def optuna_train(
  train_func: FunctionType,
  dataset: Dataset,
  hp: HyperParameterSet,
  n_trials: int,
  pruner: str = 'hyperband',
  objective_metric: str = 'val_loss',
  objective_direction: str = 'minimize',
  save_model: bool = True,
  name: str = None
):
  name = name or f'exp_{Experiment.exp_id}'

  if pruner == 'median':
    pruner_instance = optuna.pruners.MedianPruner(
      n_startup_trials=10,
      n_warmup_steps=10
    )
  elif pruner == 'hyperband':
    pruner_instance = optuna.pruners.HyperbandPruner(min_resource=7)

  L.info(f'start of optuna optimization')

  optuna_path = Path(Experiment.local_artifact_path) / 'optuna.sqlite'
  optuna_uri = f'sqlite:///{str(optuna_path.resolve())}' # absolute path

  t = Timming()
  study = optuna.create_study(
    storage=optuna_uri,
    study_name=name,
    pruner=pruner_instance,
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    direction=objective_direction,
    load_if_exists=True
  )

  study.optimize(
    func=_objective_factory(train_func, dataset, hp),
    n_trials=n_trials
  )
  t.end()

  Experiment.upload_file_gh('optuna.sqlite')

  L.info(f'optuna optimization finished in {t.duration()}')
  L.info(f'number of finished trials: {len(study.trials)}')
  L.info(f'----- begin of best trial summary -----')
  L.info(f'optimization score: {study.best_trial.value}')
  L.info(f'params:')
  for k, v in study.best_params.items():
    L.info(f'{k}: {str(v)}')
  L.info(f'----- end of best trial summary -----')
