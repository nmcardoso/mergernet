import logging
from pathlib import Path
from types import FunctionType

import optuna
import tensorflow as tf

from mergernet.core.constants import RANDOM_SEED
from mergernet.core.experiment import Experiment
from mergernet.core.hp import HyperParameterSet
from mergernet.core.utils import Timming
from mergernet.data.dataset import Dataset
from mergernet.model.callbacks import SaveBestTrialCallback

L = logging.getLogger(__name__)
optuna.logging.disable_default_handler()
optuna.logging.enable_propagation()



def _objective_factory(
  train_func: FunctionType,
  dataset: Dataset,
  hp: HyperParameterSet,
  callbacks: list = []
) -> FunctionType:
  def objective(trial: optuna.trial.FrozenTrial) -> float:
    # run train function
    hp.set_trial(trial)
    model = train_func(dataset=dataset, hp=hp, callbacks=callbacks)

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
  name: str = None,
  resume_hash: str = None
):
  exp_id = Experiment.exp_id
  name = name or f'exp_{exp_id}'

  if pruner == 'median':
    pruner_instance = optuna.pruners.MedianPruner(
      n_startup_trials=10,
      n_warmup_steps=10
    )
  elif pruner == 'hyperband':
    pruner_instance = optuna.pruners.HyperbandPruner(min_resource=7)

  L.info(f'start of optuna optimization')

  optuna_path = Path(Experiment.local_run_path) / 'optuna.sqlite'
  optuna_uri = f'sqlite:///{str(optuna_path.resolve())}' # absolute path

  if resume_hash is not None:
    L.info(f'Downloading optuna study of exp {exp_id} run {resume_hash}')
    Experiment.download_file_gh('optuna.sqlite', exp_id, resume_hash)

  t = Timming()

  if resume_hash is None:
    study = optuna.create_study(
      storage=optuna_uri,
      study_name=name,
      pruner=pruner_instance,
      sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
      direction=objective_direction,
      load_if_exists=True
    )
  else:
    study = optuna.load_study(
      storage=optuna_uri,
      study_name=name,
      pruner=pruner_instance,
      sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
      direction=objective_direction,
      load_if_exists=True
    )

  callbacks = []
  if save_model:
    callbacks.append(
      SaveBestTrialCallback(study, objective_metric, objective_direction, name)
    )

  study.optimize(
    func=_objective_factory(train_func, dataset, hp, callbacks=callbacks),
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

  model = tf.keras.models.load_model(
    Path(Experiment.local_run_path) / f'{name}.h5'
  )
  return model
