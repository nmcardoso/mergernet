import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
import wandb

from mergernet.core.experiment import Experiment
from mergernet.data.dataset import Dataset

L = logging.getLogger(__name__)



class WandbGraphicsCallback(tf.keras.callbacks.Callback):
  def __init__(self, validation_data, labels):
    self.validation_data = validation_data
    self.labels = labels

  def on_train_end(self, logs: dict = None):
    history = deepcopy(self.model.history.history)

    probs = self.model.predict(self.validation_data)
    y_true_one_hot = np.concatenate([y for _, y in self.validation_data], axis=0)
    y_true = np.argmax(y_true_one_hot, axis=-1)

    wandb.log({
      'confusion_matrix': wandb.plot.confusion_matrix(
        probs=probs,
        y_true=y_true,
        class_names=self.labels,
      ),
      'pr-curve': wandb.plot.pr_curve(
        y_true=y_true,
        y_probas=probs,
        labels=self.labels,
      ),
      'roc-curve': wandb.plot.roc_curve(
        y_true=y_true,
        y_probas=probs,
        labels=self.labels,
      )
    })

    self.model.history.history = history



class DeltaStopping(tf.keras.callbacks.Callback):
  def __init__(self):
    super(DeltaStopping, self).__init__()

  def on_epoch_end(self, epoch, logs=None):
    if (epoch > 2) and (logs['accuracy'] - logs['val_accuracy']) > 0.1:
      self.model.stop_training = True



class PruneCallback(tf.keras.callbacks.Callback):
  def __init__(self, trial: optuna.trial.FrozenTrial, objective_metric: str):
    super(PruneCallback, self).__init__()
    self.trial = trial
    self.objective_metric = objective_metric # default: "val_loss"


  def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
    self.trial.report(value=logs[self.objective_metric], step=epoch)

    if self.trial.should_prune():
      self.model.stop_training = True
      L.info(f'Trial pruned at epoch {epoch + 1}')



class TFKerasPruningCallback(tf.keras.callbacks.Callback):
  """tf.keras callback to prune unpromising trials.

  This callback is intend to be compatible for TensorFlow v1 and v2,
  but only tested with TensorFlow v2.

  See `the example <https://github.com/optuna/optuna-examples/blob/main/
  tfkeras/tfkeras_integration.py>`__
  if you want to add a pruning callback which observes the validation accuracy.

  Parameters
  ----------
    trial:
      A `optuna.trial.Trial` corresponding to the current evaluation of the
      objective function.
    monitor:
      An evaluation metric for pruning, e.g., ``val_loss`` or ``val_acc``.
  """

  def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
    super().__init__()
    self._trial = trial
    self._monitor = monitor

  def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
    logs = logs or {}
    current_score = logs.get(self._monitor)

    if current_score is None:
      L.error(f'Metric {self._monitor} not found by TFKerasPruningCallback')
      return

    # Report current score and epoch to Optuna's trial.
    self._trial.report(float(current_score), step=epoch)

    # Prune trial if needed
    if self._trial.should_prune():
      message = 'Trial was pruned at epoch {}'.format(epoch)
      raise optuna.TrialPruned(message)



class SaveBestTrialCallback(tf.keras.callbacks.Callback):
  def __init__(
    self,
    study: optuna.study.Study,
    objective_metric: str,
    objective_direction: str,
    name: str = 'model'
  ):
    super(SaveBestTrialCallback, self).__init__()
    self.name = name
    self.study = study
    self.objective_metric = objective_metric
    self.default_value = -np.inf if objective_direction == 'maximize' else np.inf
    self.operator = np.greater if objective_direction == 'maximize' else np.less


  def on_train_end(self, logs: Dict[str, Any]):
    try:
      best_value = self.study.best_value
    except:
      best_value = self.default_value

    current_value = logs[self.objective_metric]
    if self.operator(current_value, best_value):
      L.info(f'New best metric detected. {self.objective_metric}: {current_value}')

      # save model
      save_path = Experiment.local_exp_path / f'{self.name}.h5'
      self.model.save(save_path, overwrite=True)
      L.info(f'Trial saved in {str(save_path)}')
      Experiment.register_artifact(f'{self.name}.h5', 'gdrive')

      # save history as csv
      h = self.model.history.history
      if h:
        hist = {'epoch': range(len(h['loss'])), **h}
        hist_df = pd.DataFrame(hist)
        save_path = Experiment.local_exp_path / f'history_{self.name}.csv'
        hist_df.to_csv(save_path, index=False)
        L.info(f'History saved in {str(save_path)}')
        Experiment.register_artifact(f'history_{self.name}.csv', 'github')
