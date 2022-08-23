import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import tensorflow as tf
import optuna

from mergernet.core.experiment import Experiment


L = logging.getLogger(__name__)


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
      L.info(f'[PRUNER] trial pruned at epoch {epoch + 1}')


class SaveCallback(tf.keras.callbacks.Callback):
  def __init__(
    self,
    name: str,
    study: optuna.study.Study,
    objective_metric: str,
    objective_direction: str
  ):
    super(SaveCallback, self).__init__()
    self.name = name
    self.study = study
    self.objective_metric = objective_metric
    self.default_value = -np.inf if objective_direction == 'maximize' else np.inf
    self.operator = np.greater if objective_direction == 'maximine' else np.less


  def on_train_end(self, logs: Dict[str, Any]):
    try:
      best_value = self.study.best_value
    except:
      best_value = self.default_value

    if self.operator(logs[self.objective_metric], best_value):
      e = Experiment()
      save_path = Path(e.local_artifact_path) / (self.name + '.h5')
      if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
      self.model.save(save_path, overwrite=True)


class TelemetryCallback(tf.keras.callbacks.Callback):
  def __init__(self, trial: optuna.trial.FrozenTrial):
    super(TelemetryCallback, self).__init__()
    self.trial = trial

  def on_train_end(self, logs: Dict[str, Any]):
    pass
