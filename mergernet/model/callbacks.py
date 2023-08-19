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



class MyWandbCallback(wandb.keras.WandbCallback):
  """
  `WandbCallback` automatically integrates keras with wandb.

  `WandbCallback` will automatically log history data from any metrics
  collected by keras: loss and anything passed into `keras_model.compile()`.
  `WandbCallback` will set summary metrics for the run associated with the
  "best" training step, where "best" is defined by the `monitor` and `mode`
  attributes.  This defaults to the epoch with the minimum `val_loss`.
  `WandbCallback` will by default save the model associated with the best
  `epoch`. `WandbCallback` can optionally log gradient and parameter histograms.
  `WandbCallback` can optionally save training and validation data for wandb
  to visualize.

  Parameters
  ----------
  monitor: str
    name of metric to monitor.  Defaults to `val_loss`.
  mode: str
    one of {`auto`, `min`, `max`}.
      `min` - save model when monitor is minimized
      `max` - save model when monitor is maximized
      `auto` - try to guess when to save the model (default).
  save_model:
    True - save a model when monitor beats all previous epochs
    False - don't save models
  save_graph: boolean
    if True save model graph to wandb (default to True).
  save_weights_only: boolean
    if True, then only the model's weights will be
    saved (`model.save_weights(filepath)`), else the full model
    is saved (`model.save(filepath)`).
  log_weights: boolean
    if True save histograms of the model's layer's weights.
  log_gradients: boolean
    if True log histograms of the training gradients
  training_data: tuple
    Same format `(X,y)` as passed to `model.fit`.  This is needed
    for calculating gradients - this is mandatory if `log_gradients` is `True`.
  validation_data: tuple
    Same format `(X,y)` as passed to `model.fit`.  A set of data for wandb to
    visualize. If this is set, every epoch, wandb will make a small number of
    predictions and save the results for later visualization. In case you are
    working with image data, please also set `input_type` and `output_type`
    in order to log correctly.
  generator: generator
    a generator that returns validation data for wandb to visualize.  This
    generator should return tuples `(X,y)`.  Either `validate_data` or generator
    should be set for wandb to visualize specific data examples. In case you are
    working with image data, please also set `input_type` and `output_type` in
    order to log correctly.
  validation_steps: int
    if `validation_data` is a generator, how many
    steps to run the generator for the full validation set.
  labels: list
    If you are visualizing your data with wandb this list of labels
    will convert numeric output to understandable string if you are building a
    multiclass classifier.  If you are making a binary classifier you can pass
    in a list of two labels ["label for false", "label for true"]. If
    `validate_data` and generator are both false, this won't do anything.
  predictions: int
    The number of predictions to make for visualization each epoch, max is 100.
  input_type: string
    type of the model input to help visualization. can be one of:
    (`image`, `images`, `segmentation_mask`, `auto`).
  output_type: string
    type of the model output to help visualization. can be one of:
    (`image`, `images`, `segmentation_mask`, `label`).
  log_evaluation: boolean
    if True, save a Table containing validation data and the model's
    predictions at each epoch. See `validation_indexes`,
    `validation_row_processor`, and `output_row_processor` for additional
    details.
  class_colors: [float, float, float]
    if the input or output is a segmentation mask,
    an array containing an rgb tuple (range 0-1) for each class.
  log_batch_frequency: integer
    if None, callback will log every epoch.
    If set to integer, callback will log training metrics every
    `log_batch_frequency` batches.
  log_best_prefix: string
    if None, no extra summary metrics will be saved.
    If set to a string, the monitored metric and epoch will be prepended with
    this value and stored as summary metrics.
  validation_indexes: [wandb.data_types._TableLinkMixin]
    an ordered list of index keys to associate
    with each validation example.  If log_evaluation is True and
    `validation_indexes` is provided, then a Table of validation data will not
    be created and instead each prediction will be associated with the row
    represented by the `TableLinkMixin`. The most common way to obtain such
    keys are is use `Table.get_index()` which will return a list of row keys.
  validation_row_processor: Callable
    a function to apply to the validation data, commonly used to visualize
    the data. The function will receive an `ndx` (int) and a `row` (dict).
    If your model has a single input, then `row["input"]` will be the input
    data for the row. Else, it will be keyed based on the name of the input
    slot. If your fit function takes a single target, then `row["target"]`
    will be the target data for the row. Else, it will be keyed based on the
    name of the output slots. For example, if your input data is a single
    ndarray, but you wish to visualize the data as an Image, then you can
    provide `lambda ndx, row: {"img": wandb.Image(row["input"])}` as the
    processor. Ignored if log_evaluation is False or `validation_indexes`
    are present.
  output_row_processor: Callable
    same as `validation_row_processor`, but applied to the model's output.
    `row["output"]` will contain the results of the model output.
  infer_missing_processors: bool
    Determines if `validation_row_processor` and `output_row_processor`
    should be inferred if missing. Defaults to True. If `labels` are provided,
    we will attempt to infer classification-type processors where appropriate.
  log_evaluation_frequency: int
    Determines the frequency which evaluation results will be logged.
    Default 0 (only at the end of training). Set to 1 to log every epoch,
    2 to log every other epoch, and so on. Has no effect when log_evaluation
    is False.
  compute_flops: bool
    Compute the FLOPs of your Keras Sequential or Functional model in GigaFLOPs
    unit.

  Examples
  --------
  ```python
  model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    callbacks=[WandbCallback()],
  )
  ```
  """
  def __init__(
    self,
    monitor: str = 'val_loss',
    mode: str = 'auto',
    log_gradients: bool = False,
    training_data: tf.data.Dataset = None,
    validation_data: tf.data.Dataset = None,
    labels: List[str] = [],
    predictions: int = 36,
    **kwargs,
  ):
    super().__init__(
      monitor=monitor,
      verbose=0,
      mode=mode,
      save_weights_only=False,
      log_weights=False,
      log_gradients=log_gradients,
      save_model=False,
      training_data=training_data,
      validation_data=validation_data,
      labels=labels,
      predictions=predictions,
      generator=None,
      input_type=None,
      output_type=None,
      log_evaluation=False,
      validation_steps=None,
      class_colors=None,
      log_batch_frequency=None,
      log_best_prefix='best_',
      save_graph=True,
      validation_indexes=None,
      validation_row_processor=None,
      prediction_row_processor=None,
      infer_missing_processors=True,
      log_evaluation_frequency=0,
      compute_flops=True,
    )

  def on_train_end(self, logs = None):
    super().on_train_end(logs)

    history = deepcopy(self.model.history.history)

    probs = self.model.predict(self.validation_data)
    y_true_one_hot = np.concatenate([y for _, y in self.validation_data], axis=0)
    y_true = np.argmax(y_true_one_hot, axis=-1)

    # print('probs')
    # print(probs)

    # print('y_true')
    # print(y_true)

    # print('class_names')
    # print(self.labels)

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


  def on_epoch_end(self, epoch: int, logs: dict = None):
    super().on_epoch_end(epoch, logs)

    lr = self.optimizer.get_config().get('learning_rate', None)
    if lr is not None:
      L.info(f'**Learning rate: {lr}')
      wandb.log({'computed_lr': str(lr)}, commit=False)



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



class MWandbCallback(tf.keras.callbacks.Callback):
  def __init__(self,
    validation_data: tf.data.Dataset,
    class_names: List[str],
    monitor: str,
    mode: str
  ):
    super().__init__()
    self.validation_data = validation_data
    self.class_names = class_names
    self.monitor = monitor
    self.mode = mode
    self._best_metric = -np.inf if mode.startswith('max') else np.inf
    self._monitor_op = np.greater if mode.startswith('max') else np.less
    self._graph_rendered = False

    # get the previous best metric for resumed runs
    previous_best = wandb.run.summary.get(f'best_{self.monitor}')
    if previous_best is not None:
      self._best = previous_best


  def on_train_begin(self, logs: dict = None):
    try:
      wandb.summary['GFLOPs'] = self.get_flops()
    except Exception as e:
      wandb.termwarn('Unable to compute FLOPs for this model.')


  def on_train_end(self, logs: dict = None):
    history = deepcopy(self.model.history.history)

    probs = self.model.predict(self.validation_data)
    y_true_one_hot = np.concatenate([y for _, y in self.validation_data], axis=0)
    y_true = np.argmax(y_true_one_hot, axis=-1)

    wandb.log({
      'confusion_matrix': wandb.plot.confusion_matrix(
        probs=probs,
        y_true=y_true,
        class_names=self.class_names,
      ),
      'pr-curve': wandb.plot.pr_curve(
        y_true=y_true,
        y_probas=probs,
        labels=self.class_names,
      ),
      'roc-curve': wandb.plot.roc_curve(
        y_true=y_true,
        y_probas=probs,
        labels=self.class_names,
      )
    })

    self.model.history.history = history


  def on_epoch_end(self, epoch: int, logs: dict = None):
    wandb.log({'epoch': epoch}, commit=False)
    wandb.log(deepcopy(logs), commit=True)

    current_metric = logs.get(self.monitor, None)
    if current_metric and self._monitor_op(current_metric, self._best_metric):
      self._best_metric = current_metric
      wandb.run.summary[f'best_{self.monitor}'] = self._best_metric
      wandb.run.summary['best_epoch'] = epoch


  def on_train_batch_end(self, batch, logs: dict = None):
    if not self._graph_rendered:
      # Couldn't do this in train_begin because keras may still not be built
      wandb.run.summary['graph'] = wandb.Graph.from_keras(self.model)
      self._graph_rendered = True


  def get_flops(self) -> float:
    """
    Calculate FLOPS [GFLOPs] for a keras model in inference mode.
    It uses tf.compat.v1.profiler under the hood.
    """
    if not hasattr(self, 'model'):
      raise wandb.Error('self.model must be set before using this method.')

    if not isinstance(
      self.model, (tf.keras.models.Sequential, tf.keras.models.Model)
    ):
      raise ValueError(
        'Calculating FLOPS is only supported for '
        '`tf.keras.Model` and `tf.keras.Sequential` instances.'
      )

    from tensorflow.python.framework.convert_to_constants import \
      convert_variables_to_constants_v2_as_graph

    # Compute FLOPs for one sample
    batch_size = 1
    inputs = [
      tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
      for inp in self.model.inputs
    ]

    # Convert tf.keras model into frozen graph to count FLOPs about operations
    # used at inference
    real_model = tf.function(self.model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPs with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = (
      tf.compat.v1.profiler.ProfileOptionBuilder(
        tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
      )
      .with_empty_output()
      .build()
    )

    flops = tf.compat.v1.profiler.profile(
      graph=frozen_func.graph, run_meta=run_meta, cmd='scope', options=opts
    )

    # Convert to GFLOPs
    return (flops.total_float_ops / 1e9) / 2



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
