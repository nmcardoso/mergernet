import logging
import secrets

import optuna
import mlflow
import tensorflow as tf
from mergernet.core.artifacts import ArtifactHelper
from optuna.integration.mlflow import MLflowCallback

from mergernet.core.dataset import Dataset
from mergernet.model.callback import DeltaStopping
from mergernet.model.preprocessing import load_jpg, one_hot
from mergernet.core.utils import Timming


L = logging.getLogger('job')

# this implementation mimics the mlflow integration of optuna
# `RUN_ID_ATTRIBUTE_KEY` was extracted from following code:
# https://github.com/optuna/optuna/blob/master/optuna/integration/mlflow.py
RUN_ID_ATTRIBUTE_KEY = 'mlflow_run_id'



class HyperModel:
  def __init__(
    self,
    dataset: Dataset,
    optuna_uri: str = 'sqlite:///optuna.sqlite',
    mlflow_uri: str = 'sqlite:///mlflow.sqlite',
    nest_trials: bool = False
  ):
    ah = ArtifactHelper()
    self.optuna_uri = optuna_uri or ah.optuna_uri
    self.mlflow_uri = mlflow_uri or ah.mlflow_uri
    self.dataset = dataset
    self.nest_trials = nest_trials
    self.nest_trials = True

    self.mlflow = MLflowCallback(
      tracking_uri=self.mlflow_uri,
      metric_name='optuna_score',
      nest_trials=self.nest_trials,
      tag_study_user_attrs=False
    )


  def prepare_data(self, dataset):
    ds_train, ds_test = dataset.get_fold(0)
    L.info('[DATASET] Fold 0 loaded')

    if not dataset.in_memory:
      ds_train = ds_train.map(load_jpg)
      ds_test = ds_test.map(load_jpg)
      L.info('[DATASET] apply: load_jpg')

    ds_train = ds_train.map(one_hot)
    ds_test = ds_test.map(one_hot)
    L.info('[DATASET] apply: one_hot')

    ds_train = ds_train.cache()
    ds_test = ds_test.cache()

    ds_train = ds_train.shuffle(5000)
    ds_test = ds_test.shuffle(1000)
    L.info('[DATASET] apply: shuffle')

    ds_train = ds_train.batch(64)
    ds_test = ds_test.batch(64)
    L.info('[DATASET] apply: batch')

    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    ds_train = ds_train
    ds_test = ds_test
    class_weights = dataset.compute_class_weight()

    return ds_train, ds_test, class_weights



  def build_model(self, trial, input_shape, pretrained_weights):
    conv_block = tf.keras.applications.ResNet50(
      input_shape=input_shape,
      include_top=False,
      weights=pretrained_weights
    )

    preprocess_input = tf.keras.applications.resnet.preprocess_input

    data_aug_layers = [
      tf.keras.layers.RandomFlip(mode='horizontal', seed=42),
      tf.keras.layers.RandomFlip(mode='vertical', seed=42),
      tf.keras.layers.RandomRotation(
        (-0.08, 0.08),
        fill_mode='reflect',
        interpolation='bilinear',
        seed=42
      ),
      tf.keras.layers.RandomZoom(
        (-0.15, 0.0),
        fill_mode='reflect',
        interpolation='bilinear',
        seed=42
      )
    ]

    data_aug_block = tf.keras.Sequential(data_aug_layers, name='data_augmentation')

    inputs = tf.keras.Input(shape=input_shape)
    x = data_aug_block(inputs)
    x = preprocess_input(x)
    x = conv_block(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(trial.suggest_categorical('dense_units_1', [64, 128, 256, 512, 1024]), activation='relu')(x)
    x = tf.keras.layers.Dropout(trial.suggest_uniform('dropout_rate_1', 0.1, 0.5))(x)
    x = tf.keras.layers.Dense(trial.suggest_categorical('dense_units_2', [64, 128, 256, 512, 1024]), activation='relu')(x)
    x = tf.keras.layers.Dropout(trial.suggest_uniform('dropout_rate_2', 0.1, 0.5))(x)

    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
      optimizer=tf.keras.optimizers.Adam(
        trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
      ),
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
      metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='accuracy')
      ]
    )
    return model


  def objective(self, trial):
    tf.keras.backend.clear_session()

    batch_size = 64
    epochs=3

    ds_train, ds_test, class_weights = self.prepare_data(self.dataset)

    model = self.build_model(
      trial,
      input_shape=(128, 128, 3),
      pretrained_weights='imagenet',
    )

    t = Timming()
    t.start()
    L.info('[TRAIN] Starting training loop')

    ah = ArtifactHelper()

    history = model.fit(
      ds_train,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=ds_test,
      class_weight=class_weights,
      callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3),
        DeltaStopping()
      ]
    )

    t.end()
    L.info(f'[TRAIN] Training finished without errors in {t.duration()}.')


    with mlflow.start_run(run_name=str(trial.number), nested=self.nest_trials) as run:
      trial.set_system_attr(RUN_ID_ATTRIBUTE_KEY, run.info.run_id)

      h = history.history
      epochs = len(h[list(h.keys())[0]])

      for i in range(epochs):
        metrics = {}
        for name in h.keys():
          metrics[name] = h[name][i]
        run.log_metrics(metrics)

    ev = model.evaluate(ds_test)
    idx = model.metrics_names.index('accuracy')
    return ev[idx]



  def hypertrain(self):
    study = optuna.create_study(storage='sqlite:///optuna.sqlite', study_name='test', direction='maximize')
    study.optimize(self.objective, n_trials=2, callbacks=[self.mlflow])

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
      print('    {}: {}'.format(key, value))