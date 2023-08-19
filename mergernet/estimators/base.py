from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Tuple, Union

import tensorflow as tf
import tensorflow_model_analysis as tfma

from mergernet.core.constants import RANDOM_SEED
from mergernet.core.experiment import Experiment
from mergernet.core.hp import HyperParameterSet
from mergernet.data.dataset import Dataset
from mergernet.model.utils import setup_seeds
from mergernet.services.google import GDrive

setup_seeds()


class EstimatorConfig:
  def __init__(
    self,
    name: str = None,
    url: List[str] = [],
    model_path: str = None,
    archive_path: str = None,
    fmt: str = None, # hdf5 or savedmodel
  ):
    self.name = name
    self.url = url
    self.model_path = model_path
    self.archive_path = archive_path
    self.fmt = fmt


class EstimatorRegistry:
  ZOOBOT_GREYSCALE = EstimatorConfig(
    name='zoobot_greyscale',
    url=[GDrive.get_url('1jAGWjuyOOyBBwYHwj07m2jr-gGFTlmdn')],
    model_path='zoobot_greyscale',
    archive_path='zoobot_grey.tar.xz',
    fmt='savedmodel',
  )

  ZOOBOT_RGB = EstimatorConfig(
    name='zoobot_rgb',
    url=[GDrive.get_url('1gVoWP9QFCJXgFn8cxt_Q5HMuqBjeuoIB')],
    model_path='zoobot_rgb',
    archive_path='zoobot_rgb.tar.xz',
    fmt='savedmodel',
  )


class Estimator(ABC):
  registry = EstimatorRegistry()

  def __init__(self, hp: HyperParameterSet, dataset: Dataset):
    self.dataset = dataset
    self.hp = hp
    self._tf_model = None


  @abstractmethod
  def build(self, freeze_conv: bool = False):
    raise NotImplementedError('Concret class must implement build method')


  @abstractmethod
  def train(self, run_name: str = '') -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    raise NotImplementedError('Concret class must implement train method')


  @abstractmethod
  def predict(self):
    raise NotImplementedError('Concret class must implement predict method')


  @property
  def tf_model(self):
    if self._tf_model is None:
      raise RuntimeError('Build model first calling build method')
    return self._tf_model


  def plot(self, filename: str = 'model.png'):
    tf.keras.utils.plot_model(
      self.tf_model,
      to_file=Experiment.local_exp_path / filename,
      expand_nested=True
    )


  def set_trainable(self, tf_model: tf.keras.Model, layer: str, trainable: bool):
    for l in tf_model.layers:
      if l.name == layer:
        l.trainable = trainable


  def compile_model(
    self,
    tf_model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    metrics: list = [],
    label_smoothing: float = 0.0
  ):
    hp_metrics = [self.get_metric(m) for m in self.hp.get('metrics', default=[]) if m is not None]
    tf_model.compile(
      optimizer=optimizer,
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing),
      metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        *hp_metrics,
        *metrics
      ]
    )


  def download(self, config: EstimatorConfig, replace: bool = False):
    if config is None: return
    destination_path = Experiment.local_exp_path / config.archive_path

    if not destination_path.exists() or replace:
      for i, url in enumerate(config.url):
        try:
          tf.keras.utils.get_file(
            fname=destination_path.resolve(),
            origin=url,
            cache_subdir=destination_path.parent.resolve(),
            extract=config.fmt == 'savedmodel'
          )
          break
        except:
          if i == len(self.config.url) - 1:
            raise RuntimeError("Can't download model")


  def get_dataaug_block(
    self,
    flip_horizontal: bool = True,
    flip_vertical: bool = True,
    rotation: Union[Tuple[float, float], bool] = (-0.08, 0.08),
    zoom: Union[Tuple[float, float], bool] = (-0.15, 0.0)
  ):
    data_aug_layers = []

    if flip_horizontal:
      data_aug_layers.append(
        tf.keras.layers.RandomFlip(mode='horizontal', seed=RANDOM_SEED)
      )

    if flip_vertical:
      data_aug_layers.append(
        tf.keras.layers.RandomFlip(mode='vertical', seed=RANDOM_SEED)
      )

    if rotation:
      data_aug_layers.append(
        tf.keras.layers.RandomRotation(
          rotation,
          fill_mode='reflect',
          interpolation='bilinear',
          seed=RANDOM_SEED
        )
      )

    if zoom:
      data_aug_layers.append(
        tf.keras.layers.RandomZoom(
          zoom,
          fill_mode='reflect',
          interpolation='bilinear',
          seed=RANDOM_SEED
        )
      )

    return tf.keras.Sequential(data_aug_layers, name='data_augmentation')


  def get_metric(self, metric: str):
    if metric == 'f1':
      return tf.keras.metrics.F1Score(name='f1')
    elif metric == 'tpr':
      return tfma.metrics.TPR(name='tpr', class_id=self.hp.get('positive_class_id'))
    elif metric == 'tnr':
      return tfma.metrics.TNR(name='tnr', class_id=self.hp.get('negative_class_id'))
    elif metric == 'fpr':
      return tfma.metrics.FPR(name='fpr', class_id=self.hp.get('positive_class_id'))
    elif metric == 'fnr':
      return tfma.metrics.FNR(name='fnr', class_id=self.hp.get('negative_class_id'))
    elif metric == 'precision':
      return tf.keras.metrics.Precision(name='precision')
    elif metric == 'recall':
      return tf.keras.metrics.Recall(name='recall')


  def get_optimizer(
    self,
    optimizer: str,
    lr: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]
  ) -> tf.keras.optimizers.Optimizer:
    if optimizer == 'adam':
      return tf.keras.optimizers.Adam(
        learning_rate=lr,
        weight_decay=self.hp.get('weight_decay')
      )
    elif optimizer == 'adamw':
      return tf.keras.optimizers.AdamW(
        learning_rate=lr,
        weight_decay=self.hp.get('weight_decay')
      )
    elif optimizer == 'lion':
      return tf.keras.optimizers.Lion(
        learning_rate=lr,
        weight_decay=self.hp.get('weight_decay')
      )
    elif optimizer == 'sgd':
      return tf.keras.optimizers.experimental.SGD(
        learning_rate=lr,
        weight_decay=self.hp.get('weight_decay'),
        nesterov=self.hp.get('nesterov', default=False),
        momentum=self.hp.get('momentum', default=0.0)
      )
    elif optimizer == 'nadam':
      return tf.keras.optimizers.experimental.Nadam(
        learning_rate=lr,
        weight_decay=self.hp.get('weight_decay')
      )
    elif optimizer == 'rmsprop':
      return tf.keras.optimizers.experimental.RMSprop(
        learning_rate=lr,
        weight_decay=self.hp.get('weight_decay'),
        momentum=self.hp.get('momentum', default=0.0)
      )


  def get_scheduler(self, scheduler: str, lr: float) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """
    For cosine_restarts scheduler, the learning rate multiplier first decays
    from 1 to alpha for first_decay_steps steps. Then, a warm restart is
    performed. Each new warm restart runs for t_mul times more steps and
    with m_mul times initial learning rate as the new learning rate.

    Parameters
    ----------
    scheduler : str
      Scheduler name
    lr : float
      Initial learning rate

    Returns
    -------
    tf.keras.optimizers.schedules.LearningRateSchedule
      A LearningRateSchedule instance
    """
    if scheduler == 'cosine_restarts':
      return tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=lr,
        first_decay_steps=self.hp.get('lr_decay_steps', default=40),
        t_mul=self.hp.get('lr_decay_t', default=2.0),
        m_mul=self.hp.get('lr_decay_m', default=1.0),
        alpha=self.hp.get('lr_decay_alpha', default=0.0),
      )
    elif scheduler == 'cosine':
      return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr,
        decay_steps=self.hp.get('lr_decay_steps', default=40),
        alpha=self.hp.get('lr_decay_alpha', default=0.0),
        warmup_target=self.get('lr_warmup_target', default=None),
        warmup_steps=self.get('lr_warmup_steps', default=0)
      )
    elif scheduler == 'exponential':
      return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=self.hp.get('lr_decay_steps', default=40),
        decay_rate=self.hp.get('lr_decay_rate', default=40),
      )
    else:
      return None


  def get_conv_arch(self, pretrained_arch: str) -> Tuple[Callable, Callable]:
    if pretrained_arch == 'xception':
      preprocess_input = tf.keras.applications.xception.preprocess_input
      base_model = tf.keras.applications.Xception
    elif pretrained_arch == 'vgg16':
      preprocess_input = tf.keras.applications.vgg16.preprocess_input
      base_model = tf.keras.applications.VGG16
    elif pretrained_arch == 'vgg19':
      preprocess_input = tf.keras.applications.vgg19.preprocess_input
      base_model = tf.keras.applications.VGG19
    elif pretrained_arch == 'resnet50':
      preprocess_input = tf.keras.applications.resnet.preprocess_input
      base_model = tf.keras.applications.ResNet50
    elif pretrained_arch == 'resnet101':
      preprocess_input = tf.keras.applications.resnet.preprocess_input
      base_model = tf.keras.applications.ResNet101
    elif pretrained_arch == 'resnet50v2':
      preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
      base_model = tf.keras.applications.ResNet50V2
    elif pretrained_arch == 'resnet101v2':
      preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
      base_model = tf.keras.applications.ResNet101V2
    elif pretrained_arch == 'inceptionv3':
      preprocess_input = tf.keras.applications.inception_v3.preprocess_input
      base_model = tf.keras.applications.InceptionV3
    elif pretrained_arch == 'inceptionresnetv2':
      preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
      base_model = tf.keras.applications.InceptionResNetV2
    elif pretrained_arch == 'densenet169':
      preprocess_input = tf.keras.applications.densenet.preprocess_input
      base_model = tf.keras.applications.DenseNet169
    elif pretrained_arch == 'densenet201':
      preprocess_input = tf.keras.applications.densenet.preprocess_input
      base_model = tf.keras.applications.DenseNet201
    elif pretrained_arch == 'efficientnetb0':
      preprocess_input = tf.keras.applications.efficientnet.preprocess_input
      base_model = tf.keras.applications.EfficientNetB0
    elif pretrained_arch == 'efficientnetb1':
      preprocess_input = tf.keras.applications.efficientnet.preprocess_input
      base_model = tf.keras.applications.EfficientNetB1
    elif pretrained_arch == 'efficientnetb2':
      preprocess_input = tf.keras.applications.efficientnet.preprocess_input
      base_model = tf.keras.applications.EfficientNetB2
    elif pretrained_arch == 'efficientnetb3':
      preprocess_input = tf.keras.applications.efficientnet.preprocess_input
      base_model = tf.keras.applications.EfficientNetB3
    elif pretrained_arch == 'efficientnetb4':
      preprocess_input = tf.keras.applications.efficientnet.preprocess_input
      base_model = tf.keras.applications.EfficientNetB4
    elif pretrained_arch == 'efficientnetb5':
      preprocess_input = tf.keras.applications.efficientnet.preprocess_input
      base_model = tf.keras.applications.EfficientNetB5
    elif pretrained_arch == 'efficientnetb6':
      preprocess_input = tf.keras.applications.efficientnet.preprocess_input
      base_model = tf.keras.applications.EfficientNetB6
    elif pretrained_arch == 'efficientnetb7':
      preprocess_input = tf.keras.applications.efficientnet.preprocess_input
      base_model = tf.keras.applications.EfficientNetB7
    elif pretrained_arch == 'efficientnetv2b0':
      preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input
      base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0
    elif pretrained_arch == 'efficientnetv2b1':
      preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input
      base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B1
    elif pretrained_arch == 'efficientnetv2b2':
      preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input
      base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B2
    elif pretrained_arch == 'convnext_tiny':
      preprocess_input = tf.keras.applications.convnext.preprocess_input
      base_model = tf.keras.applications.ConvNeXtTiny
    elif pretrained_arch == 'convnext_small':
      preprocess_input = tf.keras.applications.convnext.preprocess_input
      base_model = tf.keras.applications.ConvNeXtSmall
    elif pretrained_arch == 'convnext_base':
      preprocess_input = tf.keras.applications.convnext.preprocess_input
      base_model = tf.keras.applications.ConvNeXtBase
    return base_model, preprocess_input
