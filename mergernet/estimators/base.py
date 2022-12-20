from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Tuple, Union

import tensorflow as tf

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
  ZOOBOT_GREY = EstimatorConfig(
    name='zoobot_grey',
    url=[GDrive.get_url('1jAGWjuyOOyBBwYHwj07m2jr-gGFTlmdn')],
    model_path='zoobot_grey',
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


  def set_trainable(self, tf_model: tf.keras.Model, layer: str, trainable: bool):
    for l in tf_model.layers:
      if l.name == layer:
        l.trainable = trainable


  def compile_model(
    self,
    tf_model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    metrics: list = []
  ):
    tf_model.compile(
      optimizer=optimizer,
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
      metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        *metrics
      ]
    )


  def download(self, config: EstimatorConfig, replace: bool = False):
    if config is None: return
    destination_path = Path(Experiment.local_exp_path) / config.model_path

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
            raise RuntimeError("Can't download images archive")


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
    return base_model, preprocess_input
