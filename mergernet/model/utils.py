import logging
import os
import random
from pathlib import Path
from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from mergernet.core.constants import RANDOM_SEED
from mergernet.core.experiment import Experiment

L = logging.getLogger(__name__)



def setup_seeds():
  np.random.seed(RANDOM_SEED)
  random.seed(RANDOM_SEED)
  tf.random.set_seed(RANDOM_SEED)
  os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)



def set_trainable_state(
  model: tf.keras.Model,
  layer: str,
  trainable: bool
):
  for l in model.layers:
    if l.name == layer:
      l.trainable = trainable



def load_model(name: str, exp_id: int) -> tf.keras.Model:
  path = Experiment.download_file_gd(name, exp_id)
  L.info(f'Loading model {str(path)}')
  return tf.keras.models.load_model(path, compile=False)



def history_to_dataframe(history: Union[dict, None]) -> Union[pd.DataFrame, None]:
  if history:
    hist = {'epoch': range(len(history['loss'])), **history}
    hist_df = pd.DataFrame(hist)
    return hist_df
  return None



def get_conv_arch(
  pretrained_arch: str
) -> Tuple[Callable, Callable]:
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
