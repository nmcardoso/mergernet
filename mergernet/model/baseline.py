import os
from pathlib import Path
import random
from typing import Tuple
from mergernet.core.constants import RANDOM_SEED

from mergernet.core.dataset import Dataset
from mergernet.model.preprocessing import load_rgb, one_hot

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_addons.optimizers import RectifiedAdam


np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)



class ConvolutionalClassifier:
  def __init__(self, dataset: Dataset):
    self.dataset = dataset


  def compile_model(
    self,
    dense_layers: None,
    pretrained_weights: str = 'imagenet',
    pretrained_arch: str = 'inception',
    input_shape: Tuple = (128, 128, 3),
    optimizer: str = 'adam',
    learning_rate: float = 1e-4,
    data_aug: Tuple = ('flip', 'rotation'),
    verbose: bool = True
  ):
    IMAGE_SIZE = input_shape[:2]

    if pretrained_arch == 'inception':
      preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
      base_model = tf.keras.applications.InceptionResNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=pretrained_weights
      )
    elif pretrained_arch == 'vgg16':
      preprocess_input = tf.keras.applications.vgg16.preprocess_input
      base_model = tf.keras.applications.VGG16(
        input_shape=input_shape,
        include_top=False,
        weights=pretrained_weights
      )
    elif pretrained_arch == 'vgg19':
      preprocess_input = tf.keras.applications.vgg19.preprocess_input
      base_model = tf.keras.applications.VGG19(
        input_shape=input_shape,
        include_top=False,
        weights=pretrained_weights
      )
    elif pretrained_arch == 'efficientnetb2':
      preprocess_input = tf.keras.applications.efficientnet.preprocess_input
      base_model = tf.keras.applications.EfficientNetB2(
        input_shape=input_shape,
        include_top=False,
        weights=pretrained_weights
      )
    elif pretrained_arch == 'efficientnetb7':
      preprocess_input = tf.keras.applications.efficientnet.preprocess_input
      base_model = tf.keras.applications.EfficientNetB7(
        input_shape=input_shape,
        include_top=False,
        weights=pretrained_weights
      )
    elif pretrained_arch in ['densenet', 'densenet201']:
      preprocess_input = tf.keras.applications.densenet.preprocess_input
      base_model = tf.keras.applications.DenseNet201(
        input_shape=input_shape,
        include_top=False,
        weights=pretrained_weights
      )
    elif pretrained_arch == 'densenet169':
      preprocess_input = tf.keras.applications.densenet.preprocess_input
      base_model = tf.keras.applications.DenseNet169(
        input_shape=input_shape,
        include_top=False,
        weights=pretrained_weights
      )


    data_aug_layers = [
      tf.keras.layers.RandomFlip(mode='horizontal', seed=42),
      tf.keras.layers.RandomFlip(mode='vertical', seed=42)
    ]
    if 'rotation' in data_aug:
      data_aug_layers.append(
        tf.keras.layers.RandomRotation(
          (-0.08, 0.08),
          fill_mode='reflect',
          interpolation='bilinear',
          seed=42
        )
      )
    if 'zoom' in data_aug:
      data_aug_layers.append(
        tf.keras.layers.RandomZoom(
          (-0.15, 0.0),
          fill_mode='reflect',
          interpolation='bilinear',
          seed=42
        )
      )
    data_augmentation = tf.keras.Sequential(data_aug_layers, name='data_augmentation')


    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    if dense_layers is None:
      x = tf.keras.layers.GlobalAveragePooling2D()(x)
      x = tf.keras.layers.Dropout(0.4)(x)
    else:
      x = tf.keras.layers.Flatten()(x)
      x = dense_layers(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    if optimizer == 'std_adam':
      optimizer = tf.keras.optimizers.Adam(learning_rate)
    elif optimizer == 'radam':
      optimizer = RectifiedAdam(learning_rate)
    elif optimizer == 'nadam':
      optimizer = tf.keras.optimizers.Nadam(learning_rate)
    elif optimizer == 'rmsprop':
      optimizer = tf.keras.optimizers.RMSprop(learning_rate)

    model.compile(
      optimizer=optimizer,
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
      metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(curve='ROC', name='AUC_ROC'),
        tf.keras.metrics.AUC(curve='PR', name='AUC_PR')
      ]
    )

    return model



  def train(
    self,
    data_aug: Tuple = ('flip', 'rotation'),
    train_epochs: int = 12,
    sampling: str = None,
    dense_layers=None,
    pretrained_weights: str = 'imagenet',
    pretrained_arch: str = 'inception_v4',
    input_shape: Tuple = (128, 128, 3),
    batch_size: str = 32,
    learning_rate: float = 1e-4,
    optimizer: str = 'rmsprop',
    save_plot_path: Path = None,
    verbose: bool = True
  ):
    IMAGE_SIZE = input_shape[:2]

    tf.keras.backend.clear_session()

    model = self.compile_model(
      dense_layers=dense_layers,
      pretrained_weights=pretrained_weights,
      pretrained_arch=pretrained_arch,
      input_shape=input_shape,
      optimizer=optimizer,
      learning_rate=learning_rate,
      data_aug=data_aug,
      verbose=verbose,
      pretrained_trainable=False
    )

    ds_train, ds_test = self.dataset.get_fold(0)

    ds_train = ds_train.map(load_rgb(self.dataset.config.images_path))
    ds_train = ds_train.map(one_hot)
    # ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(5000)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # class_weights = self.dataset.compute_class_weight() if sampling == 'class_weight' else None

    print(f'>>> Training with {pretrained_arch}')

    history = model.fit(
      ds_train,
      batch_size=batch_size,
      epochs=train_epochs,
      validation_data=ds_test,
      # class_weight=class_weights,
      verbose=int(verbose)
    )

    if save_plot_path is not None or verbose:
      h = history.history
      plt.figure(figsize=(7, 4.2))
      plt.plot(h['accuracy'], label='Training Accuracy')
      plt.plot(h['val_accuracy'], label='Validation Accuracy')
      plt.plot(h['loss'], label='Training Loss')
      plt.plot(h['val_loss'], label='Validation Loss')
      plt.ylim([max([0, min(plt.ylim())]), min([1, max(plt.ylim())])])
      plt.xlabel('epoch')
      plt.ylabel('metric')
      plt.legend(ncol=2).get_frame().set_alpha(0.75)
      if save_plot_path is not None:
        plt.savefig(save_plot_path, bbox_inches='tight', pad_inches=0.01)
      if verbose:
        plt.show()
      plt.clf()
      plt.close()

    return (model, history)



class Metamodel:
  pass
