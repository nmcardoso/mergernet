import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from zoobot.shared import compress_representations, label_metadata
from zoobot.tensorflow.data_utils import image_datasets
from zoobot.tensorflow.estimators import define_model, preprocess
from zoobot.tensorflow.predictions import predict_on_dataset

from mergernet.core.constants import RANDOM_SEED
from mergernet.core.experiment import Experiment
from mergernet.core.hp import HyperParameterSet
from mergernet.core.utils import save_table
from mergernet.data.dataset import Dataset
from mergernet.estimators.base import Estimator, EstimatorConfig

L = logging.getLogger(__name__)


class ZoobotEstimator(Estimator):
  def __init__(
    self,
    hp: HyperParameterSet,
    dataset: Dataset,
    config: EstimatorConfig,
    predict_only: bool = True,
    crop_size: int = None,
    resize_size: int = None,
  ):
    super().__init__(hp, dataset)
    self.config = config
    self.predict_only = predict_only
    self.crop_size = crop_size
    self.resize_size = resize_size
    self.zoobot_dataset = None


  def _prepare_dataset(self):
    iaunames = self.dataset.get_X()
    paths = self.dataset.get_images_paths(iaunames)
    raw_image_ds = image_datasets.get_image_dataset(
      image_paths=[str(p.resolve()) for p in paths],
      file_format=self.dataset.config.image_extension,
      requested_img_size=self.dataset.config.image_shape[1],
      batch_size=128,
    )
    preprocessing_config = preprocess.PreprocessingConfig(
      label_cols=[],  # no labels are needed in predictions
      input_size=self.dataset.config.image_shape[1],
      make_greyscale=True,
      normalise_from_uint8=True  # False for tfrecords with 0-1 floats, True for png/jpg with 0-255 uints
    )
    self.zoobot_dataset = preprocess.preprocess_dataset(
      raw_image_ds,
      preprocessing_config
    )


  def build(self, include_top: bool = True) -> tf.keras.Model:
    self.download(self.config)

    self._tf_model = define_model.get_model(
      output_dim=len(label_metadata.decals_label_cols),
      weights_loc=Experiment.local_exp_path / self.config.model_path / 'checkpoint',
      include_top=include_top,
      input_size=self.dataset.config.image_shape[1],
      crop_size=self.crop_size,
      resize_size=self.resize_size,
      expect_partial=self.predict_only
    )

    if not include_top:
      self._tf_model.add(tf.keras.layers.GlobalAveragePooling2D())

    return self._tf_model


  def train(self):
    pass


  def predict(
    self,
    n_samples: int = 5,
    filename: Union[str, Path] = None,
    rebuild: bool = False,
  ):
    if self.zoobot_dataset is None:
      self._prepare_dataset()

    if self._tf_model is None or rebuild:
      self.build()

    predict_on_dataset.predict(
      self.zoobot_dataset,
      self.tf_model,
      n_samples,
      label_metadata.decals_label_cols,
      str((Experiment.local_exp_path / filename).resolve())
    )


  def cnn_representations(
    self,
    filename: Union[str, Path] = None,
    rebuild: bool = False,
    include_cols: Dict[str, List[Any]] = None,
  ):
    if self.zoobot_dataset is None:
      self._prepare_dataset()

    self.build(include_top=False)

    preds = self.tf_model.predict(self.zoobot_dataset)

    columns = [f'feat_{i}' for i in range(len(preds[0]))]
    df = pd.DataFrame(preds, columns=columns)
    df.insert(0, 'iauname', self.dataset.get_X())

    if include_cols is not None:
      for i, (col_name, col_data) in enumerate(include_cols.items()):
        df.insert(i + 1, col_name, col_data)

    save_table(df, Experiment.local_exp_path / filename, default=False)


  def pca(
    self,
    features: np.ndarray,
    n_components: int,
    filename: str = None,
    include_iauname: bool = True,
  ):
    df = compress_representations.create_pca_embedding(features, n_components)

    if include_iauname:
      df.insert(0, 'iauname', self.dataset.get_X())

    if filename is not None:
      save_table(df, Experiment.local_exp_path / filename, default=False)

    return df
