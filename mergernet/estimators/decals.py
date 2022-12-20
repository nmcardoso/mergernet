import logging
from pathlib import Path
from typing import Union

import tensorflow as tf
from zoobot.shared import label_metadata
from zoobot.tensorflow.data_utils import image_datasets
from zoobot.tensorflow.estimators import define_model, preprocess
from zoobot.tensorflow.predictions import predict_on_dataset

from mergernet.core.constants import RANDOM_SEED
from mergernet.core.experiment import Experiment
from mergernet.core.hp import HyperParameterSet
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
  ):
    super().__init__(hp, dataset)
    self.config = config
    self.predict_only = predict_only
    self.zoobot_dataset = None


  def _prepare_dataset(self):
    paths = self.dataset.get_images_paths()
    raw_image_ds = image_datasets.get_image_dataset(
      image_paths=[str(p.resolve()) for p in paths],
      file_format=self.dataset.config.X_column_suffix[1:],
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


  def build(self, crop_size: int, resize_size: int) -> tf.keras.Model:
    self.download(self.config)

    self._tf_model = define_model.get_model(
      checkpoint_loc=Path(Experiment.local_exp_path) / self.config.model_path,
      include_top=True,
      input_size=self.dataset.config.image_shape[1],
      crop_size=crop_size,
      resize_size=resize_size,
      expect_partial=self.predict_only
    )

    return self._tf_model


  def train(self):
    pass


  def predict(self, n_samples: int = 5, output_path: Union[str, Path] = None):
    if self.zoobot_dataset is None:
      self._prepare_dataset()

    label_cols = label_metadata.decals_label_cols

    predict_on_dataset.predict(
      self.zoobot_dataset,
      self.tf_model,
      n_samples,
      label_cols,
      output_path
    )