
import logging

from mergernet.core.constants import RANDOM_SEED
from mergernet.core.dataset import Dataset
from mergernet.core.artifacts import ArtifactHelper
from mergernet.core.utils import Timming
from mergernet.model.preprocessing import load_jpg, one_hot

import autokeras as ak
import tensorflow as tf



L = logging.getLogger('job')



class AutoKerasSimpleClassifier:
  def __init__(self, dataset: Dataset):
    self.dataset = dataset


  def fit(
    self,
    project_name: str,
    directory: str,
    max_trials: int = 1,
    tuner: str = None,
    overwrite: bool = False,
    batch_size: int = 64
  ):
    ds_train, ds_test = self.dataset.get_fold(0)

    ds_train = ds_train.map(load_jpg)
    ds_test = ds_test.map(load_jpg)

    ds_train = ds_train.map(one_hot)
    ds_test = ds_test.map(one_hot)

    ds_train = ds_train.shuffle(5000)
    ds_test = ds_test.shuffle(1000)

    ds_train = ds_train.batch(batch_size)
    ds_test = ds_test.batch(batch_size)

    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    L.info('[DATASET] dataset loaded')
    L.info(f'[BUILD] creating classifier {project_name} with following attributes:')
    L.info(f'[BUILD] directory: {directory}')
    L.info(f'[BUILD] max trials: {max_trials}')
    L.info(f'[BUILD] tuner: {tuner}')

    params = dict(
      num_classes=3,
      loss='categorical_crossentropy',
      project_name=project_name,
      directory=directory,
      max_trials=max_trials,
      overwrite=overwrite,
      seed=RANDOM_SEED
    )

    if tuner is not None: params.update(dict(tuner=tuner))

    clf = ak.ImageClassifier(**params)

    L.info(f'[TRAIN] starting train loop.')
    t = Timming()
    t.start()
    clf.fit(x = ds_train, validation_data=ds_test)
    t.end()
    L.info(f'[TRAIN] train loop finished in {t.duration()}.')

    ah = ArtifactHelper()

    L.info(f'[PRED] model predictions on test set.')
    preds = clf.predict(x=ds_test)
    ah.upload_json(preds, 'predictions.json')

    L.info(f'[EVAL] model evaluation.')
    scores = clf.evaluate(x=ds_test)
    ah.upload_json(preds, 'evaluation.json')

    L.info(f'[SAVE] exporting the best model.')
    model = clf.export_model()
    ah.upload_model(model)
