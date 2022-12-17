import functools
import logging
import secrets
import tempfile
from inspect import getdoc
from io import BytesIO, StringIO
from pathlib import Path
from shutil import copy2
from time import time
from types import FunctionType
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb

from mergernet.core.constants import DATA_ROOT, ENV
from mergernet.core.utils import SingletonMeta, classproperty, serialize
from mergernet.services.github import GithubService
from mergernet.services.google import GDrive

L = logging.getLogger(__name__)


DEV_LOCAL_SHARED_PATTERN = str(DATA_ROOT) + '/dev_workspace/shared_data/'
DEV_LOCAL_EXP_PATTERN = str(DATA_ROOT) + '/dev_workspace/experiments/{exp_id}/'
DEV_GD_EXP_PATTERN = str(DATA_ROOT) + '/dev_workspace/drive/MyDrive/mergernet/experiments/{exp_id}/'

LOCAL_SHARED_PATTERN = 'shared_data/'
LOCAL_EXP_PATTERN = 'experiments/{exp_id}/'
GH_EXP_PATTERN = 'experiments/{exp_id}/'
GD_EXP_PATTERN = 'drive/MyDrive/mergernet/experiments/{exp_id}/'



def backup_model(
  model: tf.keras.Model,
  dataset: Any,
  save_history: bool = True,
  save_test_preds: bool = True,
  save_dataset_config: bool = True,
  save_model: bool = True,
):
  """
  Parameters
  ----------
  model: tf.keras.Model
    The keras model to backup

  dataset: Dataset
    The dataset instance

  save_history: bool
    True if train history must be saved in github

  save_test_preds: bool
    True if the predictions in test dataset must be saved in github

  save_dataset_config: bool
    True if dataset_config must be saved in github

  save_model: bool
    True if tensorflow model must be saved in google drive
  """
  if save_dataset_config:
    Experiment.upload_file_gh('dataset_config.json', dataset.config.__dict__)

  if model is None:
    L.info('No model received. Nothing to backup')
    return

  if save_history:
    if model.history is not None:
      history = model.history.history
      Experiment.upload_file_gh('history.json', history)

  if save_test_preds:
    _, ds_test = dataset.get_fold(0)
    ds_test = dataset.prepare_data(ds_test, kind='pred')
    X = dataset.get_X_by_fold(0, kind='test')
    test_preds = model.predict(ds_test)

    df = pd.read_csv(dataset.config.table_path)
    x_col_name = dataset.config.X_column
    df = df.set_index(x_col_name).loc[X].reset_index(inplace=False)
    for label, index in dataset.config.label_map.items():
      y_hat = [pred[index] for pred in test_preds]
      df[f'prob_{label}'] = y_hat
    Experiment.upload_file_gh('test_preds_fold_0.csv', df)

  if save_model:
    model.save(Path(Experiment.gd_run_path) / 'model.h5')



class Experiment:
  """
  This class stores all relevant information for an experiment, which can be
  accessed from anywhere in the application as it's an implementation of the
  Singleton pattern.

  The experiments are tracked using two values: `exp_id` and `run_id`. The
  first value is a human-readable integer set by user and is related with
  a task specified in entrypoint file. And the second value is a auto-generated
  hex token used to track different re-runs of same experiment.

  Attributes
  ----------
  exp_id: int
    Identifier of the experiment set by user.

  local_shared_path: str
    The path in local environment where shared files (e.g. dataset) are stored.
    This path is shared to any run of any experiemnt.

  local_artifact_path: str
    The path in local environment where artifacts (outputs of an experiment,
    e.g. model predictions) are stored. This path is relative to a specific run.

  gh_artifact_path: str
    The path in Github artifacts repo where the artifacts of a experiment run
    will be uploaded to. This path is relative to a specific run.

  gd_artifact_path: str
    The path in Google Drive where the artifacts of a experiment run will be
    uploaded to. This path is relative to a specifict run.

  notes: str
    Notes for current run of this experiment
  """
  _exp_created = False
  _registered_artifacts = []
  exp_id = None
  exp_name = None
  local_shared_path = None
  local_exp_path = None
  gd_exp_path = None
  notes = None


  def _pre_run(self):
    """
    Pre-run tasks automatically performed when `run` is called, that includes:

    - clear log file
    - setup experiment global attributes

    See Also
    --------
    mergernet.core.experiment._setup_experiment_attributes
    """
    # clear previous log
    self.log_path = Path(tempfile.gettempdir()) / 'mergernet.log'
    open(self.log_path, 'w').close()

    # configure experiment attributes
    self._setup_experiment_attributes()


  def _post_run(self):
    """
    Post-run tasks automatically performed when `run` is called, that includes:

    - upload registered artifacts

    See Also
    --------
    mergernet.core.experiment.upload_registered_artifacts
    """
    # upload all registered artifacts
    Experiment.upload_registered_artifacts()


  def run(self):
    """
    Run the experiement defined in the abstract `call` method executing
    pre-run and post-run tasks

    See Also
    --------
    mergernet.core.experiment._pre_run,
    mergernet.core.experiment._post_run
    """
    self._pre_run()
    self.call()
    self._post_run()


  def _setup_experiment_attributes(self):
    """
    Configure the experiment identifiers and file system to store
    the files needed (e.g. dataset) and the files
    generated by the experiment (e.g. models, dataset, logs, predictions)
    """
    # setup ids
    Experiment.exp_id = self.exp_id
    Experiment.exp_name = f'exp_{self.exp_id}'

    Experiment.log_wandb = self.log_wandb

    # setup paths
    exp_params = dict(exp_id=Experiment.exp_id)
    if ENV == 'dev':
      Experiment.local_shared_path = DEV_LOCAL_SHARED_PATTERN.format(**exp_params)
      Experiment.local_exp_path = DEV_LOCAL_EXP_PATTERN.format(**exp_params)
      Experiment.gd_exp_path = DEV_GD_EXP_PATTERN.format(**exp_params)
    else:
      Experiment.local_shared_path = LOCAL_SHARED_PATTERN.format(**exp_params)
      Experiment.local_exp_path = LOCAL_EXP_PATTERN.format(**exp_params)
      Experiment.gd_exp_path = GD_EXP_PATTERN.format(**exp_params)

    # prepare local experiment environment creating directory structure
    Path(Experiment.local_shared_path).mkdir(parents=True, exist_ok=True)
    Path(Experiment.local_exp_path).mkdir(parents=True, exist_ok=True)
    Path(Experiment.gd_exp_path).mkdir(parents=True, exist_ok=True)

    # signaling that this method was called
    Experiment._exp_created = True

    L.info(f'New experiment created with exp_id = {self.exp_id}')


  @classmethod
  def upload_file_gd(cls, fname: str, data: Any = None):
    """
    Uploads a file to google drive inside `gd_artifact_path`

    Parameters
    ----------
    fname: str
      The file name

    data: Any, optional
      The content of the file. If not specified, this method will consider
      the data of the file with same name as `fname` inside the
      `local_artifact_path` folder. If specified, it can be a json serializable
      python object or the bytes of the file.
    """
    if not cls._exp_created: raise ValueError('Experiment must be created')

    from_path = Path(cls.local_exp_path) / fname
    to_path = Path(cls.gd_exp_path) / fname

    try:
      if data is None:
        copy2(from_path, to_path)
      elif type(data) == str:
        with open(to_path, 'w') as fp:
          fp.write(data)
      elif isinstance(data, tf.keras.Model):
        data.save(to_path)
      else:
        with open(to_path, 'w') as fp:
          fp.write(serialize(data))
      L.info(f'Artifact {fname} was successfully uploaded to google drive')
    except:
      L.error(f'Error during artifact {fname} upload to google drive')


  @classmethod
  def download_file_gd(cls, fname: str, exp_id: int = None):
    """
    Downloads a file from google drive inside `gd_artifact_path`

    Parameters
    ----------
    fname: str
      The file name

    exp_id: int, optional
      The experiement identifier
    """
    if not cls._exp_created: raise ValueError('Experiment must be created')
    exp_id = exp_id or cls.exp_id

    from_path = Path(cls.gd_exp_path) / fname
    to_path = Path(cls.local_exp_path) / fname

    path = copy2(from_path, to_path)
    return Path(path)


  @classmethod
  def register_artifact(cls, fname: str, service: str):
    """
    Register an artifact that will be automatically uploaded to corresponding
    service at the end of the experiment

    Parameters
    ----------
    fname: str
      The artifact filename
    service: str
      The service name, one of: ``gdrive``, ``github`` or ``both``
    """
    if not cls._exp_created: raise ValueError('Experiment must be created')
    cls._registered_artifacts.append({
      'name': fname,
      'service': service
    })
    L.info(f'New artifact registered. name: {fname}, service: {service}')


  @classmethod
  def upload_registered_artifacts(cls):
    """
    Uploads all registered artifacts
    """
    for artifact in cls._registered_artifacts:
      if artifact['service'] in ('gdrive', 'both'):
        cls.upload_file_gd(artifact['name'])


  @classmethod
  def init_wandb(
    cls,
    config: dict = {},
    job_type: str = None,
    name: str = None,
    tags: list = []
  ):
    """
    Creates a wandb project

    Parameters
    ----------
    config: dict
      The configuration dict

    job_type: str
      The job type

    name: str
      The run name

    tags: list
      The run tags

    See Also
    --------
    mergernet.core.experiment.finish_wandb,
    mergernet.core.experiment.Tracer
    """
    wandb.init(
    project=cls.exp_name,
    entity='nmcardoso',
    config=config,
    save_code=False,
    job_type=job_type,
    tags=tags,
    name=name,
    notes=cls.notes
  )


  @classmethod
  def finish_wandb(cls):
    """
    Closes the connection with current wandb project
    """
    wandb.finish()


  class Tracer:
    """
    Context manager that creates an wandb project

    Parameters
    ----------
    config: dict
      The configuration dict

    job_type: str
      The job type

    name: str
      The run name

    tags: list
      The run tags

    See Also
    --------
    mergernet.core.experiment.init_wandb
    """
    def __init__(
      self,
      config: dict = {},
      job_type: str = None,
      name: str = 'run-0',
      tags: list = []
    ):
      self.config = config
      self.job_type = job_type
      self.name = name
      self.tags = tags

    def __enter__(self):
      Experiment.init_wandb(
        config=self.config,
        job_type=self.job_type,
        name=self.name,
        tags=self.tags
      )

    def __exit__(self):
      Experiment.finish_wandb()
