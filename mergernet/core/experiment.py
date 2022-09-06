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

from mergernet.core.constants import DATA_ROOT, ENV
from mergernet.core.utils import SingletonMeta, serialize
from mergernet.services.github import GithubService
from mergernet.services.google import GDrive

L = logging.getLogger(__name__)


DEV_LOCAL_SHARED_PATTERN = str(DATA_ROOT) + '/dev_workspace/shared_data/'
DEV_LOCAL_EXP_PATTERN = str(DATA_ROOT) + '/dev_workspace/experiments/{exp_id}/'
DEV_LOCAL_RUN_PATTERN = str(DATA_ROOT) + '/dev_workspace/experiments/{exp_id}/{run_id}/'
DEV_GH_EXP_PATTERN = 'dev_experiments/{exp_id}/'
DEV_GH_RUN_PATTERN = 'dev_experiments/{exp_id}/{run_id}/'
DEV_GD_EXP_PATTERN = str(DATA_ROOT) + '/dev_workspace/drive/MyDrive/mergernet/experiments/{exp_id}/'
DEV_GD_RUN_PATTERN = str(DATA_ROOT) + '/dev_workspace/drive/MyDrive/mergernet/experiments/{exp_id}/{run_id}/'

LOCAL_SHARED_PATTERN = 'shared_data/'
LOCAL_EXP_PATTERN = 'experiments/{exp_id}/'
LOCAL_RUN_PATTERN = 'experiments/{exp_id}/{run_id}/'
GH_EXP_PATTERN = 'experiments/{exp_id}/'
GH_RUN_PATTERN = 'experiments/{exp_id}/{run_id}/'
GD_EXP_PATTERN = 'drive/MyDrive/mergernet/experiments/{exp_id}/'
GD_RUN_PATTERN = 'drive/MyDrive/mergernet/experiments/{exp_id}/{run_id}/'



def experiment_run(exp_id: int):
  """
  Decorator function used to setup experiment

  Parameters
  ----------
  exp_id: int
    The human-readable experiment id
  """
  def decorator(func: FunctionType):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      # pre-execution operations:
      # 1. clear previous log
      log_path = Path(tempfile.gettempdir()) / 'mergernet.log'
      open(log_path, 'w').close()

      # 2. create new experiment
      exp_desc = getdoc(func)
      Experiment.create(exp_id=exp_id, exp_desc=exp_desc)

      # function execution:
      # track times and execute function
      start_time = int(time())
      func(*args, **kwargs)
      end_time = int(time())

      # post-execution oprations:
      # 1. upload all registered artifacts
      Experiment.upload_registered_artifacts()

      # 2. experiment metadata upload
      exp_metadata = {
        'exp_id': Experiment.exp_id,
        'exp_desc': Experiment.exp_desc
      }
      Experiment.upload_file_gh('exp_metadata.json', exp_metadata, scope='exp')

      # 3. logs upload
      Experiment.upload_file_gh(str(log_path))

      # 4. run metadata upload
      run_metadata = {
        'start_time': start_time,
        'end_time': end_time,
        'duration': end_time - start_time,
        'exp_id': Experiment.exp_id,
        'run_id': Experiment.run_id,
        'notes': Experiment.notes
      }
      Experiment.upload_file_gh('run_metadata.json', run_metadata)
    return wrapper
  return decorator



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



class Experiment(metaclass=SingletonMeta):
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

  run_id: str
    Hex token generated automatically that represents the identifier of a run.

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
  run_id = None
  local_shared_path = None
  local_run_path = None
  gh_run_path = None
  gd_run_path = None
  notes = None


  @classmethod
  def create(cls, exp_id: int, exp_desc: str = ''):
    """
    Creates a new experiment, configuring the experiment identifiers
    and file system to store the files needed (e.g. dataset) and the files
    generated by the experiment (e.g. predictions)

    Parameters
    ----------
    exp_id: int
      Experiment identifier, the same as entrypoint file
    exp_desc: str
      Experiment description
    """
    # setup ids
    cls.exp_desc = exp_desc
    cls.exp_id = exp_id
    cls.run_id = secrets.token_hex(4)

    # setup paths
    exp_params = dict(exp_id=cls.exp_id)
    run_params = dict(exp_id=cls.exp_id, run_id=cls.run_id)
    if ENV == 'dev':
      cls.local_shared_path = DEV_LOCAL_SHARED_PATTERN.format(**run_params)
      cls.local_exp_path = DEV_LOCAL_EXP_PATTERN.format(**exp_params)
      cls.local_run_path = DEV_LOCAL_RUN_PATTERN.format(**run_params)
      cls.gh_exp_path = DEV_GH_EXP_PATTERN.format(**exp_params)
      cls.gh_run_path = DEV_GH_RUN_PATTERN.format(**run_params)
      cls.gd_exp_path = DEV_GD_EXP_PATTERN.format(**exp_params)
      cls.gd_run_path = DEV_GD_RUN_PATTERN.format(**run_params)
    else:
      cls.local_shared_path = LOCAL_SHARED_PATTERN.format(**run_params)
      cls.local_exp_path = LOCAL_EXP_PATTERN.format(**exp_params)
      cls.local_run_path = LOCAL_RUN_PATTERN.format(**run_params)
      cls.gh_exp_path = GH_EXP_PATTERN.format(**exp_params)
      cls.gh_run_path = GH_RUN_PATTERN.format(**run_params)
      cls.gd_exp_path = GD_EXP_PATTERN.format(**exp_params)
      cls.gd_run_path = GD_RUN_PATTERN.format(**run_params)

    # prepare local experiment environment creating directory structure
    Path(cls.local_shared_path).mkdir(parents=True, exist_ok=True)
    Path(cls.local_run_path).mkdir(parents=True, exist_ok=True)
    Path(cls.gd_run_path).mkdir(parents=True, exist_ok=True)

    # signaling that this method was called
    cls._exp_created = True

    L.info(f'New experiment created with exp_id = {exp_id} and run_id = {cls.run_id}')


  @classmethod
  def upload_file_gh(cls, fname: str, data: Any = None, scope: str = 'run'):
    """
    Uploads a file to github artifacts repo inside `gh_artifact_path`

    Parameters
    ----------
    fname: str
      The file name. `fname` can be also an absolute path

    data: Any, optional
      The content of the file. If not specified, this method will consider
      the data of the file with same name as `fname` inside the
      `local_artifact_path` folder. If specified, it can be a json serializable
      python object or the bytes of the file.

    scope: str, optional
      The scope (or folder) that the artifact will be uploaded. Can be one of:
      ``run`` or ``exp``. The ``run`` scope uploads the file to the run
      folder and the ``exp`` scope uploads the file to the experiment folder
      (the parent of the run folder)
    """
    if not cls._exp_created: raise ValueError('Experiment must be created')

    from_path = Path(fname) if fname.startswith('/') else \
                Path(cls.local_run_path) / fname
    base_to_path = cls.gh_run_path if scope == 'run' else cls.gh_exp_path
    to_path = base_to_path + from_path.name

    gh = GithubService()

    try:
      if data is None:
        with open(from_path, 'rb') as fp:
          gh.commit(to_path, data=fp.read(), from_bytes=True)
      elif type(data) == str:
        gh.commit(to_path, data=data, from_bytes=False)
      elif isinstance(data, pd.DataFrame):
        buffer = BytesIO()
        data.to_csv(buffer, index=False)
        gh.commit(to_path, data=buffer.getbuffer(), from_bytes=True)
      else:
        gh.commit(to_path, data=serialize(data), from_bytes=False)
      L.info(f'Artifact {fname} was successfully uploaded to github')
    except:
      L.error(f'Error during artifact {fname} upload to github')


  @classmethod
  def download_file_gh(cls, fname: str, exp_id: int = None, run_id: str = None):
    """
    Downloads a file from github artifacts repo inside `gh_artifact_path`

    Parameters
    ----------
    fname: str
      The file name

    exp_id: int, optional
      The experiement identifier

    run_id: int, optional
      The run identifier
    """
    if not cls._exp_created: raise ValueError('Experiment must be created')
    exp_id = exp_id or cls.exp_id
    run_id = run_id or cls.run_id

    to_path = Path(cls.local_run_path) / fname
    from_path = GH_RUN_PATTERN.format(exp_id=exp_id, run_id=run_id)

    gh = GithubService()
    gh.download(remote_path=from_path, dest_path=to_path)


  @classmethod
  def upload_file_gd(cls, fname: str, data: Any = None, scope: str = 'run'):
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

    from_path = Path(cls.local_run_path) / fname
    base_to_path = cls.gd_run_path if scope == 'run' else cls.gd_exp_path
    to_path = Path(base_to_path) / fname

    try:
      if data is None:
        copy2(from_path, to_path)
      elif type(data) == str:
        with open(to_path, 'w') as fp:
          fp.write(data)
      else:
        with open(to_path, 'w') as fp:
          fp.write(serialize(data))
      L.info(f'Artifact {fname} was successfully uploaded to google drive')
    except:
      L.error(f'Error during artifact {fname} upload to google drive')


  @classmethod
  def download_file_gd(cls, fname: str, exp_id: int = None, run_id: str = None):
    """
    Downloads a file from google drive inside `gd_artifact_path`

    Parameters
    ----------
    fname: str
      The file name

    exp_id: int, optional
      The experiement identifier

    run_id: int, optional
      The run identifier
    """
    if not cls._exp_created: raise ValueError('Experiment must be created')
    exp_id = exp_id or cls.exp_id
    run_id = run_id or cls.run_id

    to_path = Path(cls.local_run_path) / fname
    from_path = GD_RUN_PATTERN.format(exp_id=exp_id, run_id=run_id)

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
      if artifact['service'] in ('github', 'both'):
        cls.upload_file_gh(artifact['name'])


if __name__ == '__main__':
  print(DEV_LOCAL_RUN_PATTERN)
