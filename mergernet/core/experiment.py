import functools
import json
import secrets
import logging
from pathlib import Path
from typing import Any
from time import time

import tensorflow as tf

from mergernet.core.utils import SingletonMeta, serialize
from mergernet.core.constants import DATA_ROOT, ENV
from mergernet.services.google import GDrive
from mergernet.services.github import GithubService


L = logging.getLogger(__name__)


DEV_LOCAL_SHARED_PATTERN = str(DATA_ROOT) + '/dev_workspace/shared_data/'
DEV_LOCAL_ARTIFACT_PATTERN = str(DATA_ROOT) + '/dev_workspace/experiments/{exp_id}/{run_id}/'
DEV_GH_ARTIFACT_PATTERN = 'dev_experiments/{exp_id}/{run_id}/'
DEV_GD_ARTIFACT_PATTERN = str(DATA_ROOT) + '/dev_workspace/drive/MyDrive/mergernet/experiments/{exp_id}/{run_id}/'

LOCAL_SHARED_PATTERN = 'shared_data/'
LOCAL_ARTIFACT_PATTERN = 'experiments/{exp_id}/{run_id}/'
GH_ARTIFACT_PATTERN = 'experiments/{exp_id}/{run_id}/'
GD_ARTIFACT_PATTERN = 'drive/MyDrive/mergernet/experiments/{exp_id}/{run_id}/'



def experiment_run(exp_id: int):
  """
  Decorator function used to setup experiment

  Parameters
  ----------
  exp_id: int
    The human-readable experiment id
  """
  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      # clear previous log
      log_path = Path('/tmp/mergernet.log')
      open(log_path, 'w').close()

      # create new experiment
      Experiment.create(exp_id)

      # track times and execute function
      start_time = int(time())
      func(*args, **kwargs)
      end_time = int(time())

      # post-execution oprations: logs and metadata upload
      metadata = {
        'start_time': start_time,
        'end_time': end_time,
        'duration': end_time - start_time,
        'exp_id': Experiment.exp_id,
        'run_id': Experiment.run_id,
        'notes': Experiment.notes
      }
      Experiment.upload_file_gh(str(log_path))
      Experiment.upload_file_gh('metadata.json', metadata)
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
  if save_history:
    history = model.history.history
    Experiment.upload_file_gh('history.json', history)

  if save_test_preds:
    _, ds_test = dataset.get_fold(0)
    ds_test = dataset.prepare_data(ds_test, kind='pred')
    X = dataset.get_X_by_fold(0, kind='test')
    test_preds = model.predict(ds_test)
    test_preds = {'X': X.tolist(), 'preds': test_preds.tolist()}
    Experiment.upload_file_gh('test_preds.json', test_preds)

  if save_dataset_config:
    Experiment.upload_file_gh('dataset_config.json', dataset.config.__dict__)

  if save_model:
    model.save(Path(Experiment.gd_artifact_path) / 'model.h5')



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
  exp_id = None
  run_id = None
  local_shared_path = None
  local_artifact_path = None
  gh_artifact_path = None
  gd_artifact_path = None
  notes = None


  @classmethod
  def create(cls, exp_id: int):
    """
    Creates a new experiment, configuring the experiment identifiers
    and file system to store the files needed (e.g. dataset) and the files
    generated by the experiment (e.g. predictions)

    Parameters
    ----------
    exp_id: int
      Experiment identifier, the same as entrypoint file
    """
    cls.exp_id = exp_id
    cls.run_id = secrets.token_hex(4)
    params = dict(exp_id=cls.exp_id, run_id=cls.run_id)
    if ENV == 'dev':
      cls.local_shared_path = DEV_LOCAL_SHARED_PATTERN.format(**params)
      cls.local_artifact_path = DEV_LOCAL_ARTIFACT_PATTERN.format(**params)
      cls.gh_artifact_path = DEV_GH_ARTIFACT_PATTERN.format(**params)
      cls.gd_artifact_path = DEV_GD_ARTIFACT_PATTERN.format(**params)
    else:
      cls.local_shared_path = LOCAL_SHARED_PATTERN.format(**params)
      cls.local_artifact_path = LOCAL_ARTIFACT_PATTERN.format(**params)
      cls.gh_artifact_path = GH_ARTIFACT_PATTERN.format(**params)
      cls.gd_artifact_path = GD_ARTIFACT_PATTERN.format(**params)
    Path(cls.local_shared_path).mkdir(parents=True, exist_ok=True)
    Path(cls.local_artifact_path).mkdir(parents=True, exist_ok=True)
    Path(cls.gd_artifact_path).mkdir(parents=True, exist_ok=True)
    cls._exp_created = True


  @classmethod
  def upload_file_gh(cls, fname: str, data: Any = None):
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
    """
    if not cls._exp_created: raise ValueError('Experiment must be created')
    from_path = Path(fname) if fname.startswith('/') else \
                Path(cls.local_artifact_path) / fname
    to_path = cls.gh_artifact_path + from_path.name
    gh = GithubService()

    if data is None:
      with open(from_path, 'rb') as fp:
        gh.commit(to_path, data=fp.read(), from_bytes=True)
    elif type(data) == str:
      gh.commit(to_path, data=data, from_bytes=False)
    else:
      gh.commit(to_path, data=serialize(data), from_bytes=False)


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

    to_path = Path(cls.local_artifact_path) / fname
    from_path = GH_ARTIFACT_PATTERN.format(exp_id=exp_id, run_id=run_id)

    gh = GithubService()
    gh.download(remote_path=from_path, dest_path=to_path)


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
    from_path = Path(cls.local_artifact_path) / fname
    to_path = Path(cls.gd_artifact_path) / fname
    gd = GDrive()

    if data is None:
      gd.send(from_path, to_path)
    elif type(data) == str:
      with open(to_path, 'w') as fp:
        fp.write(data)
    else:
      with open(to_path, 'w') as fp:
        fp.write(serialize(data))



if __name__ == '__main__':
  print(DEV_LOCAL_ARTIFACT_PATTERN)
