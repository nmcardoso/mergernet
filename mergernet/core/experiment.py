import functools
import json
import secrets
import logging
from pathlib import Path
from typing import Any
from time import time

import tensorflow as tf

from mergernet.core.utils import SingletonMeta
from mergernet.core.constants import DATA_ROOT, ENV
from mergernet.core.dataset import Dataset
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
      e = Experiment()
      e.create(exp_id)

      # track times and execute function
      start_time = int(time())
      func(*args, **kwargs)
      end_time = int(time())

      # post-execution oprations: logs and metadata upload
      metadata = {
        'start_time': start_time,
        'end_time': end_time,
        'duration': end_time - start_time,
        'exp_id': e.exp_id,
        'run_id': e.run_id,
        'notes': e.notes
      }
      e.upload_file_gh(str(log_path))
      e.upload_file_gh('metadata.json', metadata)
    return wrapper
  return decorator



def backup_model(
  model: tf.keras.Model,
  dataset: Dataset,
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
  e = Experiment()

  if save_history:
    history = model.history.history
    e.upload_file_gh('history.json', history)

  if save_test_preds:
    _, ds_test = dataset.get_fold(0)
    ds_test = dataset.prepare_data(ds_test, kind='pred')
    X = dataset.get_X_by_fold(0, kind='test')
    test_preds = model.predict(ds_test)
    test_preds = {'X': X, 'preds': test_preds}
    e.upload_file_gh('test_preds.json', test_preds)

  if save_dataset_config:
    e.upload_file_gh('dataset_config.json', dataset.__dict__)

  if save_model:
    model.save(Path(e.gd_artifact_path) / 'model.h5')



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
  def __init__(self):
    self._exp_created = False
    self.exp_id = None
    self.run_id = None
    self.local_shared_path = None
    self.local_artifact_path = None
    self.gh_artifact_path = None
    self.gd_artifact_path = None
    self.notes = None


  def create(self, exp_id: int):
    """
    Creates a new experiment, configuring the experiment identifiers
    and file system to store the files needed (e.g. dataset) and the files
    generated by the experiment (e.g. predictions)

    Parameters
    ----------
    exp_id: int
      Experiment identifier, the same as entrypoint file
    """
    self.exp_id = exp_id
    self.run_id = secrets.token_hex(4)
    params = dict(exp_id=self.exp_id, run_id=self.run_id)
    if ENV == 'dev':
      self.local_shared_path = DEV_LOCAL_SHARED_PATTERN.format(**params)
      self.local_artifact_path = DEV_LOCAL_ARTIFACT_PATTERN.format(**params)
      self.gh_artifact_path = DEV_GH_ARTIFACT_PATTERN.format(**params)
      self.gd_artifact_path = DEV_GD_ARTIFACT_PATTERN.format(**params)
    else:
      self.local_shared_path = LOCAL_SHARED_PATTERN.format(**params)
      self.local_artifact_path = LOCAL_ARTIFACT_PATTERN.format(**params)
      self.gh_artifact_path = GH_ARTIFACT_PATTERN.format(**params)
      self.gd_artifact_path = GD_ARTIFACT_PATTERN.format(**params)
    Path(self.local_shared_path).mkdir(parents=True, exist_ok=True)
    Path(self.local_artifact_path).mkdir(parents=True, exist_ok=True)
    Path(self.gd_artifact_path).mkdir(parents=True, exist_ok=True)
    self._exp_created = True


  def upload_file_gh(self, fname: str, data: Any = None):
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
    if not self._exp_created: raise ValueError('Experiment must be created')
    from_path = Path(fname) if fname.startswith('/') else \
                Path(self.local_artifact_path) / fname
    to_path = self.gh_artifact_path + from_path.name
    gh = GithubService()

    if data is None:
      with open(from_path, 'rb') as fp:
        gh.commit(to_path, data=fp.read(), from_bytes=True)
    elif type(data) == str:
      gh.commit(to_path, data=data, from_bytes=False)
    else:
      gh.commit(to_path, data=json.dumps(data), from_bytes=False)


  def download_file_gh(self, fname: str, exp_id: int = None, run_id: str = None):
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
    if not self._exp_created: raise ValueError('Experiment must be created')
    exp_id = exp_id or self.exp_id
    run_id = run_id or self.run_id

    to_path = Path(self.local_artifact_path) / fname
    from_path = GH_ARTIFACT_PATTERN.format(exp_id=exp_id, run_id=run_id)

    gh = GithubService()
    gh.download(remote_path=from_path, dest_path=to_path)


  def upload_file_gd(self, fname: str, data: Any = None):
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
    if not self._exp_created: raise ValueError('Experiment must be created')
    from_path = Path(self.local_artifact_path) / fname
    to_path = Path(self.gd_artifact_path) / fname
    gd = GDrive()

    if data is None:
      gd.send(from_path, to_path)
    elif type(data) == str:
      with open(to_path, 'w') as fp:
        fp.write(data)
    else:
      with open(to_path, 'w') as fp:
        json.dump(data, fp)



if __name__ == '__main__':
  print(DEV_LOCAL_ARTIFACT_PATTERN)
