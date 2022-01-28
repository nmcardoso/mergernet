from pathlib import Path
from typing import Sequence, Union
from enum import Enum
import json

from mergernet.core.constants import GITHUB_USER, GITHUB_TOKEN, GITHUB_REPO, GITHUB_PATH, GDRIVE_PATH
from mergernet.core.utils import SingletonMeta
from mergernet.services.github import GithubService
from mergernet.services.google import GDrive

class ArtifactUploader:
  def __init__(self, github: bool = True, gdrive: bool = True, gdrive_path: Path = None):
    self.use_github = github
    self.use_gdrive = gdrive
    self.gdrive_path = gdrive_path
    self.github = GithubService(user=GITHUB_USER, token=GITHUB_TOKEN, repo=GITHUB_REPO)
    self.gdrive = GDrive(base_path=GDRIVE_PATH)


  def upload_file(self, path):
    path = Path(path)
    if self.use_github:
      if path.exists():
        with open(path, 'rb') as file:
          file_bytes = file.read()
        self.github.commit(GITHUB_PATH, file_bytes, 'main')

    if self.use_gdrive:
      self.gdrive.send(path, path)



class BaseArtifact:
  """Represents an abstract logger."""
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)
    self._saved_path = None


  def serialize(self) -> dict:
    """Recursively parse an ``BaseLogger`` instance to a python ``dict``.

    Returns
    -------
    dict
      Serialized instance.
    """
    items = [(k, v) for k, v in self.__dict__.items() if not k.startswith('_')]
    log_dict = {}
    for k, v in items:
      attr = getattr(self, k)
      if isinstance(attr, BaseArtifact):
        log_dict[k] = v.serialize()
      else:
        log_dict[k] = v
    return log_dict


  def save(self, path: Path):
    """Serializes this object data and stores a json file in the given path.

    Parameters
    ----------
    path: Path
      Path to save the generated ``json`` file.
    """
    log_dict = self.serialize()
    json.dump(log_dict, path, indent=True)
    self._saved_path = path


  def upload(self, github=True, gdrive=True):
    uploader = ArtifactUploader(github=github, gdrive=gdrive)
    uploader.upload_file(self._saved_path)

  # @staticmethod
  # def load(path: Path):
  #   """Loads a given file path and parses the file data to logger object.

  #   Parameters
  #   ----------
  #   path: Path
  #     Path of the ``json`` file to read.

  #   Returns
  #   -------
  #   Logger
  #     Log instance with parsed data.
  #   """
  #   log_dict = json.load(path)
  #   return JobReport(**log_dict)



class ArtifactTypes(Enum):
  figure = 'figure'
  model_weights = 'model_weights'
  model = 'model'
  job = 'job'



