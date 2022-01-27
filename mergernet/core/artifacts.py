from pathlib import Path
from typing import Sequence, Union
from enum import Enum
import json

from mergernet.core.constants import GITHUB_USER, GITHUB_TOKEN, GITHUB_REPO, GITHUB_PATH, GDRIVE_PATH
from mergernet.core.utils import SingletonMeta
from mergernet.services.github import GithubService
from mergernet.services.google import GDrive

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


