from pathlib import Path
from typing import Sequence, Union
from enum import Enum
import json

from mergernet.core.constants import GITHUB_BRANCH, GITHUB_USER, GITHUB_TOKEN, GITHUB_REPO, GITHUB_PATH, GDRIVE_PATH
from mergernet.core.utils import SingletonMeta
from mergernet.services.github import GithubService
from mergernet.services.google import GDrive



class ArtifactHelper(metaclass=SingletonMeta):
  use_github = True
  use_gdrive = True
  artifact_path = None
  gdrive_path = None


  def _upload_github(self, filename: str):
    path = self.artifact_path / filename
    with open(path, 'rb') as file:
      file_bytes = file.read()

    gh_path = f'{GITHUB_PATH}/{self.artifact_path.stem}/{filename}'
    self.github.commit(gh_path, file_bytes, GITHUB_BRANCH, from_bytes=True)


  def _upload_gdrive(self, filename: str):
    from_path = self.artifact_path / filename
    to_path = Path(self.artifact_path.stem) / filename
    self.gdrive.send(from_path, to_path)


  def config(
    self,
    artifact_path: Union[str, Path] = None,
    gdrive_path: Union[str, Path] = None,
    use_github: bool = None,
    use_gdrive: bool = None
  ):
    if artifact_path:
      self.artifact_path = Path(artifact_path)
      if not self.artifact_path.exists():
        self.artifact_path.mkdir(parents=True, exist_ok=True)

    if gdrive_path:
      self.gdrive_path = Path(gdrive_path)
      # TODO: check if this path exists (is_mounted() == True)

    if use_github is not None: self.use_github = use_github
    if use_gdrive is not None: self.use_gdrive = use_gdrive

    self.github = GithubService(user=GITHUB_USER, token=GITHUB_TOKEN, repo=GITHUB_REPO)
    self.gdrive = GDrive(base_path=gdrive_path)


  def upload(self, filename: str, github: bool = None, gdrive: bool = None):
    use_github = github if github is not None else self.use_github
    use_gdrive = gdrive if gdrive is not None else self.use_gdrive
    suffix = Path(filename).suffix
    github_ext = ['.json', '.log', '.csv', '.png', '.svg', '.pdf', '.jpg', '.jpeg']

    if use_github and suffix in github_ext:
      self._upload_github(filename)

    if use_gdrive:
      self._upload_gdrive(filename)


  def save_json(self, data: dict, filename: Union[str, Path]):
    path = self.artifact_path / filename
    with open(path, 'w') as fp:
      json.dump(data, fp, indent=True)



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



class FigureArtifact(BaseArtifact):
  code: str = ArtifactTypes.figure.value
  filename: str = None



class ModelWeightsArtifact(BaseArtifact):
  code: str = ArtifactTypes.model.value
  filename: str = None



class ModelMetrics(BaseArtifact):
  acc: Union[float, Sequence] = None
  loss: Sequence[float] = None
  precision: Sequence = None
  recall: Sequence = None
  roc_auc: Sequence = None
  pr_auc: Sequence = None



class Validation(BaseArtifact):
  X: Sequence = None
  y_ref: Sequence = None
  y_pred: Sequence = None
  metrics: ModelMetrics = None



class ModelArtifact(BaseArtifact):
  code: str = ArtifactTypes.model.value
  validation_type: str = None
  train: Sequence[ModelMetrics] = None
  val: Sequence[Validation] = None



class JobArtifact(BaseArtifact):
  code: str = ArtifactTypes.job.value
  jobid: int = None
  runid: int = None
  name: str = None
  description: str = None
