import logging
from pathlib import Path
from typing import Sequence, Union
from enum import Enum
import json

import tensorflow as tf

from mergernet.core.constants import GITHUB_BRANCH, GITHUB_USER, GITHUB_TOKEN, GITHUB_REPO, GITHUB_PATH, GDRIVE_PATH
from mergernet.core.utils import SingletonMeta
from mergernet.services.github import GithubService
from mergernet.services.google import GDrive


L = logging.getLogger('job')


class ArtifactHelper(metaclass=SingletonMeta):
  use_github = True
  use_gdrive = True
  artifact_path = None
  gdrive_path = None


  def _upload_github(self, path: Union[str, Path]):
    path = Path(path)
    filename = str(path.name)
    with open(path, 'rb') as file:
      file_bytes = file.read()

    gh_path = f'{GITHUB_PATH}/{self.artifact_path.stem}/{filename}'
    if self.github.commit(gh_path, file_bytes, GITHUB_BRANCH, from_bytes=True):
      L.info(f'[GITHUB] file "{filename}" successfully uploaded to github.')
    else:
      L.error(f'[GITHUB] unable to upload the file "{filename}"')


  def _upload_gdrive(self, path: Union[str, Path]):
    filename = str(path.name)
    from_path = path
    to_path = Path(self.artifact_path.stem) / filename

    if self.gdrive.send(from_path, to_path):
      L.info(f'[GDRIVE] file "{filename}" uploaded to Google Drive.')
    else:
      L.info(f'[GDRIVE] unable to upload the file "{filename}"')


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


  def upload(
    self,
    fname: str = None,
    path: Union[str, Path] = None,
    github: bool = None,
    gdrive: bool = None
  ):
    use_github = github if github is not None else self.use_github
    use_gdrive = gdrive if gdrive is not None else self.use_gdrive
    path = Path(path) if path else self.artifact_path / fname

    suffix = path.suffix
    github_ext = ['.json', '.log', '.txt', '.csv', '.png', '.svg', '.pdf', '.jpg', '.jpeg']

    if use_github and suffix in github_ext:
      self._upload_github(path)

    if use_gdrive:
      self._upload_gdrive(path)


  def upload_log(self):
    self.upload(path='/tmp/job.log', github=True, gdrive=True)


  def upload_model(self, model: tf.keras.Model):
    filename = f'{model.name}.h5'
    path = self.artifact_path / filename
    model.save(str(path))
    self.upload(path=path, github=False, gdrive=True)

    summary_fname = f'{model.name}.txt'
    summary_path = self.artifact_path / summary_fname
    with open(summary_path, 'w') as fp:
      model.summary(print_fn=lambda x: fp.write(x + '\n'))
    self.upload(path=summary_path, github=True, gdrive=True)


  def upload_json(self, data: object, filename: Union[str, Path]):
    self.save_json(data, filename)
    self.upload(fname=filename)


  def upload_text(self, data: str, filename: str):
    with open(self.artifact_path / filename, 'w') as fp:
      fp.write(data)
    self.upload(fname=filename, github=True, gdrive=True)


  def save_json(self, data: object, filename: Union[str, Path]):
    path = self.artifact_path / filename
    with open(path, 'w') as fp:
      json.dump(data, fp, indent=True)


  def upload_dir(self, path: Union[str, Path]):
    self._upload_gdrive(path)
