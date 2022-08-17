import logging
from pathlib import Path
from typing import Sequence, Union
from datetime import datetime, timezone, timedelta
from enum import Enum
import json

import tensorflow as tf

from mergernet.core.constants import GDRIVE_ARTIFACT_PATH, GH_BRANCH, GH_USER, GH_TOKEN, GH_REPO, GH_BASE_PATH
from .job import JobInfo
from mergernet.core.utils import SingletonMeta
from mergernet.services.github import GithubService
from mergernet.services.google import GDrive


L = logging.getLogger('job')


class ArtifactHelper:
  @staticmethod
  def upload_model(model: tf.keras.Model):
    path = GDRIVE_ARTIFACT_PATH.format(
      job_id=JobInfo.job_id,
      run_id=JobInfo.run_id,
      artifact_path=f'{model.name}.h5'
    )
    model.save(path)



    summary_fname = f'{model.name}.txt'
    summary_path = self.artifact_path / summary_fname
    with open(summary_path, 'w') as fp:
      model.summary(print_fn=lambda x: fp.write(x + '\n'))
    self.upload(path=summary_path, github=True, gdrive=True)


  def upload_optuna_db(self):
    name = self.optuna_db_name

    if name is None:
      raise ValueError("Can't resolve optuna db name. Set ah.optuna_db_name or arg.")

    name = str(Path(name).stem)

    local_path = self.artifact_path / 'optuna' / f'{name}.sqlite'
    remote_path = f'optuna/{name}.sqlite'

    with open(local_path, 'rb') as fp:
      self.github.commit(path=remote_path, data=fp, from_bytes=True)


  def download_optuna_db(self):
    name = self.optuna_db_name

    if name is None:
      raise ValueError("Can't resolve optuna db name. Set ah.optuna_db_name or arg.")

    name = str(Path(name).stem)

    self.github.download(
      remote_path=f'optuna/{name}.sqlite',
      dest_path=self.artifact_path / 'optuna' / f'{name}.sqlite'
    )


  def upload_log(self, name: str):
    self.upload(path=f'/tmp/{name}.log', github=True, gdrive=True)


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


  def _upload_github(self, path: Union[str, Path]):
    path = Path(path)
    filename = str(path.name)
    with open(path, 'rb') as file:
      file_bytes = file.read()

    gh_path = f'{GH_BASE_PATH}/{self.artifact_path.stem}/{filename}'
    if self.github.commit(gh_path, file_bytes, GH_BRANCH, from_bytes=True):
      L.info(f'[GITHUB] file "{filename}" successfully uploaded to github.')
    else:
      L.error(f'[GITHUB] unable to upload the file "{filename}"')


  def _upload_gdrive(self, path: Union[str, Path], root: bool = False):
    name = str(path.name)
    from_path = path

    if root:
      to_path = Path(name)
    else:
      to_path = Path(self.artifact_path.stem) / name

    if path.is_file():
      if self.gdrive.send(from_path, to_path):
        L.info(f'[GDRIVE] file "{name}" uploaded to Google Drive.')
      else:
        L.info(f'[GDRIVE] unable to upload the file "{name}"')
    else:
      if self.gdrive.send_dir(from_path, to_path):
        L.info(f'[GDRIVE] folder "{name}" uploaded to Google Drive.')
      else:
        L.info(f'[GDRIVE] unable to upload the folder "{name}"')
