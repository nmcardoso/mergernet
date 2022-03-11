import logging
import secrets
import shutil
import yaml
import re
from pathlib import Path
from typing import Any, Dict, Union

import mlflow
import optuna

from mergernet.core.constants import GDRIVE_PATH, JOBS_PATH
from mergernet.core.dataset import Dataset
from mergernet.core.entity import HyperParameterSet
from mergernet.core.utils import deep_update, unique_path
from mergernet.model.study import HyperModel

L = logging.getLogger('job')


class Job:
  def __init__(
    self,
    job_id: int,
    local_data_path: Union[str, Path] = ''
  ):
    # constructor params to class state
    self.job_id = job_id
    self.local_data_path = Path(local_data_path)

    # load job metadata
    self.jobs_map = self._scan_jobs()
    self.job = self._parse_job(job_id)

    self.experiment_name = Path(self.job['filename']).stem

    # setup remote job path
    self.remote_artifact_path = None
    self._config_remote_artifact_path()



  def run(self):
    if self.job['config']['mlflow']['enabled']:
      self._config_mlflow()

    if self.job['config']['job_type'] == 'optuna_train':
      self._config_optuna()
      self._optuna_train()
      self._upload_mldflow_artifacts()
    elif self.job['config']['job_type'] == 'predict':
      self._predict()
      self._upload_mldflow_artifacts()


  def _config_remote_artifact_path(self):
    assert GDRIVE_PATH is not None

    path = None
    if self.job['config']['mlflow']['enabled']:
      path = Path(GDRIVE_PATH) / 'mlflow' / 'artifacts'

    self.remote_artifact_path = path


  def _upload_mldflow_artifacts(self):
    mlruns = Path('mlruns')
    mlflow_folder = Path(GDRIVE_PATH) / 'mlflow'

    for exp in mlruns.iterdir():
      for run in exp.iterdir():
        artifacts = (run / 'artifacts').glob('*.*')
        for artifact in artifacts:
          dest = mlflow_folder / artifact
          if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            L.info(f'[ART] copying artifact `{str(artifact)} to `{str(dest)}`.')
            shutil.copy2(artifact, dest)


  def _config_mlflow(self):
    assert GDRIVE_PATH is not None

    mlflow_folder = Path(GDRIVE_PATH) / 'mlflow'
    if not mlflow_folder.exists():
      mlflow_folder.mkdir(exist_ok=True)

    db_name = 'mlflow.sqlite'
    db_path =  mlflow_folder / db_name
    db_uri = f'sqlite:///{str(db_path.resolve())}'

    mlflow.set_tracking_uri(db_uri)
    # mlflow.set_tracking_uri(f'file:///{str(mlflow_folder.resolve())}')
    mlflow.set_experiment(self.experiment_name)


  def _config_optuna(self):
    assert GDRIVE_PATH is not None

    optuna_folder = Path(GDRIVE_PATH) / 'optuna'
    if not optuna_folder.exists():
      optuna_folder.mkdir(exist_ok=True)

    db_path = optuna_folder / (self.experiment_name + '.sqlite')
    if db_path.exists() and not self.job['config']['resume']:
      db_path.unlink()

    self.optuna_uri = f'sqlite:///{str(db_path.resolve())}'


  def _optuna_train(self):
    ds = Dataset(
      data_path=self.local_data_path,
      ds=self.job['config']['dataset']
    )
    hp = HyperParameterSet(self.job['hyperparameters'])
    model = HyperModel(
      dataset=ds,
      name=self.experiment_name,
      hyperparameters=hp,
      epochs=self.job['config']['train_epochs'],
    )
    model.hypertrain(
      optuna_uri=self.optuna_uri,
      n_trials=self.job['config']['optuna']['n_trials'],
      pruner=self.job['config']['optuna']['pruner'],
      resume=bool(self.job['config']['resume'])
    )


  def _scan_jobs(self) -> Dict[str, Path]:
    jobs_map = {}
    pattern = re.compile('(\d+)_.*\.yaml')

    for path in JOBS_PATH.iterdir():
      match = pattern.match(path.name)
      if match:
        job_id = int(match.group(1))
        jobs_map[job_id] = path

    return jobs_map


  def _parse_job(self, job_id: int) -> Dict[str, Any]:
    assert job_id in self.jobs_map

    path = self.jobs_map[job_id]
    defaults_path = JOBS_PATH / 'defaults.yaml'

    with open(path, 'r') as fp:
      data = yaml.load(fp, yaml.FullLoader)
    data.update({ 'filename': path.name })

    extends_path = []

    if defaults_path.exists():
      extends_path.append(defaults_path)

    if 'extends' in data:
      for e in data['extends']:
        if 'id' in e:
          extends_path.append(self.jobs_map[e['id']])
        elif 'name' in e:
          extends_path.append(JOBS_PATH / (Path(e['name']).name + '.yaml'))

    extends_data = []
    extends_path = set(extends_path) # unique values

    for path in extends_path:
      with open(path, 'r') as fp:
        extends_data.append(yaml.load(fp, yaml.FullLoader))

    if len(extends_data) > 0:
      data = deep_update(*extends_data, data)

    return data
