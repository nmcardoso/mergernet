import secrets
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


class Job:
  def _optuna_train(self):
    print('aqui 1')
    ds = Dataset(data_path=self.local_data_path)
    print('aqui 2')
    hp = HyperParameterSet(self.job['hyperparameters'])
    print('aqui 3')
    model = HyperModel(
      dataset=ds,
      name=self.experiment_name,
      hyperparameters=hp,
      epochs=self.job['config']['train_epochs'],
    )
    print('aqui 4')
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
    print(extends_path)

    for path in extends_path:
      with open(path, 'r') as fp:
        extends_data.append(yaml.load(fp, yaml.FullLoader))

    if len(extends_data) > 0:
      data = deep_update(*extends_data, data)

    print(data)

    return data
