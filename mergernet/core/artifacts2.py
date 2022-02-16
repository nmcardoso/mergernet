import logging
from pathlib import Path


from mergernet.core.utils import SingletonMeta


L = logging.getLogger('job')

class ArtifactHelper(metaclass=SingletonMeta):
  def __init__(self):
    self.artifact_path = Path()
    self.mlflow_db_path = ''
    self.mlflow_db_uri = ''
    self.optuna_db_path = ''
    self.optuna_db_uri = ''

