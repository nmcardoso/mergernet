from pathlib import Path
import pandas as pd

from mergernet.core.jobs import BaseJob
from mergernet.core.dataset import Dataset
from mergernet.model.hypermodel import HyperModelTrainer, SimpleHyperModel, BayesianTuner
from mergernet.model.study import HyperModel



class Job(BaseJob):
  jobid = 2
  name = 'Test job'
  description = 'Test job description'

  def run(self):
    ds = Dataset(data_path=self.data_path)

    # hypertrain(ds)
    model = HyperModel(ds)
    model.hypertrain()

    # HyperModelTrainer(ds).fit()

    # tuner = BayesianTuner(
    #   max_trials=3,
    #   overwrite=True,
    #   directory=self.artifact_path / 'tuner',
    #   project_name='resnet'
    # )

    # tuner.search(dataset=ds)



    # model = SimpleHyperModel(ds)
    # model.pipeline()

