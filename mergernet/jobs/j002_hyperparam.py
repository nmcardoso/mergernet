from pathlib import Path

import pandas as pd

from mergernet.core.jobs import BaseJob
from mergernet.core.dataset import Dataset
from mergernet.model.hypermodel import SimpleHyperModel, BayesianTuner



class Job(BaseJob):
  jobid = 1
  name = 'Test job'
  description = 'Test job description'

  def run(self):
    ds = Dataset(data_path=self.data_path)

    tuner = BayesianTuner(
      max_trials=15,
      overwrite=True,
      directory=self.artifact_path / 'tuner',
      project_name='resnet'
    )

    tuner.search(dataset=ds)



    # model = SimpleHyperModel(ds)
    # model.pipeline()

