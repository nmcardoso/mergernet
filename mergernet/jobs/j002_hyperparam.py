from pathlib import Path
import pandas as pd

from mergernet.core.jobs import BaseJob
from mergernet.core.dataset import Dataset
from mergernet.model.hypermodel import SimpleHyperModel, BayesianTuner



class Job(BaseJob):
  jobid = 2
  name = 'Test job'
  description = 'Test job description'

  def run(self):
    ds = Dataset(data_path=self.data_path, in_memory=True)

    tuner = BayesianTuner(
      max_trials=3,
      overwrite=True,
      directory=self.artifact_path / 'tuner',
      project_name='resnet'
    )

    tuner.search(dataset=ds)



    # model = SimpleHyperModel(ds)
    # model.pipeline()

