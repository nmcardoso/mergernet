from pathlib import Path

import pandas as pd

from mergernet.core.jobs import BaseJob
from mergernet.core.dataset import Dataset
from mergernet.model.baseline import ConvolutionalClassifier
from mergernet.model.autokeras import AutoKerasSimpleClassifier



class Job(BaseJob):
  jobid = 1
  name = 'Test job'
  description = 'Test job description'

  def run(self):
    ds = Dataset(data_path=self.data_path)

    # model = ConvolutionalClassifier(ds)
    # model.train(
    #   epochs=20,
    #   optimizer='adam'
    # )

    model = AutoKerasSimpleClassifier(ds)
    model.fit(
      project_name='job_0001',
      directory=str(self.artifact_path / 'models'),
      max_trials=10
    )

