from pathlib import Path

import pandas as pd

from mergernet.core.jobs import BaseJob
from mergernet.core.dataset import Dataset
from mergernet.model.baseline import ConvolutionalClassifier



class Job(BaseJob):
  jobid = 1
  name = 'Test job'
  description = 'Test job description'

  def run(self):
    # pass
    ds = Dataset(data_path=self.data_path)

    model = ConvolutionalClassifier(ds)
    model.train(
      epochs=2,
      optimizer='adam'
    )

