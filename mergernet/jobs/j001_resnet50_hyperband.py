from mergernet.core.jobs import BaseJob
from mergernet.core.dataset import Dataset
from mergernet.model.study import HyperModel



class Job(BaseJob):
  jobid = 1
  name = 'Resnet50'
  description = (
    'Minimal resnet50 experiment with 3 classes and unclean data. '
    'Using all darg mergers with modelMag < 17.2. '
    'Using hyperband pruner.'
  )

  def run(self):
    ds = Dataset(data_path=self.data_path)

    model = HyperModel(ds, 'resnet50_minimal')
    model.hypertrain(
      n_trials=60,
      epochs=25,
      pruner='hyperband'
    )

