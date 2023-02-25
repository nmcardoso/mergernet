import pandas as pd

from mergernet.core.constants import DATA_ROOT
from mergernet.core.experiment import Experiment
from mergernet.core.utils import (compress_images, iauname,
                                  iauname_relative_path)
from mergernet.services.legacy import LegacyService


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 8
    self.log_wandb = False
    self.restart = False


  def call(self):
    path = DATA_ROOT / 'images' / 'decals_0.364_png'
    output_path = DATA_ROOT / 'images' / 'decals_0.364_tar' / 'decals_0.364.tar.xz'
    iaunames = [i.stem for i in path.glob('**/*.png')]

    compress_images(
      iaunames=iaunames,
      base_path=path,
      image_ext='png',
      output_path=output_path,
      max_files=2
    )




if __name__ == '__main__':
  Job().run()
