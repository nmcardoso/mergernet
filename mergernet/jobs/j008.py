import numpy as np
import pandas as pd

from mergernet.core.constants import DATA_ROOT
from mergernet.core.experiment import Experiment
from mergernet.core.utils import compress_files, iauname, iauname_path
from mergernet.services.legacy import LegacyService


class Job(Experiment):
  def __init__(self):
    super().__init__()
    self.exp_id = 8
    self.log_wandb = False
    self.restart = False


  def call(self):
    images_path = DATA_ROOT / 'images' / 'decals_0.364_png'
    output_folder = DATA_ROOT / 'images' / 'decals_0.364_tar'
    all_folders = [p.name for p in sorted(images_path.glob('*')) if p.is_dir()]
    folders_per_part = 10
    parts = int(np.ceil(len(all_folders) / folders_per_part))

    for part in range(parts):
      print(f'>> Part {part + 1} of {parts}')
      folders = ','.join(all_folders[part*folders_per_part : (part+1)*folders_per_part-1])
      compress_files(
        input_path=images_path / f'{{{folders}}}',
        output_path=output_folder / f'decals_0.364_part{part}.tar.xz',
        level=0,
      )
      print()




if __name__ == '__main__':
  Job().run()
