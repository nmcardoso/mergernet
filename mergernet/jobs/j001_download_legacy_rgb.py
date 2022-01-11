from mergernet.core.jobs import Job

import pandas as pd

from mergernet.core.utils import load_table



class Job(Job):
  def __init__(self):
    super(Job).__init__()

    self.id = 1
    self.description = 'Download Legacy RGB stamps from GalaxyZoo table'
    self.type = 'data'


  def run(self):
    df = load_table('merger.fit')
    df[df['p_mg'] > 0.5]
