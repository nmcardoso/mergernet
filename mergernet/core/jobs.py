from pathlib import Path
from importlib import import_module
import re
import secrets

from mergernet.core.logger import Logger



class Job:
  id = 1

  def __init__(self):
    self.run_id = None
    self.logger = None


  def start_execution(self):
    self.pre_run()
    self.run()
    self.post_run()


  def pre_run(self):
    self.run_id = secrets.token_hex(nbytes=8)
    self.logger = Logger()
    print('pre run')


  def run(self):
    print('empty job')


  def post_run(self):
    self.logger.save()
    print('post run')


  def get_system_resources(self):
    pass
