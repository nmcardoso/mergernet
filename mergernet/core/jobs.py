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




class JobRunner:
  def __init__(self):
    self.jobs_path = Path(__file__).parent / '../jobs'
    self.jobs = {}
    self.fetch()


  def fetch(self):
    self.jobs = {}
    for job in self.jobs_path.iterdir():
      filename = job.stem
      match = re.search(r'^j(\d+)', filename)
      if match:
        job_id = int(match.group(1))
        self.jobs[job_id] = job


  def list_jobs(self):
    for job_path in self.jobs.values():
      print(job_path.stem)


  def run_job(self, job_id: int):
    job_path = self.jobs[job_id]
    job_module = import_module(f'..jobs.{job_path.stem}', package='mergernet.jobs')
    job = job_module.Job()
    job.start_execution()



if __name__ == '__main__':
  jr = JobRunner()
  jr.run_job(1)
