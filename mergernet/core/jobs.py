from enum import Enum
from pathlib import Path
from importlib import import_module
from typing import Sequence, Union
from datetime import datetime, timezone, timedelta
import re
import secrets
import json
from mergernet.core.artifacts import ArtifactHelper

from mergernet.core.constants import GDRIVE_PATH
from mergernet.core.logger import Logger


class BaseJob:
  def start_execution(self, data_path: Union[str, Path] = None, **kwargs):
    self.pre_run(data_path=data_path)
    self.run(**kwargs)
    self.post_run()


  def pre_run(self, data_path: Union[str, Path] = None):
    # setup storage folders
    self.runid = secrets.token_hex(3)
    self.data_path = Path(data_path)
    self.artifact_path = self.data_path / f'job_{self.jobid}_run_{self.runid}'

    # config artifact helper
    ah = ArtifactHelper()
    ah.config(
      artifact_path=self.artifact_path,
      gdrive_path=GDRIVE_PATH,
      use_gdrive=True,
      use_github=True
    )

    # create and upload job metadata artifact
    tz = timezone(timedelta(hours=-3))
    now = datetime.now(tz=tz).isoformat(sep=' ', timespec='seconds')
    job_artifact = dict(
      jobid=self.jobid,
      runid=self.runid,
      name=self.name,
      description=self.description,
      timestamp=now
    )
    ah.upload_json(job_artifact, 'job.json')


  def run(self):
    print('Job not implemented')


  def post_run(self):
    ah = ArtifactHelper()
    ah.upload_log()


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


  def run_job(self, job_id: int, data_path: str, **kwargs):
    job_path = self.jobs[job_id]
    job_module = import_module(f'..jobs.{job_path.stem}', package='mergernet.jobs')
    job = job_module.Job()
    job.start_execution(data_path=data_path, **kwargs)




if __name__ == '__main__':
  jr = JobRunner()
  jr.run_job(1)
