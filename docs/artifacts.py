from pathlib import Path
import os
import json

from jinja2 import Template


DOCS_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_PATH = DOCS_PATH.parent / 'jobs_artifacts'
SAVE_PATH = DOCS_PATH / 'jobs'



template_str = (DOCS_PATH / 'artifacts_template.rst').read_text()
t = Template(template_str)


def main():
  artifacts = []

  for folder in ARTIFACTS_PATH.iterdir():
    meta_path = folder / 'job.json'
    if not meta_path.exists(): continue
    job_meta = json.loads(meta_path.read_text())
    artifacts.append((job_meta, folder))


  artifacts = list(sorted(artifacts, key=lambda a: a[0]['timestamp'], reverse=True))


  for i in range(len(artifacts)):
    meta = artifacts[i][0]
    folder = artifacts[i][1]

    log_path = folder / 'log.job'
    log = log_path.read_text() if log_path.exists() else ''

    base_url = 'https://raw.githubusercontent.com/nmcardoso/mergernet/main/jobs_artifacts'

    job_artifacts = [
      {'name': str(f.name), 'url': f'{base_url}/{str(folder.stem)}/{str(f.name)}'}
      for f in folder.iterdir()
    ]

    r = t.render(
      jobid=meta['jobid'],
      runid=meta['runid'],
      job_name=meta['name'],
      job_date=meta['timestamp'],
      job_description=meta['description'],
      job_log=log,
      job_artifacts=job_artifacts
    )

    gen_path = SAVE_PATH / f'{i:03d}.rst'
    gen_path.write_text(r)

main()
