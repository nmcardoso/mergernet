from typing import Union
from pathlib import Path
from datetime import datetime
from time import strftime, gmtime
import sys
import shutil
import json

from jinja2 import Environment, PackageLoader, select_autoescape

from mergernet.core.constants import DATA_ROOT, ENV, PROJECT_ROOT

sys.path.append(str(PROJECT_ROOT.resolve()))



if ENV.lower() == 'ci':
  BASE_URL = 'https://nmcardoso.github.io/mergernet-experiments/'
else:
  BASE_URL = '/'




def format_date(timestamp: float):
  if timestamp > 0:
    return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
  else:
    return ''



def render_page(
  output_path: Union[str, Path],
  template: str,
  **kwargs
):
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  env = Environment(
    loader=PackageLoader(__package__, 'templates'),
    autoescape=select_autoescape()
  )

  template = env.get_template(template)

  page = template.render(**kwargs, base_url=BASE_URL)

  with open(output_path, 'w') as fp:
    fp.write(page)



def render_index(source: Path, dest: Path, order: str, kind: str):
  """
  Renders the index pages for experiment and run

  Parameters
  ----------
  source: Path
    The path of artifacts
  dest: Path
    The path where the file will be saved
  order: str
    The order of the list (`mumerical` or `date`)
  kind: str
    The type of index (`experiment` or `run`)
  """
  if kind == 'experiment':
    title = 'Experiment'
    metadata_glob = '/**/metadata.json'
    template = 'index.html.j2'
  else:
    title = 'Run'
    metadata_glob = '/metadata.json'
    template = 'experiment.html.j2'

  dirs = [p for p in source.glob('*') if p.is_dir()]
  latest_ts = []
  for d in dirs:
    metas = [
      json.loads(p.read_text(encoding='utf-8'))
      for p in source.glob(f'{d.stem}{metadata_glob}')
    ]

    if len(metas) == 0:
      latest_ts.append(-1)
      continue

    metas_sorted = sorted(
      metas,
      key=lambda m: m['end_time'],
      reverse=True
    )
    ts = metas_sorted[0]['end_time']
    latest_ts.append(ts)

  if order == 'numerical':
    sorted_idx = sorted(enumerate(dirs), key=lambda t: int(t[1].stem))
  else:
    sorted_idx = sorted(enumerate(latest_ts), key=lambda t: t[1], reverse=True)
  dirs = [dirs[i] for i, _ in sorted_idx] # apply mask
  latest_ts = [latest_ts[i] for i, _ in sorted_idx] # apply mask

  items = [
    {
      'title': f'{title} {d.stem}',
      'url': f'{d.stem}/',
      'date': format_date(ts)
    }
    for d, ts in zip(dirs, latest_ts)
  ]

  render_page(
    output_path=dest,
    template=template,
    items=items,
    exp_id=source.stem
  )



def render_report(source: Path, dest: Path):
  meta_path = source / 'metadata.json'
  meta = json.loads(meta_path.read_text('utf-8')) if meta_path.exists() else {}

  train_artifacts_path = list(source.glob('*'))
  train_artifacts = [p.name for p in train_artifacts_path]

  # generate plots

  generated_artifacts = []

  # format duration
  hours, remainder = divmod(meta.get('duration', 0), 3600)
  minutes, seconds = divmod(remainder, 60)
  meta['duration'] = f'{hours:02}:{minutes:02}:{seconds:02}'
  meta['start_time'] = strftime('%H:%M:%S %Y/%m/%d', gmtime(meta.get('start_time', 0)))
  meta['end_time'] = strftime('%H:%M:%S %Y/%m/%d', gmtime(meta.get('end_time', 0)))

  render_page(
    output_path=dest,
    template='run.html.j2',
    meta=meta,
    train_artifacts=train_artifacts,
    generated_artifacts=generated_artifacts
  )

  # copy artifacts to output path
  for path in train_artifacts_path:
    shutil.copy2(path, dest.parent / path.name)



def render(source: Path, dest: Path):
  source = Path(source)
  dest = Path(dest)

  if dest.exists():
    shutil.rmtree(dest)
  dest.mkdir(parents=True, exist_ok=True)

  render_index(source, dest / 'index.html', 'numerical', 'experiment') # /index.html
  render_index(source, dest / 'date.html', 'date', 'experiment') # /date.html

  for s in source.glob('*'):
    render_index(s, dest / s.name / 'index.html', 'date', 'run') # 2/index.html

  for s in [path for path in source.glob('*/**/*') if path.is_dir()]:
    render_report(s, dest / s.parent.name / s.name / 'index.html') # 2/3290f3/index.html




if __name__ == '__main__':
  if ENV.lower() == 'ci':
    render('experiments', 'output')
  else:
    render('mocks/experiments', 'dev/output')
