from typing import Union
from pathlib import Path
import subprocess



class TensorboardService:
  def upload_assets(
    self,
    logdir: Union[str, Path],
    name: str = '',
    description: str = ''
  ):
    cmd = [
      'tensorboard', 'dev', 'upload',
      '--logdir', f'"{str(logdir)}"',
      '--name', f'"{name}"',
      '--description', f'"{description}"'
    ]

    subprocess.run(cmd)
