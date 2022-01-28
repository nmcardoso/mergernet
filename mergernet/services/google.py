from typing import Union
from pathlib import Path
from shutil import copy2



class GDrive:
  base_path = None

  def __init__(self, base_path: Union[str, Path] = None):
    if base_path: self.base_path = Path(base_path)


  def is_mounted(self) -> bool:
    if self.base_path:
      return self.base_path.is_dir()
    return False


  def send(self, local: Path, remote: Path):
    remote = self.base_path / remote
    copy(str(local), str(remote))


  def get(self, remote: Path, local: Path):
    remote = self.base_path / remote
    copy(str(remote), str(local))
