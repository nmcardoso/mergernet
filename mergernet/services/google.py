import shutil
from typing import Union
from pathlib import Path
from shutil import copy2, copytree



class GDrive:
  base_path = None

  def __init__(self, base_path: Union[str, Path] = None):
    if base_path: self.base_path = Path(base_path)


  @staticmethod
  def get_url(fileid):
    return f'https://drive.google.com/uc?export=download&id={fileid}'


  def is_mounted(self) -> bool:
    if self.base_path:
      return self.base_path.is_dir()
    return False


  def send(self, local: Path, remote: Path) -> Union[str, None]:
    remote = self.base_path / remote
    if self.is_mounted():
      if not remote.parent.exists():
        remote.parent.mkdir(parents=True, exist_ok=True)
      return copy2(str(local), str(remote))
    else:
      return None


  def send_dir(self, local: Path, remote: Path) -> Union[str, None]:
    remote = self.base_path / remote
    if self.is_mounted():
      if not remote.parent.exists():
        remote.parent.mkdir(parents=True, exist_ok=True)
      return copytree(str(local), str(remote))
    else:
      return None


  def get(self, remote: Path, local: Path) -> Union[str, None]:
    remote = self.base_path / remote
    if remote.exists():
      return copy2(str(remote), str(local))
    return None


  def exists(self, path: Union[str, Path]):
    return (self.base_path / path).exists()


  def move(self, from_path: Union[str, Path], to_path: Union[str, Path]):
    from_path = self.base_path / from_path
    to_path = self.base_path / to_path

    if not to_path.parent.exists():
      to_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(from_path, to_path)
