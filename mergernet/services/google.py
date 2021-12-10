from pathlib import Path
from shutil import copy



class GDrive:
  def __init__(self, base_path: Path):
    self.base_path = base_path


  def is_mounted(self) -> bool:
    return self.base_path.is_dir()

