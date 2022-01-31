from pathlib import Path
from typing import Union
import logging

from mergernet.core.utils import SingletonMeta


class Logger(metaclass=SingletonMeta):
  def __init__(self, save_path: Union[str, Path] = None):
    self.logger = logging.getLogger('job')
    self.logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
      fmt='%(asctime)s %(levelname)s %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(Path(save_path) / 'job.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)

    self.logger.addHandler(file_handler)
    self.logger.addHandler(stream_handler)


  def get_logger(self):
    return self.logger


if __name__ == '__main__':
  l = Logger().get_logger()
