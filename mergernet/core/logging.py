import logging
import shutil
from pathlib import Path
from tempfile import gettempdir
from typing import Union

LOG_PATH = Path(gettempdir()) / 'mergernet.log'


def configure_root_logger():
  """
  Configures the root logger
  """
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)

  formatter = logging.Formatter(
    fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
  )

  file_handler = logging.FileHandler(LOG_PATH)
  file_handler.setFormatter(formatter)
  file_handler.setLevel(logging.DEBUG)

  stream_handler = logging.StreamHandler()
  stream_handler.setFormatter(formatter)
  stream_handler.setLevel(logging.INFO)

  logger.addHandler(file_handler)
  logger.addHandler(stream_handler)
