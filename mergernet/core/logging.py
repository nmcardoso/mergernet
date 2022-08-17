from pathlib import Path
from typing import Union
import shutil
import logging


LOG_PATH = '/tmp/mergernet.log'


def get_logger(logger_name: str):
  """
  Configures a Logger instance and returns it

  Parameters
  ----------
  logger_name: str
    The logger name, usually `__name__`

  Returns
  -------
  logging.Logger
    Logger instance
  """
  logger = logging.getLogger(logger_name)
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
  stream_handler.setLevel(logging.DEBUG)

  logger.addHandler(file_handler)
  logger.addHandler(stream_handler)

  return logger


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
  stream_handler.setLevel(logging.DEBUG)

  logger.addHandler(file_handler)
  logger.addHandler(stream_handler)
