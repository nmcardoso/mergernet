from pathlib import Path
from typing import Union
import shutil
import logging

import optuna



def configure_logger(logger_name: Union[str, None], path: str):
  if logger_name:
    logger = logging.getLogger(logger_name)
  else:
    logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)

  formatter = logging.Formatter(
    fmt='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
  )

  path = Path(path)
  if path.exists():
    path.unlink()

  file_handler = logging.FileHandler(path)
  file_handler.setFormatter(formatter)
  file_handler.setLevel(logging.DEBUG)

  stream_handler = logging.StreamHandler()
  stream_handler.setFormatter(formatter)
  stream_handler.setLevel(logging.DEBUG)

  logger.addHandler(file_handler)
  logger.addHandler(stream_handler)


# configure mergernet logger
configure_logger('job', '/tmp/job.log')

# configure root logger
configure_logger(None, '/tmp/root.log')

# propagate logs to the root logger
optuna.logging.enable_propagation()

# stop showing logs in sys.stderr
optuna.logging.disable_default_handler()
