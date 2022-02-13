import logging

def configure_logger():
  logger = logging.getLogger('job')
  logger.setLevel(logging.DEBUG)

  formatter = logging.Formatter(
    fmt='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
  )

  file_handler = logging.FileHandler('/tmp/job.log')
  file_handler.setFormatter(formatter)
  file_handler.setLevel(logging.DEBUG)

  stream_handler = logging.StreamHandler()
  stream_handler.setFormatter(formatter)
  stream_handler.setLevel(logging.DEBUG)

  logger.addHandler(file_handler)
  logger.addHandler(stream_handler)

configure_logger()
