from pathlib import Path
import json



class AbstractLogger:
  """Represents an abstract logger, concrete classes implements the attrbutes."""
  def serialize(self) -> dict:
    """Recursively parse an ``AbstractLogger`` instance to a python ``dict``.

    Returns
    -------
    dict
      Serialized instance.
    """
    items = [(k, v) for k, v in self.__dict__.items() if not k.startswith('__')]
    log_dict = {}
    for k, v in items:
      attr = getattr(self, k)
      if isinstance(attr, AbstractLogger):
        log_dict[k] = v.serialize()
      else:
        log_dict[k] = v
    return log_dict



