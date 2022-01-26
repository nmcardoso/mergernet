from pathlib import Path
import json



class BaseLogger:
  """Represents an abstract logger."""
  def serialize(self) -> dict:
    """Recursively parse an ``BaseLogger`` instance to a python ``dict``.

    Returns
    -------
    dict
      Serialized instance.
    """
    items = [(k, v) for k, v in self.__dict__.items() if not k.startswith('__')]
    log_dict = {}
    for k, v in items:
      attr = getattr(self, k)
      if isinstance(attr, BaseLogger):
        log_dict[k] = v.serialize()
      else:
        log_dict[k] = v
    return log_dict



class Logger(BaseLogger):
  """This class represents the json log file data structure.
  Also provides methods that transform an instance of this class in a
  high-level log representation.
  """
  def __init__(
    self,
    name=None,
    age=None,
    train_history: TrainHistory = None
  ):
    self.name = name
    self.age = age
    self.train_history = TrainHistory(**train_history) if train_history else None


  @staticmethod
  def load(path: Path):
    """Loads a given file path and parses the file data to logger object.

    Parameters
    ----------
    path: Path
      Path of the ``json`` file to read.

    Returns
    -------
    Logger
      Log instance with parsed data.
    """
    log_dict = json.load(path)
    return Logger(**log_dict)


  def save(self, path: Path) -> None:
    """Serializes this object data and stores a json file in the given path.

    Parameters
    ----------
    path: Path
      Path to save the generated ``json`` file.
    """
    log_dict = self.serialize()
    json.dump(log_dict, path)



if __name__ == '__main__':
  l = Logger(train_history={'epochs': 1, 'key1': {'key2': '2'}})
  print(l.serialize())
