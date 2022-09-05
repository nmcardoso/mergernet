from typing import Any, Sequence, Union

import optuna


class HyperParameter:
  def set_trial(self, trial: optuna.trial.FrozenTrial):
    self._trial = trial


  def suggest(self, trial: optuna.trial.FrozenTrial = None):
    if isinstance(self, ConstantHyperParameter):
      return self.value

    _trial = trial or self._trial
    if isinstance(self, CategoricalHyperParameter):
      fn = _trial.suggest_categorical
    elif isinstance(self, FloatHyperParameter):
      fn = _trial.suggest_float
    elif isinstance(self, IntHyperParameter):
      fn = _trial.suggest_int

    filtered_args = {
      k: v for k, v in self.__dict__.items()
      if not k.startswith('_')
    }

    return fn(**filtered_args)


  @staticmethod
  def from_dict(params: dict):
    t = params.pop('type')
    if t == 'categorical':
      E = CategoricalHyperParameter
    elif t == 'float':
      E = FloatHyperParameter
    elif t == 'int':
      E = IntHyperParameter
    elif t == 'constant':
      E = ConstantHyperParameter
    return E(**params)



class CategoricalHyperParameter(HyperParameter):
  def __init__(self, name: str, choices: Sequence):
    self.name = name
    self.choices = choices



class FloatHyperParameter(HyperParameter):
  def __init__(
    self,
    name: str,
    low: float,
    high: float,
    step: float = None,
    log: bool = False
  ):
    self.name = name
    self.low = low
    self.high = high
    self.step = step
    self.log = log



class IntHyperParameter(HyperParameter):
  def __init__(
    self,
    name: str,
    low: int,
    high: int,
    step: int = 1,
    log: bool = False
  ):
    self.name = name
    self.low = low
    self.high = high
    self.step = step
    self.log = log



class ConstantHyperParameter(HyperParameter):
  def __init__(self, name: str, value: Any):
    self.name = name
    self.value = value



class HP:
  @staticmethod
  def cat(name: str, choices: Sequence) -> CategoricalHyperParameter:
    return CategoricalHyperParameter(name, choices)


  @staticmethod
  def const(name: str, value: Any) -> ConstantHyperParameter:
    return CategoricalHyperParameter(name, value)


  @staticmethod
  def num(
    name: str,
    low: Union[float, int],
    high: Union[float, int],
    step: Union[float, int] = None,
    log: bool = False,
    dtype: Union[float, int] = float
  ) -> Union[FloatHyperParameter, IntHyperParameter]:
    if dtype == float:
      return FloatHyperParameter(name, low, high, step, log)
    else:
      return IntHyperParameter(name, low, high, step, log)



class HyperParameterSet:
  """
  Represents a set of hyperparameters and handles the hyperparameters
  register and access.

  Parameters
  ----------
  *args: HyperParameter
    Any sequence of HyperParameter subclass
  """
  def __init__(self, *args: HyperParameter):
    self.hps = {}

    for hp in args:
      if isinstance(hp, HyperParameter):
        self.hps[hp.name] = hp


  def add(self, hyperparameters: Sequence[Union[dict, HyperParameter]]):
    """
    Parses a sequence of dictionaries that represents the hyperparameters

    Parameters
    ----------
    hyperparameters: array-like of dictionaries or arrar-like of HyperParameter
      The list of hyperparameters that will be added to this
      hyperparameters set
    """
    for item in hyperparameters:
      if type(item) == dict:
        name = item['name']
        self.hps.update({ name: HyperParameter.from_dict(item) })
      else:
        name = getattr(item, name)
        self.hps.update({ name: item })


  def get(self, hp: str, trial: optuna.trial.FrozenTrial = None) -> Any:
    """
    Get the value of a hyperparameter identified by its name.
    For hyperparameters different than ConstantHyperParameter, this method
    will use optuna's seggest api

    Parameters
    ----------
    hp: str
      The hyperparamer name
    trial: optuna.trial.FrozenTrial
      The optuan trial instance

    Returns
    -------
    Any
      The hyperparameter value

    See Also
    --------
    mergernet.core.hp.HyperParameter.suggest
    """
    if trial is None:
      return self.hps[hp].suggest()
    else:
      return self.hps[hp].suggest(trial)


  def set_trial(self, trial: optuna.trial.FrozenTrial):
    """
    Sets the optuna's trial for all hyperparameter in this set

    Parameters
    ----------
    trial: optuna.trial.FrozenTrial
      The trial that will be added
    """
    for hp in self.hps.values():
      hp.set_trial(trial)
