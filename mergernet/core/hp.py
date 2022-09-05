from typing import Any, Sequence, Union

import optuna


class HyperParameter:
  def __init__(self):
    self._trial = None
    self.attrs = {}

  def set_attr(self, key, value):
    self.attrs[key] = value

  def set_trial(self, trial: optuna.trial.FrozenTrial):
    self._trial = trial

  def suggest(self, trial: optuna.trial.FrozenTrial = None):
    pass

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
    super(CategoricalHyperParameter, self).__init__()
    self.set_attr('name', name)
    self.set_attr('choices', choices)

  def suggest(self, trial: optuna.trial.FrozenTrial) -> Any:
    _trial = trial or self._trial
    return _trial.suggest_categorical(**self.attrs)



class FloatHyperParameter(HyperParameter):
  def __init__(
    self,
    name: str,
    low: float,
    high: float,
    step: float = None,
    log: bool = False
  ):
    super(FloatHyperParameter, self).__init__()
    self.set_attr('name', name)
    self.set_attr('low', low)
    self.set_attr('high', high)
    self.set_attr('step', step)
    self.set_attr('log', log)

  def suggest(self, trial: optuna.trial.FrozenTrial) -> float:
    _trial = trial or self._trial
    return _trial.suggest_float(**self.attrs)



class IntHyperParameter(HyperParameter):
  def __init__(
    self,
    name: str,
    low: int,
    high: int,
    step: int = 1,
    log: bool = False
  ):
    super(IntHyperParameter, self).__init__()
    self.set_attr('name', name)
    self.set_attr('low', low)
    self.set_attr('high', high)
    self.set_attr('step', step)
    self.set_attr('log', log)

  def suggest(self, trial: optuna.trial.FrozenTrial = None) -> int:
    _trial = trial or self._trial
    return _trial.suggest_int(**self.attrs)



class ConstantHyperParameter(HyperParameter):
  def __init__(self, name: str, value: Any):
    super(ConstantHyperParameter, self).__init__()
    self.set_attr('name', name)
    self.set_attr('value', value)

  def suggest(self, trial: optuna.trial.FrozenTrial = None) -> Any:
    return self.attrs['value']



class HP:
  @staticmethod
  def cat(name: str, choices: Sequence) -> CategoricalHyperParameter:
    return CategoricalHyperParameter(name, choices)


  @staticmethod
  def const(name: str, value: Any) -> ConstantHyperParameter:
    return ConstantHyperParameter(name, value)


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
