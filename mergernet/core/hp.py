from typing import Any, Sequence, Union

import optuna


class HyperParameter:
  def __init__(self):
    self._trial = None
    self.last_value = None
    self.attrs = {}

  def set_attr(self, key, value):
    self.attrs[key] = value

  def set_trial(self, trial: optuna.trial.FrozenTrial):
    self._trial = trial

  def suggest(self, trial: optuna.trial.FrozenTrial = None):
    pass

  def to_dict(self, show_name: bool = False):
    if show_name:
      return self.attrs
    else:
      copy = self.attrs.copy()
      copy.pop('name')
      return copy

  def clear_last_value(self):
    self.last_value = None

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

  def suggest(self, trial: optuna.trial.FrozenTrial = None) -> Any:
    _trial = trial or self._trial
    self.last_value = _trial.suggest_categorical(**self.attrs)
    return self.last_value



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

  def suggest(self, trial: optuna.trial.FrozenTrial = None) -> float:
    _trial = trial or self._trial
    self.last_value = _trial.suggest_float(**self.attrs)
    return self.last_value



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
    self.last_value = _trial.suggest_int(**self.attrs)
    return self.last_value



class ConstantHyperParameter(HyperParameter):
  def __init__(self, name: str, value: Any):
    super(ConstantHyperParameter, self).__init__()
    self.set_attr('name', name)
    self.set_attr('value', value)
    self.last_value = value

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
    Any HyperParameter instance created by `HP` factory

  See Also
  --------
  mergernet.core.hp.HP
  """
  def __init__(self, *args: HyperParameter):
    self.hps = {}

    for hp in args:
      if isinstance(hp, HyperParameter):
        name = hp.attrs['name']
        self.hps[name] = hp
      elif isinstance(hp, dict):
        pass


  def add(self, hyperparameters: Sequence[Union[dict, HyperParameter]]):
    """
    Parses a sequence of dictionaries that represents the hyperparameters

    Parameters
    ----------
    hyperparameters: array-like of dictionaries or arrar-like of HyperParameter
      The list of hyperparameters that will be added to this
      hyperparameters set
    """
    for hp in hyperparameters:
      if type(hp) == dict:
        name = hp['name']
        self.hps.update({ name: HyperParameter.from_dict(hp) })
      else:
        name = hp.attrs['name']
        self.hps.update({ name: hp })


  def get(
    self,
    name: str,
    trial: optuna.trial.FrozenTrial = None,
    default: Any = None
  ) -> Any:
    """
    Get the value of a hyperparameter identified by its name.
    For hyperparameters different than ConstantHyperParameter, this method
    will use optuna's seggest api

    Parameters
    ----------
    name: str
      The hyperparamer name
    trial: optuna.trial.FrozenTrial
      The optuan trial instance
    default: Any
      Default value returned if the specified hyperparameter name wasn't found

    Returns
    -------
    Any
      The hyperparameter value

    See Also
    --------
    mergernet.core.hp.HyperParameter.suggest
    """
    if not name in self.hps:
      return default

    if trial is None:
      return self.hps[name].suggest()
    else:
      return self.hps[name].suggest(trial)


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


  def to_values_dict(self):
    """
    Returns a dict representation of this hyperparameters set with hp name
    as dict key and last optuna's suggested value as dict value
    """
    return { name: hp.last_value for name, hp in self.hps.items() }


  def clear_values_dict(self):
    """
    Clear ``last_value`` property of the hyperparameter, relevant when training
    with conditional hyperparameters
    """
    for hp in self.hps.values():
      hp.clear_last_value()
