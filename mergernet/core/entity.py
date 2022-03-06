from typing import Any, Sequence, Union
import optuna



class HyperParameter:
  def set_trial(self, trial: optuna.trial.FrozenTrial):
    self._trial = trial

  def suggest(self, trial: optuna.trial.FrozenTrial = None):
    _trial = trial or self._trial
    if isinstance(self, CategoricalHyperParameter):
      fn = _trial.suggest_categorical
    elif isinstance(self, DiscreteUniformHyperParameter):
      fn = _trial.suggest_discrete_uniform
    elif isinstance(self, FloatHyperParameter):
      fn = _trial.suggest_float
    elif isinstance(self, IntHyperParameter):
      fn = _trial.suggest_int
    elif isinstance(self, LogUniformHyperParameter):
      fn = _trial.suggest_loguniform
    elif isinstance(self, UniformHyperParameter):
      fn = _trial.suggest_uniform
    elif isinstance(self, ConstantHyperParameter):
      return self.value
    filtered_args = {
      k: v for k, v in self.__dict__.items()
      if not k.startswith('_')
    }
    return fn(**filtered_args)

  @classmethod
  def from_dict(cls, params: dict):
    t = params.pop('type')
    if t == 'categorical':
      E = CategoricalHyperParameter
    elif t == 'discrete_uniform':
      E = DiscreteUniformHyperParameter
    elif t == 'float':
      E = FloatHyperParameter
    elif t == 'int':
      E = IntHyperParameter
    elif t == 'loguniform':
      E = LogUniformHyperParameter
    elif t == 'uniform':
      E = UniformHyperParameter
    elif t == 'constant':
      E = ConstantHyperParameter
    return E(**params)



class CategoricalHyperParameter(HyperParameter):
  def __init__(self, name: str, choices: Sequence):
    self.name = name
    self.choices = choices



class DiscreteUniformHyperParameter(HyperParameter):
  def __init__(self, name: str, low: float, high: float, q: float):
    self.name = name
    self.low = low
    self.high = high
    self.q = q



class FloatHyperParameter(HyperParameter):
  def __init__(self, name: str, low: float, high: float, step: float = None, log: bool = False):
    self.name = name
    self.low = low
    self.high = high
    self.step = step
    self.log = log



class IntHyperParameter(HyperParameter):
  def __init__(self, name: str, low: int, high: int, step: int = 1, log: bool = False):
    self.name = name
    self.low = low
    self.high = high
    self.step = step
    self.log = log



class LogUniformHyperParameter(HyperParameter):
  def __init__(self, name: str, low: float, high: float):
    self.name = name
    self.low = low
    self.high = high



class UniformHyperParameter(HyperParameter):
  def __init__(self, name: str, low: float, high: float):
    self.name = name
    self.low = low
    self.high = high



class ConstantHyperParameter(HyperParameter):
  def __init__(self, name: str, value: Any):
    self.name = name
    self.value = value



class HyperParameterSet:
  def __init__(self, hyperparameters: Sequence[Union[dict, HyperParameter]]):
    for item in hyperparameters:
      if type(item) == dict:
        name = item['name']
        self.__dict__.update({ name: HyperParameter.from_dict(item) })
      else:
        name = item.name
        self.__dict__.update({ name: item })

  def set_trial(self, trial: optuna.trial.FrozenTrial):
    for hp in self.__dict__.values():
      hp.set_trial(trial)



class JobSpec:
  name: str
  dataset: str
  hyperparameters: HyperParameterSet



if __name__ == '__main__':
  hp = HyperParameter.from_dict({
    'type': 'categorical',
    'name': 'params',
    'choices': [0, 1, 2]
  })
  print(type(hp))
  print(hp.__dict__)
