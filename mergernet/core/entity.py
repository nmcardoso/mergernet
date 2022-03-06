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



