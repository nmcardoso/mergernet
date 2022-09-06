from mergernet.core.experiment import backup_model, experiment_run
from mergernet.core.hp import HP, HyperParameterSet
from mergernet.data.dataset import Dataset
from mergernet.model.inference import Predictor
from mergernet.model.utils import load_model


@experiment_run(4)
def run():
  """
  Evaluatge the best model of experiment 3 in BLIND_SPLUS_GAL80_LS10_RGB_128
  dataset

  Dataset
  -------
    - BLIND_SPLUS_GAL80_LS10_RGB_128
  """
  ds = Dataset(config=Dataset.registry.BLIND_SPLUS_GAL80_LS10_RGB_128)
  model = load_model('model.h5', 3, '6bc82d80')
  p = Predictor(model, ds)
  p.predict()
  p.upload(label_map=Dataset.registry.BIN_LEGACY_NORTH_RGB_128.label_map)



if __name__ == '__main__':
  # run()
  print(Dataset.registry.BIN_LEGACY_NORTH_RGB_128.label_map.items())
