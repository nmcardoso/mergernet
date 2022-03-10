from setuptools import setup, find_packages

setup(
  name='mergernet',
  version='0.1.0',
  description='deep learning model',
  author='Natanael',
  author_email='nauxmac@gmail.com',
  packages=find_packages(),
  include_package_data=True,
  package_data={'mergernet': ['jobs/*.yaml']},
  install_requires=[
    'wheel',
    'numpy>=1.19.5',
    'pandas>=1.1.5',
    'tensorflow>=2.7',
    'tqdm>=4.62.3',
    'requests>=2.23',
    'Pillow>=7.1.2',
    'tensorflow_addons',
    'keras-tuner==1.1.0',
    'tensorboard',
    'optuna',
    'mlflow',
    'plotly'
  ]
)
