from setuptools import find_packages, setup

setup(
  name='mergernet',
  version='0.2.0',
  description='deep learning model',
  author='Natanael',
  author_email='nauxmac@gmail.com',
  packages=find_packages(),
  include_package_data=True,
  package_data={
    'mergernet': []
  },
  install_requires=[
    'wheel',
    'numpy>=1.19.5',
    'pandas>=1.1.5',
    'tensorflow>=2.7',
    'scikit-learn>=1.0.1',
    'tqdm>=4.62.3',
    'requests>=2.23',
    'Pillow>=7.1.2',
    'astropy>=4.3.1'
    'tensorflow_addons',
    'tensorboard',
    'optuna',
    'plotly',
    'wandb',
    'keras_applications', # zoobot
    'tensorflow_probability>=0.11', # zoobot
    'seaborn' , # zoobot
    'zoobot',
    'pydot', # tf.keras.utils.plot_model
  ],
  extras_require={
    'docs': [
      'Jinja2>=3.1',
      'sphinxcontrib-napoleon',
      'sphinx',
      'furo',
      'ipykernel',
      'sphinx_copybutton',
    ]
  }
)
