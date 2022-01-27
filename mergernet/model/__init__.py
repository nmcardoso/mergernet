import os
import random

import numpy as np
import tensorflow as tf

from mergernet.core.constants import RANDOM_SEED




# Set seeds
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)


