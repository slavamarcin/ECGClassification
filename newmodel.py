import sys
import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import keras as keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sns.set_style('darkgrid')
NUMPY_DATA_FILE = 'data.npz'

thismodule = sys.modules[__name__]

with np.load(NUMPY_DATA_FILE) as data:
    for k in data.keys():
        setattr(thismodule, k, data[k].astype(float))

