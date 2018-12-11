#!/usr/bin/env python

import mpl_toolkits # import before pathlib
import sys
import pathlib

from tensorflow import set_random_seed

# sys.path.append(pathlib.Path(__file__).parent)
from train_model import *
from model import *
from dataset import *

np.random.seed(19)
set_random_seed(19)

BASE_MODEL = 'vgg19'
raw_data_path = 'raw_data.pickle'

# Load audio segments using pydub
raw_data, y_true = load_raw_data()
# with open(raw_data_path, 'wb') as f:
#     pickle.dump(raw_data, f)
# exit(0)
# with open(raw_data_path, 'rb') as f:
#     raw_data = pickle.load(f)

weight_param_path = f"model/{BASE_MODEL}.weights.best.hdf5"
# model_dilation
model = build_model(weight_param_path, create_model=create_model_vgg19_plain)
model.summary()
# model_dilation
np.random.seed(19)
set_random_seed(19)

for i in range(0, 4):
    print(f"num:{i}")
    train_model(model, raw_data, weight_param_path)
    eval_model(model, raw_data)
