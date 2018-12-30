#!/usr/bin/env python

import mpl_toolkits # import before pathlib
import sys
import pathlib

from tensorflow import set_random_seed

# sys.path.append(pathlib.Path(__file__).parent)
from train_model import *
from model import *
from dataset import *

# np.random.seed(19)
# set_random_seed(19)

# BASE_MODEL = 'vgg19'
# BASE_MODEL = 'incepstionresnetv2'
BASE_MODEL = 'resnet50'
# BASE_MODEL = 'adams'
if BASE_MODEL == 'resnet50':
    create_model = create_model_resnet50_plain
elif BASE_MODEL == 'vgg19':
    create_model = create_model_vgg19_plain
elif BASE_MODEL == 'incepstionresnetv2':
    create_model = create_model_inceptionresnetv2_plain
elif BASE_MODEL == 'adams':
    create_model = create_model_adams
else:
    raise Exception("unimplemented model")


def main():
    raw_data_path = 'raw_data.pickle'

    # Load audio segments using pydub
    raw_data, y_true = load_raw_data()
    # with open(raw_data_path, 'wb') as f:
    #     pickle.dump(raw_data, f)
    # exit(0)
    # with open(raw_data_path, 'rb') as f:
    #     raw_data = pickle.load(f)

    weight_param_path = f"model/{BASE_MODEL}.weights.best.hdf5"
    model = create_model(input_shape=(IMAGE_SIZE//2, IMAGE_SIZE//2, TRAIN_COLOR_NUM))
    # model = create_model(input_shape=(299, 299, TRAIN_COLOR_NUM))
    model = build_model(model, weight_param_path)
    for i in range(0, 100):
        print(f"num:{i}. start train")
        train_model(model, raw_data, weight_param_path)
        # print(f"rebuild model.")
        # model = build_model(model, None)
    eval_model(model, raw_data)


if __name__ == "__main__":
    main()
