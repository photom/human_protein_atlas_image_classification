import os
import pickle
import sys
import pathlib
from pathlib import Path
import csv
from enum import Enum

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# import mpl_toolkits # import before pathlib

# sys.path.append(pathlib.Path(__file__).parent)
IMAGE_BASE_DIR = '../hpaic/input/'
COLORS = ['red', 'green', 'blue', 'yellow', ]
TRAIN_COLOR_NUM = 3
# TRAIN_DIRS = [f"{IMAGE_BASE_DIR}/train", f"{IMAGE_BASE_DIR}/HPAv18"]
TRAIN_DIRS = [f"{IMAGE_BASE_DIR}/train", ]
# TRAIN_ANSWER_FILES = ['train.csv', 'HPAv18RBGY_wodpl.csv']
TRAIN_ANSWER_FILES = ['train.csv', ]
TEST_DIR = IMAGE_BASE_DIR + 'test'
# dataset ratio
TRAIN_RATIO = 0.90
VALIDATE_RATIO = 0.05
TEST_RATIO = 0.05

CLASS_NUM = 28
IMAGE_SIZE = 512
HOT_ENCODE_MAP = pd.get_dummies(pd.Series(list(range(CLASS_NUM)))).as_matrix()
VOID = [0 for _ in range(CLASS_NUM)]
IMAGE_MINMAX_MAP = {}


class DataType(Enum):
    train = 1
    validate = 2
    test = 3


class Dataset:
    def __init__(self, data_list: list):
        self.data_list = data_list


class DataUnit:
    def __init__(self, uuid: str, answers: np.array, source_dir: str):
        self.uuid = uuid
        self.answers = answers
        self.source_dir = source_dir


def load_raw_data():
    data_list = []
    for idx, train_answer_file in enumerate(TRAIN_ANSWER_FILES):
        source_dir = TRAIN_DIRS[idx]
        with open(train_answer_file, 'r') as f:
            csv_reader = csv.reader(f)
            # skip header
            next(csv_reader)
            # read lines
            for row in csv_reader:
                if len(row) != 2:
                    continue
                uuid = row[0]
                answers = []
                encoded_answers = []
                for answer in row[1].split():
                    answers.append(int(answer.strip()))
                for i in range(CLASS_NUM):
                    if i in answers:
                        encoded_answers.append(1)
                    else:
                        encoded_answers.append(0)
                data_list.append(DataUnit(uuid, np.array(encoded_answers), source_dir))
    np.random.shuffle(data_list)
    y_true = [data.answers for data in data_list]

    return Dataset(data_list), np.array(y_true)


def load_test_data():
    data_list = []
    uuids = set()
    uuid_sample = '00008af0-bad0-11e8-b2b8-ac1f6b6435d0'
    uuid_len = len(uuid_sample)
    for image_file in os.listdir(TEST_DIR):
        uuids.add(image_file[:uuid_len])
    for uuid in sorted(list(uuids)):
        data_list.append(DataUnit(uuid, None, TRAIN_DIRS[0]))
    return Dataset(data_list)


def create_xy(dataset: Dataset, datatype: DataType):
    sample_num = len(dataset.data_list)
    train_num = int(sample_num * TRAIN_RATIO)
    validate_num = int(sample_num * VALIDATE_RATIO)
    test_num = int(sample_num * VALIDATE_RATIO)
    while True:
        if datatype == DataType.train:
            random_index = np.random.randint(train_num)
        elif datatype == DataType.validate:
            random_index = train_num + np.random.randint(validate_num)
        else:
            random_index = train_num + validate_num + np.random.randint(test_num)
        data_unit = dataset.data_list[random_index]
        file_path = Path(data_unit.source_dir, f"{data_unit.uuid}_yellow.png")
        if file_path.exists():
            break

    x = []
    for color in COLORS:
        file_name = f"{data_unit.uuid}_{color}.png"
        file_path = Path(data_unit.source_dir, file_name)
        img = Image.open(str(file_path))
        img = img.resize((IMAGE_SIZE // 2, IMAGE_SIZE // 2), Image.LANCZOS)
        # img = img.resize((299, 299), Image.LANCZOS)
        img.convert('L')
        img = np.array(img)
        # if file_name in IMAGE_MINMAX_MAP:
        #     min_val, max_val = IMAGE_MINMAX_MAP[file_name]
        # else:
        #     min_val, max_val = float(img.min()), float(img.max())
        #     IMAGE_MINMAX_MAP[file_name] = (min_val, max_val)
        # if max_val > min_val + np.finfo(float).eps:
        #     normalized = (img - min_val) / (max_val - min_val)
        # else:
        #     normalized = img
        x.append(img)

    # merge and normalize
    # x = [x[0] / 2.0 + x[1] / 2.0, x[2] / 2.0 + x[1] / 2.0, x[3] / 2.0 + x[1] / 2.0]
    x = [x[0] / 2.0 + x[3] / 2.0, x[1] / 2.0 + x[3] / 2.0, x[2]]
    x = np.array(x)
    # print(x)
    if data_unit.uuid in IMAGE_MINMAX_MAP:
        min_val, max_val = IMAGE_MINMAX_MAP[data_unit.uuid]
    else:
        min_val, max_val = float(x.min()), float(x.max())
        IMAGE_MINMAX_MAP[data_unit.uuid] = (min_val, max_val)
    if max_val > min_val + np.finfo(float).eps:
        x = (x - min_val) / (max_val - min_val)
    # x = x / 255.0
    # print(x)
    x = np.stack(x, axis=2)
    # print(f"create_training_sample x:{x.shape} y:{data_unit.answers.shape}")
    return x, data_unit.answers


def create_unit_dataset(data_unit: DataUnit, dir_sr: str):
    x = []
    for color in COLORS:
        file_path = Path(dir_sr, f"{data_unit.uuid}_{color}.png")
        img = Image.open(str(file_path))
        img = img.resize((IMAGE_SIZE // 2, IMAGE_SIZE // 2), Image.LANCZOS)
        # img = img.resize((299, 299), Image.LANCZOS)
        img.convert('L')
        img = np.array(img)
        # min_val, max_val = float(img.min()), float(img.max())
        # if max_val > min_val + np.finfo(float).eps:
        #     normalized = (img - min_val) / (max_val - min_val)
        # else:
        #     normalized = img
        x.append(img)

    # merge and normalize
    x = [x[0] / 2.0 + x[1] / 2.0, x[2] / 2.0 + x[1] / 2.0, x[3] / 2.0 + x[1] / 2.0]
    x = np.array(x)
    if data_unit.uuid in IMAGE_MINMAX_MAP:
        min_val, max_val = IMAGE_MINMAX_MAP[data_unit.uuid]
    else:
        min_val, max_val = float(x.min()), float(x.max())
        IMAGE_MINMAX_MAP[data_unit.uuid] = (min_val, max_val)
    if max_val > min_val + np.finfo(float).eps:
        x = (x - min_val) / (max_val - min_val)
    # x = x / 255.0
    x = np.stack(x, axis=2)
    return x


def create_dataset(dataset: Dataset, num: int, datatype: DataType = DataType.train):
    train_dataset_x = []
    train_dataset_y = []
    for i in range(num):
        x, y = create_xy(dataset, datatype)
        train_dataset_x.append(x)
        train_dataset_y.append(y)
    train_dataset_x = np.array(train_dataset_x)
    train_dataset_y = np.array(train_dataset_y)
    return train_dataset_x, train_dataset_y


def load_dataset(filename):
    with open(filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        train_dataset = pickle.load(f)
    return train_dataset
