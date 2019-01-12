import os
import pickle
import sys
import pathlib
from pathlib import Path
import csv
from enum import Enum

import pandas as pd
import numpy as np
from PIL import Image
from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold
from skmultilearn import model_selection
from keras.callbacks import Callback


RANDOM_NUM = 77777
IMAGE_BASE_DIR = '../hpaic/input/'
COLORS = ['red', 'green', 'blue', 'yellow', ]
TRAIN_COLOR_NUM = 3
# TRAIN_DIRS = [f"{IMAGE_BASE_DIR}/train", f"{IMAGE_BASE_DIR}/HPAv18"]
TRAIN_DIRS = [f"{IMAGE_BASE_DIR}/train", ]
# TRAIN_ANSWER_FILES = ['train.csv', 'HPAv18RBGY_wodpl.csv']
TRAIN_ANSWER_FILES = ['train.csv', ]
TEST_DIR = IMAGE_BASE_DIR + 'test'
# dataset ratio
TRAIN_RATIO = 0.95
VALIDATE_RATIO = 0.05
TEST_RATIO = 0.05

CLASS_NUM = 28
IMAGE_SIZE = 512
HOT_ENCODE_MAP = pd.get_dummies(pd.Series(list(range(CLASS_NUM)))).as_matrix()
IMAGE_MINMAX_MAP = {}
BATCH_SIZE = 20


class DataType(Enum):
    train = 1
    validate = 2
    test = 3


class Dataset(Callback):
    def __init__(self, data_list: np.array, y_list: np.array):
        super(Dataset, self).__init__()
        self.data_list = data_list
        self.y_list = y_list
        self.train_index_list = np.array([])
        self.validate_index_list = np.array([])
        self.train_counter = 0
        self.validate_counter = 0
        rmskf = RepeatedMultilabelStratifiedKFold(n_splits=10, n_repeats=100, random_state=RANDOM_NUM)
        self.index_generator = rmskf.split(self.data_list, self.y_list)

    def on_train_begin(self, logs=None):
        train_index, validate_index = next(self.index_generator)
        self.train_index_list = train_index
        self.validate_index_list = validate_index
        self.train_counter = 0
        self.validate_counter = 0
        print(f"reset dataset. train_dataset:{len(train_index)} validate_dataset:{len(validate_index)}")

    def on_epoch_end(self, epoch, logs=None):
        pass

    def increment_train(self):
        self.train_counter = (self.train_counter + 1) % len(self.train_index_list)

    def increment_validate(self):
        self.validate_counter = (self.validate_counter + 1) % len(self.validate_index_list)

    def next_train_data(self):
        index = self.train_index_list[self.train_counter]
        self.increment_train()
        return self.data_list[index]

    def next_validate_data(self):
        index = self.validate_index_list[self.validate_counter]
        self.increment_validate()
        return self.data_list[index]


class TestDataset:
    def __init__(self, data_list: list):
        self.data_list = data_list


class DataUnit:
    def __init__(self, uuid: str, answers: np.array, source_dir: str):
        self.uuid = uuid
        self.answers = answers
        self.source_dir = source_dir


def load_raw_data():
    data_list = []

    y_list = []
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
                for answer in row[1].split():
                    answers.append(int(answer.strip()))
                y = np.zeros(CLASS_NUM)
                for answer in answers:
                    y[answer] = 1
                data_unit = DataUnit(uuid, y, source_dir)
                data_list.append(data_unit)
                y_list.append(y)

    return Dataset(np.array(data_list), np.array(y_list))


def load_test_data():
    data_list = []
    uuids = set()
    uuid_sample = '00008af0-bad0-11e8-b2b8-ac1f6b6435d0'
    uuid_len = len(uuid_sample)
    for image_file in os.listdir(TEST_DIR):
        uuids.add(image_file[:uuid_len])
    for uuid in sorted(list(uuids)):
        data_list.append(DataUnit(uuid, None, TRAIN_DIRS[0]))
    return TestDataset(data_list)


def create_xy(dataset: Dataset, datatype: DataType):
    while True:
        if datatype == DataType.train:
            data_unit = dataset.next_train_data()
        elif datatype == DataType.validate:
            data_unit = dataset.next_validate_data()
        else:
            raise RuntimeError(f"invalid data type. type={datatype}")
        file_path = Path(data_unit.source_dir, f"{data_unit.uuid}_yellow.png")
        if file_path.exists():
            break
        else:
            print(f"image does not exists. path={str(file_path)}")

    x = []
    for color in COLORS:
        file_name = f"{data_unit.uuid}_{color}.png"
        file_path = Path(data_unit.source_dir, file_name)
        img = Image.open(str(file_path))
        # img = img.resize((IMAGE_SIZE // 2, IMAGE_SIZE // 2), Image.LANCZOS)
        img = img.resize((192, 192), Image.LANCZOS)
        img.convert('L')
        img = np.array(img)
        x.append(img)

    # merge and normalize
    x = np.array(x)
    # print(x)
    if data_unit.uuid in IMAGE_MINMAX_MAP:
        min_val, max_val = IMAGE_MINMAX_MAP[data_unit.uuid]
    else:
        min_val, max_val = float(x.min()), float(x.max())
        IMAGE_MINMAX_MAP[data_unit.uuid] = (min_val, max_val)
    if max_val > min_val + np.finfo(float).eps:
        x = (x - min_val) / (max_val - min_val)
    else:
        x = x / 255.0
    # print(x)
    x = np.stack(x, axis=2)
    # y = data_unit.answers[klass]
    # print(f"create_training_sample x:{x} y:{y}")
    return x, data_unit.answers
    # return x, y


def create_unit_dataset(data_unit: DataUnit, dir_str: str):
    x = []
    for color in COLORS:
        file_path = Path(dir_str, f"{data_unit.uuid}_{color}.png")
        img = Image.open(str(file_path))
        # img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        # img = img.resize((IMAGE_SIZE // 2, IMAGE_SIZE // 2), Image.LANCZOS)
        img = img.resize((192, 192), Image.LANCZOS)
        img.convert('L')
        img = np.array(img)
        x.append(img)
    # merge and normalize
    x = np.array(x)
    if data_unit.uuid in IMAGE_MINMAX_MAP:
        min_val, max_val = IMAGE_MINMAX_MAP[data_unit.uuid]
    else:
        min_val, max_val = float(x.min()), float(x.max())
        IMAGE_MINMAX_MAP[data_unit.uuid] = (min_val, max_val)
    if max_val > min_val + np.finfo(float).eps:
        x = (x - min_val) / (max_val - min_val)
    else:
        x = x / 255.0
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
