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
COLORS = ['red', 'blue', 'green', 'yellow', ]
TRAIN_DIR = 'train'
TRAIN_ANSWER_FILE = 'train.csv'
TEST_DIR = 'test'
# dataset ratio
TRAIN_RATIO = 0.90
VALIDATE_RATIO = 0.05
TEST_RATIO = 0.05

CLASS_NUM = 28
IMAGE_SIZE = 512
HOT_ENCODE_MAP = pd.get_dummies(pd.Series(list(range(CLASS_NUM)))).as_matrix()
VOID = [0 for _ in range(CLASS_NUM)]


class DataType(Enum):
    train = 1
    validate = 2
    test = 3


class Dataset:
    def __init__(self, data_list: list):
        self.data_list = data_list


class DataUnit:
    def __init__(self, uuid: str, answers: np.array):
        self.uuid = uuid
        self.answers = answers


def load_raw_data():
    data_list = []
    with open(TRAIN_ANSWER_FILE, 'r') as f:
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
            data_list.append(DataUnit(uuid, np.array(encoded_answers)))
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
    for uuid in uuids:
        data_list.append(DataUnit(uuid, None))
    return Dataset(data_list)


def create_xy(dataset: Dataset, datatype: DataType):
    sample_num = len(dataset.data_list)
    train_num = int(sample_num * TRAIN_RATIO)
    validate_num = int(sample_num * VALIDATE_RATIO)
    test_num = int(sample_num * VALIDATE_RATIO)
    if datatype == DataType.train:
        random_index = np.random.randint(train_num)
    elif datatype == DataType.validate:
        random_index = train_num + np.random.randint(validate_num)
    else:
        random_index = train_num + validate_num + np.random.randint(test_num)
    data_unit = dataset.data_list[random_index]
    x = []
    for color in COLORS:
        file_path = Path(TRAIN_DIR, f"{data_unit.uuid}_{color}.png")
        img = Image.open(str(file_path))
        img.convert('L')
        x.append((np.array(img) / 255.0))
    x = np.array(x)
    x = np.stack(x, axis=2)
    # print(f"create_training_sample x:{x.shape} y:{data_unit.answers.shape}")
    return x, data_unit.answers


def create_unit_dataset(data_unit: DataUnit, dir_sr: str):
    x = []
    for color in COLORS:
        file_path = Path(dir_sr, f"{data_unit.uuid}_{color}.png")
        img = Image.open(str(file_path))
        img.convert('L')
        x.append((np.array(img) / 255.0))
    x = np.array(x)
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
