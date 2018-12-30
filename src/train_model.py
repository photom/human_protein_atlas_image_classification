import string
import random
import sys
import pickle
import pathlib

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# sys.path.append(pathlib.Path(__file__).parent)
from dataset import *
from metrics import *

BATCH_SIZE = 5
datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             zca_whitening=False,  # for memory error
                             rotation_range=90,
                             width_shift_range=0.05,
                             height_shift_range=0.05,
                             zoom_range=0.1,
                             brightness_range=[0.8, 1.2],
                             shear_range=0.10,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='constant',
                             cval=0.0,
                             # fill_mode='reflect',
                             )


def create_callbacks(name_weights, patience_lr=10, patience_es=10):
    mcp_save = ModelCheckpoint(name_weights, save_weights_only=True,
                               save_best_only=True, monitor='val_loss', mode='min')
    # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_lr, verbose=1, min_delta=1e-4, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience_es, verbose=1, mode='auto')
    # return [early_stopping, mcp_save, reduce_lr_loss]
    # return [f1metrics, early_stopping, mcp_save]
    return [early_stopping, mcp_save]


def load_dataset(filename):
    with open(filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        train_dataset = pickle.load(f)
    return train_dataset


def next_dataset(dataset: Dataset, batch_size: int, datatype: DataType):
    """ Obtain a batch of training data
    """
    while True:
        train_dataset_x = []
        train_dataset_y = []
        for i in range(batch_size):
            x, y = create_xy(dataset, datatype)
            if datatype != DataType.test:
                datagen.fit([x])
            train_dataset_x.append(x)
            train_dataset_y.append(y)
        x_train, y_train = np.array(train_dataset_x), np.array(train_dataset_y)
        if False:
        # if datatype != DataType.test:
            x_batched, y_batched = next(datagen.flow(x_train, y=y_train,
                                                     # save_to_dir='../hpaic/augment',
                                                     batch_size=batch_size))
            # print(f"x_list:{np.array(x_list).shape} y_data:{np.array(train_dataset_y).shape}")
            yield x_batched, y_batched
        else:
            yield x_train, y_train


def train_model(model: Model, dataset: Dataset, model_filename: str,
                batch_size=BATCH_SIZE,
                num_train_examples=29450 // 8,
                # num_valid_samples=1550 // 2,
                # num_train_examples=29450 // 4,
                num_valid_samples=1550 // 4,
                epochs=1,):
    callbacks = create_callbacks(model_filename)
    steps_per_epoch = num_train_examples // batch_size
    # steps_per_epoch = 200
    # epochs = 100
    validation_steps = num_valid_samples // batch_size
    model.fit_generator(generator=next_dataset(dataset, batch_size, DataType.train),
                        epochs=epochs,
                        validation_data=next_dataset(dataset, batch_size, DataType.validate),
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        callbacks=callbacks, verbose=1)


def predict(data_unit: DataUnit, model: Model):
    x = create_unit_dataset(data_unit, TEST_DIR)
    # print(f"x:{x.shape}")
    # predict
    result = model.predict(np.array([x]))
    predicted = np.round(result)
    return predicted


def eval_model(model: Model, dataset: Dataset):
    sample_num = len(dataset.data_list)
    test_num = int(sample_num * VALIDATE_RATIO)
    test_num = test_num

    metrics = MacroF1Score()
    score = None
    result = None
    num_step = test_num // BATCH_SIZE
    counter = 0
    for x, y in next_dataset(dataset, BATCH_SIZE, DataType.test):
        if counter >= num_step:
            break
        else:
            counter += 1
        result = model.predict(x)
        predicted = np.round(result)
        score = metrics(y, predicted)
        result = K.eval(score)

    print(f"counter={counter} macro_f1_score={result}")
    # tf.keras.backend.clear_session()
