import string
import random
import sys
import pickle
import pathlib

from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa

# sys.path.append(pathlib.Path(__file__).parent)
from dataset import *
from metrics import *


seq = iaa.Sequential([
    iaa.OneOf([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Flipud(0.5),  # vertically flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                      ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-180, 180),
            shear=(-8, 8)
        )
    ])], random_order=True)


def create_callbacks(dataset: Dataset, metrics: F1Metrics, name_weights, patience_lr=10, patience_es=150):
    mcp_save = ModelCheckpoint(name_weights, save_weights_only=True,
                               save_best_only=True, monitor='val_loss', mode='min')
    # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_lr, verbose=1, min_delta=1e-4, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience_es, verbose=1, mode='auto')
    # return [early_stopping, mcp_save, reduce_lr_loss]
    # return [f1metrics, early_stopping, mcp_save]
    return [early_stopping, mcp_save, metrics, dataset]


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
            train_dataset_x.append(x)
            train_dataset_y.append(y)
        x, y = np.array(train_dataset_x), np.array(train_dataset_y)
        # print(f"x_list:{np.array(x).shape} y_data:{np.array(y).shape}")
        if datatype != DataType.test:
            do_augment = np.random.randint(10)
            if do_augment:
                yield seq.augment_images(x), y
            else:
                yield x, y
        else:
            yield x, y


def train_model(model: Model, metrics: F1Metrics, dataset: Dataset, model_filename: str,
                batch_size=BATCH_SIZE,
                num_train_examples=(105678 * TRAIN_RATIO),
                num_valid_samples=(105678 * VALIDATE_RATIO),
                # num_train_examples=29450 // 4,
                # num_valid_samples=1550,
                epochs=100,):
    callbacks = create_callbacks(dataset, metrics, model_filename)
    steps_per_epoch = num_train_examples // batch_size
    # steps_per_epoch = 200

    # print(f"class_weights:{class_weight}")
    validation_steps = num_valid_samples // batch_size
    model.fit_generator(generator=next_dataset(dataset, batch_size, DataType.train),
                        epochs=epochs,
                        validation_data=next_dataset(dataset, batch_size, DataType.validate),
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        # class_weight=class_weight,
                        callbacks=callbacks, verbose=1)


def predict(data_unit: DataUnit, model: Model):
    x = create_unit_dataset(data_unit, TEST_DIR)
    # print(f"x:{x.shape}")
    # predict
    result = model.predict(np.array([x]))
    predicted = np.round(result)
    return predicted


def eval_model(model: Model, dataset: TestDataset):
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
