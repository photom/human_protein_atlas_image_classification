from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed, \
    Conv3D, Conv2D, Conv1D, Flatten, MaxPooling1D, MaxPooling3D, MaxPooling2D, \
    GlobalAveragePooling2D
from keras.layers import GRU, Bidirectional, BatchNormalization
from keras.layers import Input, ELU, Lambda
from keras.layers import Reshape
from keras.optimizers import Adam, Nadam
from keras import backend as keras_backend
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras_contrib.applications import resnet
from keras_contrib.applications.resnet import ResNet152, ResNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras_contrib.layers import advanced_activations
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, Concatenate, ReLU, LeakyReLU
import tensorflow as tf

from dataset import *
from train_model import *
from metrics import MacroF1Score


def elu(x, alpha=0.05):
    return K.elu(x, alpha)


def create_model_resnet50_plain(input_shape, dropout=0.3, datatype: DataType = DataType.train):
    """
    Function creating the model's graph in Keras.
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    x_input = Input(name='my_input', shape=input_shape)
    x = x_input
    x = Conv2D(name='my_conf2', padding='valid', strides=2, filters=3, kernel_size=(5, 5))(x)
    # x = Conv2D(name='my_conf1', padding='same', filters=30, kernel_size=(2, 2))(x)
    x = BatchNormalization()(x)
    # x = ReLU()(x)
    # x = Conv2D(name='my_conf2', padding='same', filters=3, kernel_size=(2, 2))(x)
    # x = BatchNormalization()(x)

    base_input = Input(shape=(IMAGE_SIZE//2, IMAGE_SIZE//2, 3))
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=base_input)
    # base_model.summary()
    base_model.layers.pop(0)
    base_model.layers.pop(0)
    base_model.summary()
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    if datatype != DataType.test:
        x = Dropout(dropout)(x)
    x = Dense(CLASS_NUM, activation='sigmoid')(x)
    model = Model(inputs=[x_input], outputs=[x])
    model.summary()
    return model


def create_model_resnet152_plain(input_shape, dropout=0.5, datatype: DataType = DataType.train):
    """
    Function creating the model's graph in Keras.
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    x_input = Input(name='my_input', shape=input_shape)
    x = x_input
    x = Conv2D(name='my_conf', padding='same', filters=3, kernel_size=(2, 2))(x)
    # x = MaxPooling2D(name='my_max_pool', pool_size=(2, 2), strides=2, padding='same')(x)
    x = BatchNormalization(name='my_batch')(x)

    base_input = Input(shape=(IMAGE_SIZE//2, IMAGE_SIZE//2, 3))
    base_model = ResNet((IMAGE_SIZE//2, IMAGE_SIZE//2, 3), block='bottleneck', repetitions=[3, 8, 36, 3], include_top=False)
    base_model.load_weights('resnet152_weights_tf.h5')
    # base_model.summary()
    base_model.layers.pop(0)

    x_classify = base_model(x)
    x_classify = GlobalAveragePooling2D()(x_classify)
    if datatype != DataType.test:
        x_classify = Dropout(dropout)(x_classify)
    x_classify = Dense(CLASS_NUM, activation='sigmoid')(x_classify)
    model = Model(inputs=[x_input], outputs=[x_classify])
    model.summary()
    return model


def create_model_inceptionresnetv2_plain(input_shape, dropout=0.5, datatype: DataType = DataType.train):
    """
    Function creating the model's graph in Keras.
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    x_input = Input(name='my_input', shape=input_shape)
    x = x_input
    x = Conv2D(name='my_conf', padding='same', filters=3, kernel_size=(2, 2))(x)
    # x = MaxPooling2D(name='my_max_pool', pool_size=(2, 2), strides=2, padding='same')(x)
    x = BatchNormalization(name='my_batch')(x)

    base_input = Input(shape=(IMAGE_SIZE//2, IMAGE_SIZE//2, 3))
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=base_input)
    base_model.summary()
    base_model.layers.pop(0)
    # base_model.summary()
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    if datatype != DataType.test:
        x = Dropout(dropout)(x)
    x = Dense(CLASS_NUM, activation='sigmoid')(x)
    # x_classify = Dense(CLASS_NUM, activation='sigmoid')(x_classify)
    model = Model(inputs=[x_input], outputs=[x])
    model.summary()
    return model


def create_model_vgg19_plain(input_shape, dropout=0.5, datatype: DataType = DataType.train):
    """
    Function creating the model's graph in Keras.
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    x_input = Input(shape=input_shape)
    x = x_input
    x = Conv2D(padding='same',
               filters=3, kernel_size=3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = BatchNormalization()(x)

    base_input = Input(shape=(256, 256, 3))
    base_model = VGG19(weights='imagenet', include_top=False, input_tensor=base_input)
    # base_model.summary()
    base_model.layers.pop(0)

    x_classify = base_model(x)
    x_classify = GlobalAveragePooling2D()(x_classify)
    if datatype != DataType.test:
        x_classify = Dropout(dropout)(x_classify)
    x_classify = Dense(CLASS_NUM, activation='sigmoid')(x_classify)
    model = Model(inputs=[x_input], outputs=[x_classify])
    # freeze base_model layers
    for layer in base_model.layers:
        layer.trainable = False
    # model.summary()
    return model


def create_model_mobilenet(input_shape, dropout=0.5, datatype: DataType = DataType.train):
    """
    Function creating the model's graph in Keras.
    loss: 0.8077 - f1_score: 0.4574 - val_loss: 0.8119 - val_f1_score: 0.4471
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    x_input = Input(name='my_input', shape=input_shape)
    x = x_input
    x = Conv2D(name='my_conf', padding='same', filters=3, kernel_size=(2, 2))(x)
    x = BatchNormalization(name='my_batch')(x)

    base_input = Input(shape=(IMAGE_SIZE//2, IMAGE_SIZE//2, 3))
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=base_input)
    base_model.summary()
    base_model.layers.pop(0)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    if datatype != DataType.test:
        x = Dropout(dropout)(x)
    x = Dense(CLASS_NUM, activation='sigmoid')(x)
    model = Model(inputs=[x_input], outputs=[x])
    model.summary()
    return model


def f1_loss(y_true, y_pred):
    # THRESHOLD = 0.05
    # y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


def focal_loss(y_true, y_pred, gamma=2):
    # transform back to logits
    _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred = tf.log(y_pred / (1 - y_pred))
    y_pred = tf.cast(y_pred, tf.float32)

    max_val = K.clip(-y_pred, 0, 1)
    loss = y_pred - y_pred * y_true + max_val + K.log(K.exp(-max_val) + K.exp(-y_pred - max_val))
    invprobs = tf.log_sigmoid(-y_pred * (y_true * 2.0 - 1.0))
    loss = K.exp(invprobs * gamma) * loss

    return K.mean(K.sum(loss, axis=1))


def focal_f1_combined_loss(y_true, y_pred, alpha=0.5):
    return alpha * focal_loss(y_true, y_pred) + (1 - alpha) * f1_loss(y_true, y_pred)


def build_model(model: Model, metrics: F1Metrics, dataset: Dataset = None,
                model_filename: str = None, learning_rate=0.001):
    if model_filename and os.path.exists(model_filename):
        print(f"load weights: file={model_filename}")
        model.load_weights(model_filename)
    opt = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt,
                  # loss='binary_crossentropy',
                  loss=focal_f1_combined_loss,
                  metrics=['acc', metrics.f1_macro], )
    return model
