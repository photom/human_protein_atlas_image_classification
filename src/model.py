from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed, \
    Conv3D, Conv2D, Conv1D, Flatten, MaxPooling1D, MaxPooling3D, MaxPooling2D, \
    GlobalAveragePooling2D
from keras.layers import GRU, Bidirectional, BatchNormalization
from keras.layers import Input, ELU, Lambda
from keras.layers import Reshape
from keras.optimizers import Adam
from keras import backend as keras_backend
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19

from dataset import *
from train_model import *
from metrics import MacroF1Score


def elu(x, alpha=0.05):
    return K.elu(x, alpha)


def create_model_resnet_plain(input_shape, dropout=0.5, datatype: DataType = DataType.train):
    """
    Function creating the model's graph in Keras.
    loss: 0.8077 - f1_score: 0.4574 - val_loss: 0.8119 - val_f1_score: 0.4471
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    x_input = Input(shape=input_shape)
    x = x_input
    x = Conv2D(padding='same',
               filters=3, kernel_size=3, activation=elu)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    base_input = Input(shape=(256, 256, 3))
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=base_input)
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


def create_model_vgg19_plain(input_shape, dropout=0.5, datatype: DataType = DataType.train):
    """
    Function creating the model's graph in Keras.
    loss: 0.8077 - f1_score: 0.4574 - val_loss: 0.8119 - val_f1_score: 0.4471
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    x_input = Input(shape=input_shape)
    x = x_input
    x = Conv2D(padding='same',
               filters=3, kernel_size=3, activation=elu)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

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


def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    # print(f"y_true:{y_true.shape} number_dim:{number_dim} weights:{weights.shape}")
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0., 1.], y_true[:, i])
    return weights


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean(
            (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** y_true) * K.binary_crossentropy(y_true, y_pred),
            axis=-1)

    return weighted_loss


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    # f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


def build_model(model_filename, learning_rate=0.001,
                create_model=create_model_resnet_plain, ):
    model = create_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, len(COLORS)))
    if os.path.exists(model_filename):
        print(f"load weights: file={model_filename}")
        model.load_weights(model_filename)
    # metrics.binary_accuracy
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss=f1_loss, optimizer=opt,
                  metrics=[MacroF1Score()])
    return model
