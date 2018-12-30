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
from keras_contrib.applications.resnet import ResNet152
from keras_contrib.layers import advanced_activations

from dataset import *
from train_model import *
from metrics import MacroF1Score


def elu(x, alpha=0.05):
    return K.elu(x, alpha)


def create_model_resnet50_plain(input_shape, dropout=0.3, datatype: DataType = DataType.train):
    """
    Function creating the model's graph in Keras.
    loss: 0.8887 - f1_score: 0.2217 - val_loss: 0.9394 - val_f1_score: 0.1681, test_macro_f1_score=0.190476165079368
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    x_input = Input(name='my_input', shape=input_shape)
    x = x_input
    # x = Conv2D(name='my_conf', padding='same', filters=3, kernel_size=3)(x)
    # x = MaxPooling2D(name='my_max_pool', pool_size=(2, 2), strides=2, padding='same')(x)
    x = BatchNormalization(name='my_batch')(x)

    # base_input = Input(shape=(256, 256, 3))
    # base_input = Input(shape=(IMAGE_SIZE//2, IMAGE_SIZE//2, 3))
    # base_input = BatchNormalization()(x_input)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x_input)
    # base_model = ResNet50(weights=None, include_top=False, input_tensor=base_input,
    #                       pooling='avg')
    # base_model.layers.pop()
    # base_model.layers.pop()
    # base_model.summary()
    x = base_model(x)

    # while base_model.layers[-1].name != 'activation_24':
    # for _ in range(len(base_model.layers) // 2):
    #     base_model.layers.pop()
    # base_model.layers.pop(0)
    # base_model.summary()
    # base_model.layers.pop()
    # x_classify = base_model.outputs[0]
    # x_classify = base_model.get_layer('activation_28').output
    # x_classify = GlobalAveragePooling2D()(x_classify)
    # if datatype != DataType.test:
    #     x_classify = Dropout(dropout)(x_classify)
    # x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(CLASS_NUM, activation='sigmoid')(x)
    # x_classify = Dense(CLASS_NUM, activation='sigmoid')(x_classify)
    model = Model(x_input, x)
    model.layers[2].trainable = False
    # x = Dropout(0.5)(x)
    # freeze base_model layers
    # old_input = Input(shape=input_shape)
    # old_model = ResNet50(weights='imagenet', include_top=False, input_tensor=old_input,
    #                      pooling=None)
    model.summary()
    # for idx, layer in enumerate(model.layers):
    #     if 2 < idx < len(model.layers) - 6 and layer.name.find('my_') == -1:
    #         layer.set_weights(old_model.layers[idx].get_weights())
    #         # layer.trainable = False
    return model


def create_model_resnet152_plain(input_shape, dropout=0.5, datatype: DataType = DataType.train):
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
               filters=3, kernel_size=3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # base_input = Input(shape=(256, 256, 3))
    base_model = ResNet152(input_shape=(IMAGE_SIZE//2, IMAGE_SIZE//2, 3), classes=1)
    base_model.load_weights('resnet152_weights_tf.h5')
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


def create_model_inceptionresnetv2_plain(input_shape, dropout=0.5, datatype: DataType = DataType.train):
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
    # x = Conv2D(name='my_conf', padding='same', filters=64, kernel_size=3)(x)
    # x = Conv2D(name='my_conf2', padding='same', filters=3, kernel_size=2)(x)
    # x = MaxPooling2D(name='my_max_pool', pool_size=(2, 2), strides=2, padding='same')(x)
    # x = BatchNormalization(name='my_batch')(x)

    # base_input = Input(shape=(IMAGE_SIZE//2, IMAGE_SIZE//2, 3))
    base_model = InceptionResNetV2(weights=None, include_top=False, input_tensor=x,
                                   pooling='avg')
    base_model.summary()
    # base_model.layers.pop(0)
    # base_model.summary()
    x_classify = base_model(x)
    # if datatype != DataType.test:
    #     x_classify = Dropout(dropout)(x_classify)
    x_classify = Dense(CLASS_NUM, activation='sigmoid')(x_classify)
    model = Model(inputs=[x_input], outputs=[x_classify])

    # freeze base_model layers
    old_input = Input(shape=input_shape)
    old_model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=old_input,
                                  pooling=None)
    model.summary()
    for idx, layer in enumerate(model.layers):
        if 2 < idx < len(model.layers) - 6 and layer.name.find('my_') == -1:
            layer.set_weights(old_model.layers[idx].get_weights())
            # layer.trainable = False
    return model


def create_model_vgg19_plain(input_shape, dropout=0.5, datatype: DataType = DataType.train):
    """
    Function creating the model's graph in Keras.
     loss: 0.8949 - f1_score: 0.1521 - val_loss: 0.8938 - val_f1_score: 0.1493 - test macro_f1_score=0.16140349093998343
    loss: 0.8077 - f1_score: 0.4574 - val_loss: 0.8119 - val_f1_score: 0.4471
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


def create_model_adams(input_shape, dropout=0.5, datatype: DataType = DataType.train):
    """
    Function creating the model's graph in Keras.
    loss: 0.8077 - f1_score: 0.4574 - val_loss: 0.8119 - val_f1_score: 0.4471
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    pretrain_model = InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape)

    input_x = Input(shape=input_shape)
    x = input_x
    x = BatchNormalization()(x)
    x = pretrain_model(x)
    x = Conv2D(128, kernel_size=(1, 1), activation='relu')(x)
    x = Flatten()(x)
    # x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    # x = Dropout(0.5)(x)
    output = Dense(CLASS_NUM, activation='sigmoid')(x)
    model = Model(inputs=[input_x], outputs=[output])

    model.layers[2].trainable = False
    model.summary()
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


def f1_adams(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float')
    y_pred = K.cast(y_pred, 'float')
    # tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    # fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    # fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
    tp = K.sum(y_true * y_pred, axis=-1)
    tn = K.sum((1 - y_true) * (1 - y_pred), axis=-1)
    fp = K.sum((1 - y_true) * y_pred, axis=-1)
    fn = K.sum(y_true * (1 - y_pred), axis=-1)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    # f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    print(f"y_true:{y_true.get_shape().as_list()} y_pred:{y_pred.get_shape().as_list()} tp:{tp.get_shape().as_list()} f1:{f1.get_shape().as_list()}")
    return 1 - K.mean(f1, axis=-1)


def focal_loss(gamma=2., alpha=.25):
    # https://github.com/mkocabas/focal-loss-keras
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        epsilon = K.constant(K.epsilon(), dtype="float32")
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + epsilon)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + epsilon))
    return focal_loss_fixed


def build_model(model, model_filename, learning_rate=0.00005):
    if model_filename and os.path.exists(model_filename):
        print(f"load weights: file={model_filename}")
        model.load_weights(model_filename)
    # metrics.binary_accuracy
    opt = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  # loss=focal_loss(),
                  # loss=f1_loss,
                  # metrics=["accuracy", true_positives, possible_positives, predicted_positives],)
                  # metrics=['accuracy', f1_adams], )
                  # metrics=[MacroF1Score()], )
                  metrics=['accuracy'], )
    return model
