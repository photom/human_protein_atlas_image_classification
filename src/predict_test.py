import mpl_toolkits  # import before pathlib
import sys
from pathlib import Path

from tensorflow import set_random_seed

sys.path.append(Path(__file__).parent)
from model import *
from dataset import *
from metrics import *

np.random.seed(19)
set_random_seed(19)

OUTPUT_FILE = 'test_dataset_prediction.txt'
BASE_MODEL = 'resnet50'
# BASE_MODEL = 'vgg11'
# BASE_MODEL = 'incepstionresnetv2'
# BASE_MODEL = 'adams'
# BASE_MODEL = 'michel'
if BASE_MODEL == 'resnet50':
    create_model = create_model_resnet50_plain
elif BASE_MODEL == 'vgg19':
    create_model = create_model_vgg19_plain
elif BASE_MODEL == 'incepstionresnetv2':
    create_model = create_model_inceptionresnetv2_plain
elif BASE_MODEL == 'adams':
    create_model = create_model_adams
elif BASE_MODEL == 'michel':
    create_model = create_model_michel
else:
    raise Exception("unimplemented model")


def predict(data_unit: DataUnit, test_model: Model):
    x = create_unit_dataset(data_unit, TEST_DIR)
    result = test_model.predict(np.array([x]))
    # print(f"{result}")
    predicted = np.round(result)
    # result[result >= THRESHOLD] = 0
    # result[result < THRESHOLD] = 1
    # predicted = result
    if not any([1.0 == p for p in predicted[0]]):
        predicted = np.zeros(28)
        predicted[np.argmax(result)] = 1.0
    else:
        predicted = predicted[0]
    # print(f"{predicted}")
    return predicted


def main():
    # test_model = create_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, len(COLORS)))
    # test_model = create_model(input_shape=(IMAGE_SIZE // 2, IMAGE_SIZE // 2, len(COLORS)))
    # test_model = create_model(input_shape=(IMAGE_SIZE // 2, IMAGE_SIZE // 2, TRAIN_COLOR_NUM))
    test_model = create_model(input_shape=(192, 192, len(COLORS)))
    test_dataset = load_test_data()
    weight_param_path = f"model/{BASE_MODEL}.weights.best.hdf5"
    test_model.load_weights(weight_param_path)
    content = "Id,Predicted\n"
    for i in range(len(test_dataset.data_list)):
        data_unit = test_dataset.data_list[i]
        predicted = predict(data_unit, test_model)
        decoded = []
        for idx, result in enumerate(predicted):
            if result == 1:
                decoded.append(str(idx))
        decoded = " ".join(decoded)
        print(f"i:{i} predicted:{predicted} {decoded} ")
        content += f"{data_unit.uuid},{decoded}\n"

    with open(OUTPUT_FILE, 'w') as f:
        f.write(content)


if __name__ == "__main__":
    main()
