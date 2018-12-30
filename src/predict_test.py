import mpl_toolkits  # import before pathlib
import sys
from pathlib import Path

from tensorflow import set_random_seed

sys.path.append(Path(__file__).parent)
from model import *
from dataset import *

np.random.seed(19)
set_random_seed(19)

OUTPUT_FILE = 'test_dataset_prediction.txt'
# BASE_MODEL = 'resnet50'
# BASE_MODEL = 'vgg11'
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

weight_param_path = f"model/{BASE_MODEL}.weights.best.hdf5"


def predict(data_unit: DataUnit, test_model: Model):
    x = create_unit_dataset(data_unit, TEST_DIR)
    # print(f"x:{x.shape}")
    # predict
    result = test_model.predict(np.array([x]))
    # print(result)
    predicted = np.round(result)
    return predicted


def main():
    model = create_model(input_shape=(IMAGE_SIZE // 2, IMAGE_SIZE // 2, TRAIN_COLOR_NUM))
    # model = create_model(input_shape=(299, 299, TRAIN_COLOR_NUM))
    test_dataset = load_test_data()
    test_model = build_model(model, weight_param_path)
    content = "Id,Predicted\n"
    for i in range(len(test_dataset.data_list)):
        data_unit = test_dataset.data_list[i]
        predicted = predict(data_unit, test_model)
        decoded = []
        for idx, result in enumerate(predicted[0]):
            if result == 1:
                decoded.append(str(idx))
        decoded = " ".join(decoded)
        print(f"i:{i} decoded:{decoded} {data_unit.uuid},{decoded} ")
        content += f"{data_unit.uuid},{decoded}\n"

    with open(OUTPUT_FILE, 'w') as f:
        f.write(content)


# model = create_model(input_shape=(IMAGE_SIZE//2, IMAGE_SIZE//2, TRAIN_COLOR_NUM))
# test_model = build_model(model, weight_param_path)
def sample(uuid):
    # test_model = build_model(weight_param_path, create_model=create_model)
    traindir = "../hpaic/input/train"
    data_unit = DataUnit(f"{uuid}", "16 0", traindir)
    x = create_unit_dataset(data_unit, traindir)
    # print(f"x:{x.shape}")
    # predict
    result = test_model.predict(np.array([x]))
    print(result)
    predicted = np.round(result)
    print(predicted)


if __name__ == "__main__":
    main()
    # sample('00070df0-bbc3-11e8-b2bc-ac1f6b6435d0')  # 16 0
    # sample('000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0')  # 7 1 2 0
