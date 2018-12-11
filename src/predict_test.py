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
BASE_MODEL = 'resnet50'
weight_param_path = f"model/{BASE_MODEL}.weights.best.hdf5"


def predict(data_unit: DataUnit, test_model: Model):
    x = create_unit_dataset(data_unit, TEST_DIR)
    # print(f"x:{x.shape}")
    # predict
    result = test_model.predict(np.array([x]))
    predicted = np.round(result)
    return predicted


def main():
    test_dataset = load_test_data()
    test_model = build_model(weight_param_path)
    content = "Id,Predicted\n"
    for i in range(len(test_dataset.data_list)):
        data_unit = test_dataset.data_list[i]
        predicted = predict(data_unit, test_model)
        data_unit = test_dataset.data_list[i]
        decoded = []
        for idx, result in enumerate(predicted[0]):
            if result == 1:
                decoded.append(str(idx))
        decoded = " ".join(decoded)
        content += f"{data_unit.uuid},{decoded}\n"

    with open(OUTPUT_FILE, 'w') as f:
        f.write(content)


if __name__ == "__main__":
    main()
