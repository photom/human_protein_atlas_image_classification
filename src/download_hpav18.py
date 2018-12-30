import sys
import requests
import pandas as pd
from PIL import Image
import os

COLORS = ['red', 'green', 'blue', 'yellow']
DIR = "../hpaic/input/HPAv18/"
V18_URL = 'http://v18.proteinatlas.org/images/'

imgList = pd.read_csv(sys.argv[1])

IMAGE_SIZE = 512
sample_num = 3
# for item_id in imgList['Id'][:sample_num]:
for item_id in imgList['Id']:
    segments = item_id.split('_')
    for color in COLORS:
        img_path = segments[0] + '/' + "_".join(segments[1:]) + "_" + color + ".jpg"
        img_jpg = item_id + "_" + color + ".jpg"
        img_url = V18_URL + img_path
        response = requests.get(img_url, allow_redirects=True)
        open(DIR + img_jpg, 'wb').write(response.content)
        img_obj = Image.open(DIR + img_jpg).convert('L')
        img_obj = img_obj.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        img_png = item_id + "_" + color + ".png"
        img_obj.save(DIR + img_png)
        os.remove(DIR + img_jpg)
