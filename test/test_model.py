import os
from PIL import Image

import torch
import torchvision.transforms as T

from model import SORT
from utils import plot_bbox


def test_sort():
    # img = Image.open('/Users/shuqin/Downloads/2DMOT2015/train/ETH-Sunnyday/img1/000001.jpg')
    # x = transform(img)[None]

    model = SORT()
    model.eval()

    transform = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float)
    ])

    img_list = ['000001.jpg', '000002.jpg', '000003.jpg']
    dir_path = '/Users/shuqin/Downloads/2DMOT2015/train/ETH-Sunnyday/img1'
    for img in img_list:
        img = os.path.join(dir_path, img)
        x = transform(Image.open(img))[None]
        bbox = model(x)

        img = x.cpu().numpy().squeeze().transpose(1, 2, 0)
        plot_bbox(img, bbox)


if __name__ == '__main__':
    test_sort()