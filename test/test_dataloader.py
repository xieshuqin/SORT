import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from dataloader import dataset_names, build_dataloader
from utils import plot_bbox


def test_dataloader():
    name = dataset_names()[0]
    dataloader = build_dataloader(name)
    for (img, bboxes) in dataloader:
        img = img[0].cpu().numpy().transpose(1, 2, 0)
        bboxes = bboxes[0]
        plot_bbox(img, bboxes)


if __name__ == '__main__':
    test_dataloader()