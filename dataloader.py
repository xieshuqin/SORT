import os
from typing import List

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets.folder import default_loader


def list_imgs(img_dir):
    instances = {}
    for root, _, fnames in sorted(os.walk(img_dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            frame_id = int(fname.strip('.jpg'))
            instances[frame_id] = path

    return instances


class MOTDataset(Dataset):
    def __init__(self, root, transform=None):
        super(MOTDataset, self).__init__()

        img_root = os.path.join(root, 'img1')
        self.img_paths = list_imgs(img_root)

        gt = np.loadtxt(os.path.join(root, 'gt', 'gt.txt'), delimiter=',')

        self.anns = []
        start, end = 0, 0
        while end < len(gt):
            frame_ann = []
            while end < len(gt) and gt[end, 0] == gt[start, 0]:
                frame_ann.append(gt[end, :6])
                end += 1
            self.anns.append(frame_ann)
            start = end

        self.loader = default_loader
        self.transform = transform

    def __getitem__(self, idx):
        ann = torch.Tensor(self.anns[idx])
        frame_ids, obj_ids, bboxes = ann[:, [0]], ann[:, [1]], ann[:, 2:6]

        # convert xywh to xyxy
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]

        ann = torch.cat([bboxes, obj_ids, frame_ids], dim=1)

        img = self.loader(self.img_paths[int(frame_ids[0])])
        if self.transform:
            img = self.transform(img)
        return img, ann

    def __len__(self):
        return len(self.anns)


TRAINING_DATASET_NAMES = ('ADL-Rundle-6',
                          'ADL-Rundle-8',
                          'ETH-Bahnhof',
                          'ETH-Pedcross2',
                          'ETH-Sunnyday',
                          'KITTI-13',
                          'KITTI-17',
                          'PETS09-S2L1',
                          'TUD-Campus',
                          'TUD-Stadtmitte',
                          'Venice-2')


def dataset_names():
    return TRAINING_DATASET_NAMES


def build_dataset(name):
    assert name in TRAINING_DATASET_NAMES, 'Invalid dataset name'

    transform = T.Compose([
        T.ToTensor(),
        T.ConvertImageDtype(torch.float32)
    ])
    dataset = MOTDataset(os.path.join('./data/train', name), transform=transform)
    return dataset


def build_dataloader(name):
    dataset = build_dataset(name)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return loader
