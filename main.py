import os
import numpy as np
import torch

from dataloader import build_dataloader, dataset_names
from model import SORT


def main(name):
    dataloader = build_dataloader(name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SORT().to(device)
    model.eval()

    outputs = []
    for (x, y) in dataloader:
        x = x.to(device)
        tracker_outs = model(x)

        # reformat bboxes from xyxy to xywh
        bboxes = tracker_outs[:, :4]
        bboxes[:, 2] -= bboxes[:, 0]
        bboxes[:, 3] -= bboxes[:, 1]

        # generate final output format
        tracker_ids = tracker_outs[:, [4]]
        frame_id = y[0, 0, -1] * torch.ones_like(tracker_ids)
        dummy = torch.ones_like(bboxes)
        output = torch.cat([frame_id, tracker_ids, bboxes, dummy], dim=1)
        outputs.append(output)

    outputs = torch.cat(outputs, dim=0).cpu().numpy()
    out_dir = './results'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savetxt(os.path.join(out_dir, '%s.txt' % name), outputs, fmt='%.1f', delimiter=',', )


if __name__ == '__main__':
    for name in dataset_names():
        main(name)