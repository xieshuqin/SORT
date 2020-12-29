import torch
import torchvision
import torch.nn as nn

from scipy.optimize import linear_sum_assignment as linear_assignment

from KF import ConstantVelocityKalmanFilter as KalmanFilter


def tracker2box(trackers):
    return torch.stack([t.state2box() for t in trackers])


def compute_iou(bboxes1, bboxes2):
    bboxes1 = bboxes1.unsqueeze(1)
    bboxes2 = bboxes2.unsqueeze(0)

    inter_x1 = torch.maximum(bboxes1[..., 0], bboxes2[..., 0])
    inter_y1 = torch.maximum(bboxes1[..., 1], bboxes2[..., 1])
    inter_x2 = torch.minimum(bboxes1[..., 2], bboxes2[..., 2])
    inter_y2 = torch.minimum(bboxes1[..., 3], bboxes2[..., 3])
    intersection = torch.clamp_min(inter_x2 - inter_x1 + 1, 0) * torch.clamp_min(inter_y2 - inter_y1 + 1, 0)

    area1 = (bboxes1[..., 2] - bboxes1[..., 0] + 1) * (bboxes1[..., 3] - bboxes1[..., 1] + 1)
    area2 = (bboxes2[..., 2] - bboxes2[..., 0] + 1) * (bboxes2[..., 3] - bboxes2[..., 1] + 1)
    union = area1 + area2 - intersection
    return intersection / union


def predict(trackers):
    for t in trackers:
        t.predict()


def update(trackers, observations):
    for t, o in zip(trackers, observations):
        t.update(o)


class SORT(nn.Module):
    def __init__(self):
        super(SORT, self).__init__()

        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.trackers = []
        self.tracker_ids = {}
        self.num_objs = 0
        self.t = 0

        self.PEDESTRIAN_CLS_ID = 1
        self.SCORE_THRESH = 0.9
        self.IOU_MIN = 0.3

    def forward(self, x):
        self.t += 1
        with torch.no_grad():
            det_bboxes = self.detect(x)

        if self.t == 1:
            # Initial frame, initialize tracker with det bboxes
            self.trackers = self.create_trackers(det_bboxes)
        else:
            # t > 1, first predict, then associate tracker with observation, then update.
            predict(self.trackers)
            trackers, observations, T_inds, D_inds, iou = self.associate(det_bboxes, self.trackers)
            update(trackers, observations)

            # delete and create new trackers
            self.trackers = self.delete_trackers(self.trackers, T_inds)
            new_D_inds = [i for i in range(len(det_bboxes)) if i not in D_inds and max(iou[:, i]) < self.IOU_MIN]
            self.trackers += self.create_trackers(det_bboxes[new_D_inds])

        bboxes = tracker2box(self.trackers)
        ids = torch.Tensor([self.tracker_ids[t] for t in self.trackers])

        return torch.cat([bboxes, ids[..., None]], dim=1)

    def detect(self, x):
        outputs = self.detector(x)[0]
        boxes, labels, scores = outputs['boxes'], outputs['labels'], outputs['scores']

        # filter out non-person classes
        indices = torch.eq(labels, self.PEDESTRIAN_CLS_ID)
        boxes, labels, scores = boxes[indices], labels[indices], scores[indices]
        # filter out low score boxes
        indices = torch.gt(scores, self.SCORE_THRESH)
        boxes, labels, scores = boxes[indices], labels[indices], scores[indices]

        return boxes.cpu()

    def associate(self, det_bboxes, trackers):
        # Hungarian algorithm for association
        target_bboxes = tracker2box(trackers)
        iou = compute_iou(target_bboxes, det_bboxes)
        T_inds, D_inds = linear_assignment(iou.cpu().numpy(), maximize=True)
        T_inds, D_inds = torch.from_numpy(T_inds), torch.from_numpy(D_inds)

        # Filter low IoU matched
        valid = iou[T_inds, D_inds] > self.IOU_MIN
        T_inds = T_inds[valid]
        D_inds = D_inds[valid]

        trackers = [trackers[i] for i in T_inds]
        observations = [det_bboxes[i] for i in D_inds]

        return trackers, observations, T_inds, D_inds, iou

    def create_trackers(self, det_bboxes):
        # create trackers initialised by det bboxes
        trackers = [KalmanFilter() for _ in range(len(det_bboxes))]
        for t, o in zip(trackers, det_bboxes):
            t.initialize(o)
            self.num_objs += 1
            self.tracker_ids[t] = self.num_objs

        return trackers

    def delete_trackers(self, trackers, keep_inds):
        for i in range(len(trackers)):
            if i not in keep_inds:
                del self.tracker_ids[trackers[i]]
        trackers = [trackers[i] for i in keep_inds]
        return trackers