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


class SORT(nn.Module):
    def __init__(self):
        super(SORT, self).__init__()

        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.trackers = []
        self.tracker_ids = {}
        self.num_objs = 0
        self.t = 0

        self.PEDESTRIAN_CLS_ID = 1
        self.SCORE_THRESH = 0.5
        self.IOU_MIN = 0.3

    def forward(self, x):
        self.t += 1

        with torch.no_grad():  # only do inference
            det_bboxes = self.detect(x)
            self.trackers, observations = self.get_tracker_and_observation(det_bboxes)
            self.trackers = self.filter(self.trackers, observations)

            bboxes = tracker2box(self.trackers)
            ids = torch.Tensor([self.tracker_ids[t] for t in self.trackers])
        return torch.cat([bboxes, ids[..., None]], dim=1)

    def detect(self, x):
        outputs = self.detector(x)[0]
        boxes, labels, scores = outputs['boxes'], outputs['labels'], outputs['scores']

        # filter out non-person classes
        indices = torch.eq(labels, self.PEDESTRIAN_CLS_ID)  # TODO: Verify PEDESTRIAN_ID
        boxes, labels, scores = boxes[indices], labels[indices], scores[indices]

        # filter out low score boxes
        indices = torch.gt(scores, self.SCORE_THRESH)
        boxes, labels, scores = boxes[indices], labels[indices], scores[indices]

        return boxes.cpu()

    def get_tracker_and_observation(self, det_bboxes):
        if self.t == 1:
            # Initial frame, set tracker to det bboxes
            self.trackers = [KalmanFilter() for _ in range(len(det_bboxes))]
            observations = det_bboxes
            for t in self.trackers:
                self.num_objs += 1
                self.tracker_ids[t] = self.num_objs
        else:
            # Consecutive frames
            T_inds, D_inds, iou = self.associate(det_bboxes, self.trackers)
            self.trackers, observations = self.create_and_delete_tracker(self.trackers, det_bboxes, T_inds, D_inds, iou)
        return self.trackers, observations

    def associate(self, det_bboxes, trackers):
        # Hungarian algorithm for association
        target_bboxes = tracker2box(trackers)
        iou = compute_iou(target_bboxes, det_bboxes)
        T_inds, D_inds = linear_assignment(iou, maximize=True)

        # Filter low IoU matched
        valid = iou[T_inds, D_inds] > self.IOU_MIN
        T_inds = T_inds[valid]
        D_inds = D_inds[valid]

        return T_inds, D_inds, iou

    def create_and_delete_tracker(self, trackers, det_bboxes, T_inds, D_inds, iou):
        # Deletion
        for i in range(len(trackers)):
            if i not in T_inds:
                del self.tracker_ids[trackers[i]]
        trackers = [trackers[i] for i in range(len(trackers)) if i in T_inds]
        observations = [det_bboxes[i] for i in D_inds]

        # Creation
        new_D_inds = [i for i in range(len(det_bboxes)) if i not in D_inds and max(iou[:, i]) < self.IOU_MIN]
        new_trackers = [KalmanFilter() for _ in new_D_inds]
        new_observation = [det_bboxes[i] for i in new_D_inds]
        for t in new_trackers:
            self.num_objs += 1
            self.tracker_ids[t] = self.num_objs

        trackers = trackers + new_trackers
        observations = observations + new_observation

        return trackers, observations

    def filter(self, trackers, observations):
        for t, o in zip(trackers, observations):
            t(o)
        return trackers
