import torch
import torch.nn as nn


def box2state(bbox):
    # compute state formatted as (cx, cy, s, r)
    # , with bbox format (x1,y1,x2,y2)
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    s = (x2 - x1 + 1) * (y2 - y1 + 1)
    r = (y2 - y1 + 1) / (x2 - x1 + 1)

    return torch.Tensor([cx, cy, s, r])


class ConstantVelocityKalmanFilter(nn.Module):
    def __init__(self):
        super(ConstantVelocityKalmanFilter, self).__init__()
        self.X = None
        self.Sigma = None

        # motion model
        self.A = torch.eye(7)
        self.A[[0, 1, 2], [4, 5, 6]] = 1
        self.R = torch.eye(7)
        self.R[-1, -1] *= 0.01
        self.R[4:, 4:] *= 0.01

        # observation model
        self.C = torch.cat([torch.eye(4), torch.zeros((4, 3))], dim=1)
        self.Q = torch.eye(4)
        self.Q[2:, 2:] *= 10

    def forward(self, bbox=None):
        self.predict()
        self.update(bbox)

    def predict(self):
        self.X = self.A @ self.X
        self.Sigma = self.A @ self.Sigma @ self.A.T + self.R

    def update(self, bbox):
        z = box2state(bbox)
        K = self.Sigma @ self.C.T @ torch.inverse(self.C @ self.Sigma @ self.C.T + self.Q)
        self.X = self.X + K @ (z - self.C @ self.X)
        self.Sigma = (torch.eye(7) - K @ self.C) @ self.Sigma

    def initialize(self, bbox):
        z = box2state(bbox)
        velocities = torch.zeros(3)  # v_x, v_y, v_s
        X = torch.cat([z, velocities])

        Sigma = torch.eye(7)
        Sigma[4:, 4:] *= 1000 # high uncertainty for initial volecity
        Sigma *= 10

        self.X = X
        self.Sigma = Sigma

    def state2box(self):
        cx, cy, s, r = self.X[:4]
        h, w = torch.sqrt(s * r), torch.sqrt(s / r)

        x1 = cx - 0.5*(w-1)
        x2 = cx + 0.5*(w-1)
        y1 = cy - 0.5*(h+1)
        y2 = cy + 0.5*(h+1)

        if s * r < 0:
            return torch.zeros(4)

        return torch.Tensor([x1, y1, x2, y2])
