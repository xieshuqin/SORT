import numpy as np
import torch
from filterpy.kalman import KalmanFilter
from KF import ConstantVelocityKalmanFilter as KF, box2state

def test_KF(bbox):
    z = box2state(bbox).numpy()[..., None]

    # Ground Truth KF
    # define constant velocity model
    kf = KalmanFilter(dim_x=7, dim_z=4)
    kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

    kf.R[2:, 2:] *= 10.
    kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
    kf.P *= 10.
    kf.Q[-1, -1] *= 0.01
    kf.Q[4:, 4:] *= 0.01
    kf.x[:4] = z

    # My KF
    my_kf = KF()
    my_kf(bbox)

    for t in range(5):
        bbox += 10

        z = box2state(bbox).numpy()[..., None]
        kf.predict()
        kf.update(z)

        my_kf(bbox)

        print('kf x ', kf.x.squeeze())
        print('my_kf x', my_kf.X)


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=4)
    torch.set_printoptions(precision=4, sci_mode=False)
    bbox = np.array([50, 70, 300, 600])
    test_KF(bbox)