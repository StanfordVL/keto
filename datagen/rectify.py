import numpy as np
import argparse
from sklearn import linear_model

parser = argparse.ArgumentParser(description='')

parser.add_argument('--point_cloud',
                    type=str,
                    required=True)

args = parser.parse_args()

point_cloud = np.squeeze(np.load(open(args.point_cloud, 'rb')))
point_cloud = point_cloud - np.mean(point_cloud, axis=0, keepdims=True)
xs, ys, _ = np.split(point_cloud, [1, 2], axis=1)

ransac = linear_model.RANSACRegressor()
ransac.fit(xs, ys)
k = np.squeeze(ransac.estimator_.coef_)

cos = 1 / np.sqrt(1 + np.square(k))
sin = cos * k

rot_mat = np.array([[cos, -sin, 0],
                    [sin, cos, 0],
                    [0, 0, 1]])
point_cloud = point_cloud.dot(rot_mat)

np.save(open(args.point_cloud, 'wb'), point_cloud)

