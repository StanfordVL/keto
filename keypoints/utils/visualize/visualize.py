import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--point_cloud',
                    type=str,
                    required=True)

parser.add_argument('--keypoints',
                    type=str,
                    required=True)

parser.add_argument('--save',
                    type=str,
                    default='./figures')

parser.add_argument('--max_num',
                    type=str,
                    default='100')

args = parser.parse_args()

if not os.path.exists(args.save):
    os.mkdir(args.save)

data_names = os.listdir(args.point_cloud)

count = 0
for name in data_names:
    count = count + 1
    if count > int(args.max_num):
        break

    point_cloud = np.squeeze(
            np.load(open(os.path.join(
                args.point_cloud, name), 'rb')))
    keypoints = np.squeeze(
            np.load(open(os.path.join(
                args.keypoints, name), 'rb')))
    score = keypoints[0]
    keypoints = np.reshape(keypoints[1:], [-1, 3])

    plt.figure()
    plt.scatter(point_cloud[:, 0],
                point_cloud[:, 1],
                s=0.5, c='grey')
    plt.scatter(keypoints[0, 0],
                keypoints[0, 1],
                s=30, c='green')
    plt.scatter(keypoints[1, 0],
                keypoints[1, 1],
                s=30, c='darkred')
    if keypoints.shape[0] > 2:
        plt.scatter(keypoints[2:, 0] * 0.05 + keypoints[1, 0],
                    keypoints[2:, 1] * 0.05 + keypoints[1, 1],
                    s=30, c='darkred')
    plt.axis('equal')
    plt.title('score: {:.2f}'.format(score))
    plt.savefig(os.path.join(
        args.save, str(count).zfill(6)))
    plt.close()


