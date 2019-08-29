import numpy as np


def match_keypoints(point_cloud, 
                    point_cloud_data,
                    keypoints_data,
                    max_iter=600):
    dist_record = 10
    index_record = -1
    point_cloud = np.reshape(point_cloud, [1024, 1, 3])
    point_cloud = point_cloud - np.mean(
            point_cloud, axis=0, keepdims=True)
    for index in range(max_iter):
        point_cloud_curr = np.squeeze(
                np.copy(point_cloud_data[index]))
        point_cloud_curr = point_cloud_curr - np.mean(
                point_cloud_curr, axis=0, keepdims=True)
        dist = point_cloud - point_cloud_curr
        dist = np.linalg.norm(dist, axis=2)
        dist = np.mean(np.amin(dist, axis=0)
                ) + np.mean(np.amin(dist, axis=1))
        if dist < dist_record:
            dist_record = dist
            index_record = index
    keypoints = np.squeeze(
            keypoints_data[index_record]).astype(np.float32)
    return np.split(keypoints, [1, 2], axis=0)
