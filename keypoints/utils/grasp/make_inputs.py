import os
import h5py
import numpy as np

point_cloud_dir = '../point_cloud'
grasp_4dof_dir = '../grasp_4dof'
save_path = './data.hdf5'

pos_point_cloud = []
pos_grasp = []
neg_point_cloud = []
neg_grasp = []

scale = 20

data_list = os.listdir(point_cloud_dir)

for data_name in data_list:

    with open(os.path.join(
            point_cloud_dir, data_name), 'rb') as f:
        point_cloud = np.load(f)

    with open(os.path.join(
            grasp_4dof_dir, data_name), 'rb') as f:
        grasp_4dof = np.load(f)

    grasp = grasp_4dof

    if grasp[0] > 0.002:
        pos_point_cloud.append(
            point_cloud[np.newaxis])
        pos_grasp.append(grasp[np.newaxis, 1:])

    else:
        neg_point_cloud.append(
            point_cloud[np.newaxis])
        neg_grasp.append(grasp[np.newaxis, 1:])

scale_grasp = np.reshape([scale, scale, scale, 1, 1, 1], (1, 6))

pos_point_cloud = np.concatenate(pos_point_cloud, axis=0) * scale
pos_grasp = np.concatenate(pos_grasp, axis=0) * scale_grasp

neg_point_cloud = np.concatenate(neg_point_cloud, axis=0) * scale
neg_grasp = np.concatenate(neg_grasp, axis=0) * scale_grasp

for converted_data in [pos_point_cloud,
                       pos_grasp, neg_point_cloud, neg_grasp]:
    print('shape: {}'.format(converted_data.shape))

with h5py.File(save_path, 'w') as f:
    f.create_dataset('pos_point_cloud', data=pos_point_cloud)
    f.create_dataset('pos_grasp', data=pos_grasp)
    f.create_dataset('neg_point_cloud', data=neg_point_cloud)
    f.create_dataset('neg_grasp', data=neg_grasp)
    f.close()
