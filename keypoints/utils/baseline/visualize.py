import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def visualize_action(point_cloud,
                     grasp_point,
                     funct_point,
                     funct_vect,
                     grasp_xy,
                     grasp_rz,
                     output='./visualize'):
    [point_cloud, grasp_point, funct_point, 
        funct_vect, grasp_xy] = [
            np.squeeze(input) for input in [
                point_cloud, grasp_point, funct_point,
                funct_vect, grasp_xy]]
    if grasp_xy[1] > 0:
        return
    grasp_xy = np.concatenate([grasp_xy, [0]])
    point_cloud = point_cloud - grasp_point
    rot_mat = np.array([[np.cos(grasp_rz), -np.sin(grasp_rz), 0],
                        [np.sin(grasp_rz), np.cos(grasp_rz), 0],
                        [0, 0, 1]])
    point_cloud = np.dot(point_cloud, rot_mat.T) + grasp_xy
    funct_point = np.dot(rot_mat, funct_point - grasp_point) + grasp_xy
    if not os.path.exists(output):
        os.mkdir(output)
    save_path = os.path.join(output,
            str(np.random.randint(1e+4)).zfill(4))
    plt.figure()
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], s=0.1)
    plt.scatter(funct_point[0], funct_point[1], s=10.0)
    plt.axis('equal')
    plt.savefig(save_path)
    plt.close()

    return
   

ACTION_DATA = 'data_action_hammer.hdf5'
KEYPOINT_DATA = 'data_hammer.hdf5'
MAX_OUTPUT = 100

f_action = h5py.File(ACTION_DATA, 'r')
f_keypoint = h5py.File(KEYPOINT_DATA, 'r')

point_cloud = f_action['pos_point_cloud']
action = f_action['pos_action']
keypoint = f_keypoint['pos_keypoints']

for index in range(MAX_OUTPUT):
    p = point_cloud[index]
    a = action[index]
    f_k = keypoint[index][1]
    f_v = keypoint[index][2]
    visualize_action(p, a[0:3], f_k, f_v, a[3:5], a[5])
