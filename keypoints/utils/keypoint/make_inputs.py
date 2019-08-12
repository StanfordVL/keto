import os
import h5py
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    '--point_cloud',
    type=str)

parser.add_argument(
    '--keypoints',
    type=str)

parser.add_argument(
    '--grasps',
    type=str)

parser.add_argument(
    '--save',
    type=str,
    default='./data.hdf5')

args = parser.parse_args()

point_cloud_dir = args.point_cloud
keypoints_dir = args.keypoints
grasps_dir = args.grasps
save_path = args.save

pos_point_cloud = []
neg_point_cloud = []

pos_actions = []
neg_actions = []

scale = 20
debug = True
max_plot = 100

data_list = os.listdir(point_cloud_dir)

for data_name in data_list:

    p = np.load(open(os.path.join(
        point_cloud_dir, data_name), 'rb'))

    k = np.load(open(os.path.join(
        keypoints_dir, data_name), 'rb')).flatten()
    
    g = np.load(open(os.path.join(
        grasps_dir, data_name), 'rb')).flatten()

    action = np.concatenate([np.reshape(g[1:], [1, 2, 3]),
        np.reshape(k[1:], [1, 2, 3])], axis=1)

    if k[0] == 2.0:
        pos_point_cloud.append(np.reshape(p, [1, 1024, 3]))
        pos_actions.append(action)

    elif k[0] < 2.0:
        neg_point_cloud.append(np.reshape(p, [1, 1024, 3]))
        neg_actions.append(action)

    else:
        raise ValueError

rescale = np.reshape([scale, 1.0, scale, scale], [1, 4, 1])

pos_point_cloud = np.concatenate(pos_point_cloud, axis=0) * scale
neg_point_cloud = np.concatenate(neg_point_cloud, axis=0) * scale

pos_actions = np.concatenate(pos_actions, axis=0) * rescale
neg_actions = np.concatenate(neg_actions, axis=0) * rescale

print('Found new data')
for name, converted_data in [('Pos PC', pos_point_cloud),
                             ('Neg PC', neg_point_cloud),
                             ('Pos AT', pos_actions),
                             ('Neg AT', neg_actions)]:
    print('{} shape: {}'.format(name, converted_data.shape))


if not os.path.exists(save_path):
    print('Creating new file')
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('pos_point_cloud', data=pos_point_cloud,
                         maxshape=(None, 1024, 3))
        f.create_dataset('neg_point_cloud', data=neg_point_cloud,
                         maxshape=(None, 1024, 3))
        f.create_dataset('pos_actions', data=pos_actions,
                         maxshape=(None, 4, 3))
        f.create_dataset('neg_actions', data=neg_actions,
                         maxshape=(None, 4, 3))
        f.close()

else:
    print('Appending to existing file')
    with h5py.File(save_path, 'a') as f:

        dataset = f['pos_point_cloud']
        shape = [dataset.shape[0] + pos_point_cloud.shape[0], 1024, 3]
        dataset.resize(shape)
        dataset[-pos_point_cloud.shape[0]:] = pos_point_cloud
        print('Pos PC shape: {}'.format(shape))

        dataset = f['neg_point_cloud']
        shape = [dataset.shape[0] + neg_point_cloud.shape[0], 1024, 3]
        dataset.resize(shape)
        dataset[-neg_point_cloud.shape[0]:] = neg_point_cloud
        print('Neg PC shape: {}'.format(shape))

        dataset = f['pos_actions']
        shape = [dataset.shape[0] + pos_actions.shape[0], 4, 3]
        dataset.resize(shape)
        dataset[-pos_actions.shape[0]:] = pos_actions
        print('Pos AT shape: {}'.format(shape))

        dataset = f['neg_actions']
        shape = [dataset.shape[0] + neg_actions.shape[0], 4, 3]
        dataset.resize(shape)
        dataset[-neg_actions.shape[0]:] = neg_actions
        print('Neg AT shape: {}'.format(shape))
