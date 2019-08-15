import os
import h5py
import argparse
import numpy as np

from  progress.bar import ShadyBar

parser = argparse.ArgumentParser()

parser.add_argument(
    '--point_cloud',
    type=str)

parser.add_argument(
    '--keypoints',
    type=str)

parser.add_argument(
    '--num_keypoints',
    type=str,
    default='2')

parser.add_argument(
    '--save',
    type=str,
    default='./data.hdf5')

args = parser.parse_args()

point_cloud_dir = args.point_cloud
keypoints_dir = args.keypoints
save_path = args.save
num_keypoints = int(args.num_keypoints)

pos_point_cloud = []
neg_point_cloud = []

pos_keypoints = []
neg_keypoints = []

scale = 20
debug = True
max_plot = 100

data_list = os.listdir(point_cloud_dir)

bar = ShadyBar('Processing inputs', max=len(data_list))

for data_name in data_list:
    p = np.load(open(os.path.join(
        point_cloud_dir, data_name), 'rb'))
    k = np.load(open(os.path.join(
        keypoints_dir, data_name), 'rb'))
    k = np.reshape(k, [-1])

    if k[0] < 0:
        continue

    elif k[0] == 2.0:
        pos_point_cloud.append(np.reshape(p, [1, 1024, 3]))
        pos_keypoints.append(np.reshape(k[1:], [1, num_keypoints, 3]))

    elif k[0] < 2.0:
        neg_point_cloud.append(np.reshape(p, [1, 1024, 3]))
        neg_keypoints.append(np.reshape(k[1:], [1, num_keypoints, 3]))

    else:
        raise ValueError

    bar.next()
bar.finish()

pos_point_cloud = np.concatenate(pos_point_cloud, axis=0) * scale
neg_point_cloud = np.concatenate(neg_point_cloud, axis=0) * scale

stack_scale = np.ones(shape=[1, num_keypoints, 3])
stack_scale[:, :2] = scale

pos_keypoints = np.concatenate(pos_keypoints, axis=0) * stack_scale
neg_keypoints = np.concatenate(neg_keypoints, axis=0) * stack_scale

print('Found new data')
for name, converted_data in [('Pos PC', pos_point_cloud),
                             ('Neg PC', neg_point_cloud),
                             ('Pos KP', pos_keypoints),
                             ('Neg KP', neg_keypoints)]:
    print('{} shape: {}'.format(name, converted_data.shape))


if not os.path.exists(save_path):
    print('Creating new file')
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('pos_point_cloud', data=pos_point_cloud,
                         maxshape=(None, 1024, 3))
        f.create_dataset('neg_point_cloud', data=neg_point_cloud,
                         maxshape=(None, 1024, 3))
        f.create_dataset('pos_keypoints', data=pos_keypoints,
                         maxshape=(None, num_keypoints, 3))
        f.create_dataset('neg_keypoints', data=neg_keypoints,
                         maxshape=(None, num_keypoints, 3))
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

        dataset = f['pos_keypoints']
        shape = [dataset.shape[0] + pos_keypoints.shape[0], num_keypoints, 3]
        dataset.resize(shape)
        dataset[-pos_keypoints.shape[0]:] = pos_keypoints
        print('Pos KP shape: {}'.format(shape))

        dataset = f['neg_keypoints']
        shape = [dataset.shape[0] + neg_keypoints.shape[0], num_keypoints, 3]
        dataset.resize(shape)
        dataset[-neg_keypoints.shape[0]:] = neg_keypoints
        print('Neg KP shape: {}'.format(shape))
