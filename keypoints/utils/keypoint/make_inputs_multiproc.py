import os
import h5py
import argparse
import numpy as np
from multiprocessing import Manager, Pool

parser = argparse.ArgumentParser()

parser.add_argument(
    '--point_cloud',
    type=str)

parser.add_argument(
    '--keypoints',
    type=str)

parser.add_argument(
    '--num_keypoints',
    type=int,
    default='3')

parser.add_argument(
    '--save',
    type=str,
    default='./data.hdf5')

args = parser.parse_args()

point_cloud_dir = args.point_cloud
keypoints_dir = args.keypoints
num_keypoints = args.num_keypoints
save_path = args.save

scale = 20

batch_size = 3000

data_list = os.listdir(point_cloud_dir)

data_sublists = [data_list[x:x + batch_size]
                 for x in range(0, len(data_list), batch_size)]


class Logger(object):
    """Logging utils."""  
    def __init__(self, output='./output.log'):
        """Initialization."""
        self.output = output
        return

    def write(self, message):
        """Writes the message to the log file."""
        with open(self.output, 'a') as f:
            f.write(message + '\r\n')
        return


logger = Logger(output=save_path+'.log')


def save_data(pos_point_cloud,
              neg_point_cloud,
              pos_keypoints,
              neg_keypoints,
              save_path,
              lock,
              scale=20):
    """Saves the data to hdf5 file.

    Args: 
        pos_point_cloud: The point cloud associated 
            associated with the positive keypoints.
        neg_point_cloud: The point cloud associated 
            associated with the negative keypoints.
        pos_grasp: The positive keypoints.
        neg_grasp: The negative keypoints.
        save_path: The hdf5 file name.
        scale: The constant to be multiplied with the
            point cloud and grasp coordinates to fit 
            the input scale of the network.

    Returns:
        None.
    """
    pos_point_cloud = np.concatenate(pos_point_cloud, axis=0) * scale
    neg_point_cloud = np.concatenate(neg_point_cloud, axis=0) * scale

    stack_scale = np.ones(shape=[1, num_keypoints, 3])
    stack_scale[:, :2] = scale

    pos_keypoints = np.concatenate(pos_keypoints, axis=0) * stack_scale
    neg_keypoints = np.concatenate(neg_keypoints, axis=0) * stack_scale

    logger.write('Found new data')
    for name, converted_data in [('Pos PC', pos_point_cloud),
                                 ('Neg PC', neg_point_cloud),
                                 ('Pos KP', pos_keypoints),
                                 ('Neg KP', neg_keypoints)]:
        logger.write('{} shape: {}'.format(name, converted_data.shape))

    lock.acquire()
    if not os.path.exists(save_path):
        logger.write('Creating new file')

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
        logger.write('Appending to existing file')
        with h5py.File(save_path, 'a') as f:

            dataset = f['pos_point_cloud']
            shape = [dataset.shape[0] + pos_point_cloud.shape[0], 1024, 3]
            dataset.resize(shape)
            dataset[-pos_point_cloud.shape[0]:] = pos_point_cloud
            logger.write('Pos PC shape: {}'.format(shape))

            dataset = f['neg_point_cloud']
            shape = [dataset.shape[0] + neg_point_cloud.shape[0], 1024, 3]
            dataset.resize(shape)
            dataset[-neg_point_cloud.shape[0]:] = neg_point_cloud
            logger.write('Neg PC shape: {}'.format(shape))

            dataset = f['pos_keypoints']
            shape = [dataset.shape[0] + pos_keypoints.shape[0], 
                    num_keypoints, 3]
            dataset.resize(shape)
            dataset[-pos_keypoints.shape[0]:] = pos_keypoints
            logger.write('Pos KP shape: {}'.format(shape))

            dataset = f['neg_keypoints']
            shape = [dataset.shape[0] + neg_keypoints.shape[0], 
                    num_keypoints, 3]
            dataset.resize(shape)
            dataset[-neg_keypoints.shape[0]:] = neg_keypoints
            logger.write('Neg KP shape: {}'.format(shape))
    lock.release()

def append_data(data_list, lock, save_path):
    """Appends data to a hdf5 file.

    Args:
        data_list: The list of data to be saved.
        lock: The lock to avoid io conflict.
        save_path: The path to hdf5 file.
        
    Returns:
        None.
    """
    pos_point_cloud = []
    neg_point_cloud = []
    pos_keypoints = []
    neg_keypoints = []
    logger.write('List length: {}'.format(len(data_list)))
    for idata, data_name in enumerate(data_list):
        if idata % 100 == 0:
            logger.write('Appending {}'.format(str(idata).zfill(6)))

        p = np.load(open(os.path.join(
                    point_cloud_dir, data_name), 'rb'))
        k = np.load(open(os.path.join(
                    keypoints_dir, data_name), 'rb'))
        k = np.reshape(k, [-1])

        if k[0] == 3:
            pos_point_cloud.append(np.reshape(p, [1, 1024, 3]))
            pos_keypoints.append(np.reshape(k[1:], [1, num_keypoints, 3]))

        elif k[0] == 1 or k[0] == 2:
            neg_point_cloud.append(np.reshape(p, [1, 1024, 3]))
            neg_keypoints.append(np.reshape(k[1:], [1, num_keypoints, 3]))
        else:
            pass

    save_data(pos_point_cloud,
              neg_point_cloud,
              pos_keypoints,
              neg_keypoints,
              save_path,
              lock)
    return


processes = []
m = Manager()
lock = m.Lock()

pool = Pool(processes=12)

args_append_data = [(data_sublist, lock, save_path) 
        for data_sublist in data_sublists]

pool.starmap(append_data, args_append_data)
