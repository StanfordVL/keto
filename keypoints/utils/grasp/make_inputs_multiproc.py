import os
import h5py
import numpy as np
from multiprocessing import Manager, Pool

point_cloud_dir = './point_cloud'
grasp_4dof_dir = './grasp_4dof'
save_path = './data.hdf5'

scale = 20

batch_size = 3000

data_list = os.listdir(point_cloud_dir)

data_sublists = [data_list[x:x + batch_size]
                 for x in range(0, len(data_list), batch_size)]


class Logger(object):

    def __init__(self, output='./output.log'):
        self.output = output
        return

    def write(self, message):
        with open(self.output, 'a') as f:
            f.write(message + '\r\n')
        return

logger = Logger()


def save_data(pos_point_cloud,
              neg_point_cloud,
              pos_grasp,
              neg_grasp,
              save_path,
              scale=20):

    scale_grasp = np.reshape([scale, scale, scale, 1, 1, 1], (1, 6))

    pos_point_cloud = np.concatenate(pos_point_cloud, axis=0) * scale
    pos_grasp = np.concatenate(pos_grasp, axis=0) * scale_grasp

    neg_point_cloud = np.concatenate(neg_point_cloud, axis=0) * scale
    neg_grasp = np.concatenate(neg_grasp, axis=0) * scale_grasp

    if not os.path.exists(save_path):
        logger.write('Creating new file')
        with h5py.File(save_path, 'w') as f:
            for converted_data in [pos_point_cloud,
                                   pos_grasp, neg_point_cloud, neg_grasp]:
                logger.write('shape: {}'.format(converted_data.shape))

            f.create_dataset('pos_point_cloud', data=pos_point_cloud,
                             maxshape=(None, 1024, 3))
            f.create_dataset('neg_point_cloud', data=neg_point_cloud,
                             maxshape=(None, 1024, 3))
            f.create_dataset('pos_grasp', data=pos_grasp,
                             maxshape=(None, 6))
            f.create_dataset('neg_grasp', data=neg_grasp,
                             maxshape=(None, 6))
            f.close()

    else:
        print('Appending to existing file')
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

            dataset = f['pos_grasp']
            shape = [dataset.shape[0] + pos_grasp.shape[0], 6]
            dataset.resize(shape)
            dataset[-pos_grasp.shape[0]:] = pos_grasp
            logger.write('Pos GP shape: {}'.format(shape))

            dataset = f['neg_grasp']
            shape = [dataset.shape[0] + neg_grasp.shape[0], 6]
            dataset.resize(shape)
            dataset[-neg_grasp.shape[0]:] = neg_grasp
            logger.write('Neg GP shape: {}'.format(shape))


def append_data(data_list, lock, save_path='./data.hdf5'):
    pos_point_cloud = []
    neg_point_cloud = []
    pos_grasp = []
    neg_grasp = []
    logger.write('List length: {}'.format(len(data_list)))
    for idata, data_name in enumerate(data_list):
        if idata % 100 == 0:
            logger.write('Appending {}'.format(str(idata).zfill(6)))
        with open(os.path.join(
                point_cloud_dir, data_name), 'rb') as f:
            point_cloud = np.load(f)

        with open(os.path.join(
                grasp_4dof_dir, data_name), 'rb') as f:
            grasp_4dof = np.load(f)

        grasp = grasp_4dof

        lock.acquire()
        if grasp[0] > 0.002:
            pos_point_cloud.append(
                point_cloud[np.newaxis])
            pos_grasp.append(grasp[np.newaxis, 1:])

        else:
            neg_point_cloud.append(
                point_cloud[np.newaxis])
            neg_grasp.append(grasp[np.newaxis, 1:])
        lock.release()

    lock.acquire()
    save_data(pos_point_cloud,
              neg_point_cloud,
              pos_grasp,
              neg_grasp,
              save_path)
    lock.release()
    return


processes = []
m = Manager()
lock = m.Lock()

pool = Pool(processes=12)

args_append_data = [(data_sublist, lock) for data_sublist in data_sublists]

pool.starmap(append_data, args_append_data)
