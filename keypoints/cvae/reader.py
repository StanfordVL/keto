import h5py
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)


class GraspReader(object):

    def __init__(self, data_path, trainval_ratio=0.8):
        logging.info('Loading {}'.format(data_path))
        f = h5py.File(data_path, 'r')
        self.pos_p = f['pos_point_cloud']
        self.pos_g = f['pos_grasp']
        self.neg_p = f['neg_point_cloud']
        self.neg_g = f['neg_grasp']
        self.trainval_ratio = trainval_ratio
        print('Number positive: {}, negative: {}'.format(
            self.pos_p.shape[0], self.neg_p.shape[0]))
        print('Point cloud std: {}'.format(
            np.std(self.pos_p, axis=(0, 1))))
        print('Grasp std: {}'.format(
            np.std(self.pos_g, axis=0)))
        return

    def make_suitable(self, indices):
        indices = sorted(set(list(indices)))
        return indices

    def random_rotate(self, p, g):
        """Randomly rotate point cloud and grasps
        """
        num = p.shape[0]
        drz = np.random.uniform(
            0, np.pi * 2, size=(num, 1))
        g_xyz, g_rx, g_ry, g_rz = \
            np.split(g, [3, 4, 5], axis=1)
        zeros = np.zeros_like(drz)
        ones = np.ones_like(drz)
        mat_drz = np.concatenate(
            [np.cos(drz), -np.sin(drz), zeros,
             np.sin(drz), np.cos(drz), zeros,
             zeros, zeros, ones],
            axis=1)
        mat_drz = np.reshape(mat_drz, [num, 3, 3])
        mat_drz_t = np.transpose(mat_drz, [0, 2, 1])
        p = np.matmul(p - g_xyz[:, np.newaxis],
                      mat_drz_t) + g_xyz[:, np.newaxis]
        g_rz = g_rz + drz
        g = np.concatenate(
            [g_xyz, g_rx, g_ry, g_rz], axis=1)
        return p, g

    def random_disturb(self, p, g):
        p_mean = np.mean(p, axis=1)
        g_xyz, g_rx, g_ry, g_rz = \
            np.split(g, [3, 4, 5], axis=1)
        g_xyz = g_xyz + np.random.normal(
            size=np.shape(g_xyz)) * np.std(
            g_xyz - p_mean,
            axis=0, keepdims=True)
        g_rx = g_rx + np.random.normal(
            size=np.shape(g_rx))
        g_ry = g_ry + np.random.normal(
            size=np.shape(g_ry))
        g_rz = g_rz + np.random.uniform(
            low=0, high=np.pi * 2,
            size=np.shape(g_rz))
        g = np.concatenate(
            [g_xyz, g_rx, g_ry, g_rz], axis=1)
        return p, g

    def sample_pos_train(self, size):
        indices = np.random.randint(
            low=0,
            high=int(self.pos_p.shape[0]
                     * self.trainval_ratio),
            size=size)
        indices = self.make_suitable(indices)
        pos_p = np.array(self.pos_p[indices, :, :],
                         dtype=np.float32)
        pos_g = np.array(self.pos_g[indices, :],
                         dtype=np.float32)
        return pos_p, pos_g

    def sample_neg_train(self, size):
        indices = np.random.randint(
            low=0,
            high=int(self.neg_p.shape[0]
                     * self.trainval_ratio),
            size=size)
        indices = self.make_suitable(indices)
        neg_p = np.array(self.neg_p[indices],
                         dtype=np.float32)
        neg_g = np.array(self.neg_g[indices],
                         dtype=np.float32)
        if np.random.uniform() > 0.8:
            neg_p, neg_g = self.random_disturb(
                neg_p, neg_g)
        return neg_p, neg_g

    def sample_pos_val(self, size):
        indices = np.random.randint(
            high=self.pos_p.shape[0],
            low=int(self.pos_p.shape[0]
                    * self.trainval_ratio),
            size=size)
        indices = self.make_suitable(indices)
        pos_p = np.array(self.pos_p[indices],
                         dtype=np.float32)
        pos_g = np.array(self.pos_g[indices],
                         dtype=np.float32)
        return pos_p, pos_g

    def sample_neg_val(self, size):
        indices = np.random.randint(
            high=self.neg_p.shape[0],
            low=int(self.neg_p.shape[0]
                    * self.trainval_ratio),
            size=size)
        indices = self.make_suitable(indices)
        neg_p = np.array(self.neg_p[indices],
                         dtype=np.float32)
        neg_g = np.array(self.neg_g[indices],
                         dtype=np.float32)

        if np.random.uniform() > 0.8:
            neg_p, neg_g = self.random_disturb(
                neg_p, neg_g)
        return neg_p, neg_g


class KeypointReader(object):

    def __init__(self, data_path, trainval_ratio=0.8):
        logging.info('Loading {}'.format(data_path))
        f = h5py.File(data_path, 'r')
        self.pos_p = f['pos_point_cloud']
        self.pos_k = f['pos_keypoints']
        self.neg_p = f['neg_point_cloud']
        self.neg_k = f['neg_keypoints']
        self.trainval_ratio = trainval_ratio
        print('Number positive: {}'.format(
            self.pos_p.shape[0]))
        return

    def make_suitable(self, indices):
        indices = sorted(set(list(indices)))
        return indices

    def random_rotate(self, p, k):
        """Randomly rotate point cloud and keypoints
           for data augmentation
        """
        num = p.shape[0]
        drz = np.random.uniform(
            0, np.pi * 2, size=(num, 1))
        zeros = np.zeros_like(drz)
        ones = np.ones_like(drz)
        mat_drz = np.concatenate(
            [np.cos(drz), -np.sin(drz), zeros,
             np.sin(drz), np.cos(drz), zeros,
             zeros, zeros, ones],
            axis=1)
        mat_drz = np.reshape(mat_drz, [num, 3, 3])
        mat_drz_t = np.transpose(mat_drz, [0, 2, 1])
        mean_p = np.mean(p, axis=1, keepdims=True)
        p = np.matmul(p - mean_p,
                      mat_drz_t) + mean_p
        k = np.matmul(k - mean_p,
                      mat_drz_t) + mean_p
        return p, k

    def random_disturb(self, p, k, scale_down=0.2):
        """Randomly disturb keypoints for creating
           new negative examples
        """
        mean_p = np.mean(p, axis=1, keepdims=True)
        std_p = np.std(p - mean_p, axis=1, keepdims=True)
        noise = np.random.normal(
            size=np.shape(k)) * std_p * scale_down
        k = k + noise
        return p, k

    def sample_pos_train(self, size):
        indices = np.random.randint(
            low=0,
            high=int(self.pos_p.shape[0]
                     * self.trainval_ratio),
            size=size)
        indices = self.make_suitable(indices)
        pos_p = np.array(self.pos_p[indices],
                         dtype=np.float32)
        pos_k = np.array(self.pos_k[indices],
                         dtype=np.float32)
        return pos_p, pos_k

    def sample_neg_train(self, size):
        indices = np.random.randint(
            low=0,
            high=int(self.neg_p.shape[0]
                     * self.trainval_ratio),
            size=size)
        indices = self.make_suitable(indices)
        neg_p = np.array(self.neg_p[indices],
                         dtype=np.float32)
        neg_k = np.array(self.neg_k[indices],
                         dtype=np.float32)
        if np.random.uniform() > 0.8:
            neg_p, neg_k = self.random_disturb(
                neg_p, neg_k)
        return neg_p, neg_k

    def sample_pos_val(self, size):
        indices = np.random.randint(
            high=self.pos_p.shape[0],
            low=int(self.pos_p.shape[0]
                    * self.trainval_ratio),
            size=size)
        indices = self.make_suitable(indices)
        pos_p = np.array(self.pos_p[indices],
                         dtype=np.float32)
        pos_k = np.array(self.pos_k[indices],
                         dtype=np.float32)
        return pos_p, pos_k

    def sample_neg_val(self, size):
        indices = np.random.randint(
            high=self.neg_p.shape[0],
            low=int(self.neg_p.shape[0]
                    * self.trainval_ratio),
            size=size)
        indices = self.make_suitable(indices)
        neg_p = np.array(self.neg_p[indices],
                         dtype=np.float32)
        neg_k = np.array(self.neg_k[indices],
                         dtype=np.float32)

        if np.random.uniform() > 0.8:
            neg_p, neg_k = self.random_disturb(
                neg_p, neg_k)
        return neg_p, neg_k