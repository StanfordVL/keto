import h5py
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)


class GraspReader(object):
    """Grasp data reader."""

    def __init__(self, data_path, trainval_ratio=0.9):
        """Initialization.

        Args:
            data_path: Path to the hdf5 file.
            trainval_ratio: The ratio between the training
                and validation data.
        
        Returns:
            None.
        """
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
            np.std(self.pos_p[:256], axis=(0, 1))))
        print('Grasp std: {}'.format(
            np.std(self.pos_g[:256], axis=0)))
        return

    def make_suitable(self, indices):
        """Removes the repeated indices."""
        indices = sorted(set(list(indices)))
        return indices

    def random_rotate(self, p, g):
        """Randomly rotates point cloud and grasps.
        
        Args:
            p: The point cloud tensor.
            g: The grasp.

        Returns:
            p: The rotated point cloud.
            g: The rotated grasp.
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
        """Adds random noise to the point cloud and 
           the grasp to generate negative examples.

        Args:
            p: The point cloud tensor.
            g: The grasp.

        Returns:
            p: The disturbed point cloud.
            g: The disturbed grasp.
        """
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
        """Randomly chooses postive examples for training.
        
        Args:
            size: The number of output examples.
         
        Returns:
            pos_p: The point cloud.
            pos_g: The positive grasps.
        """
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
        """Randomly chooses negative examples for training.
        
        Args:
            size: The number of output examples.
         
        Returns:
            neg_p: The point cloud.
            neg_g: The negative grasps.
        """
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
        """Randomly chooses postive examples for validation.
        
        Args:
            size: The number of output examples.
         
        Returns:
            pos_p: The point cloud.
            pos_g: The positive grasps.
        """
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
        """Randomly chooses negative examples for validation.
        
        Args:
            size: The number of output examples.
         
        Returns:
            neg_p: The point cloud.
            neg_g: The negative grasps.
        """
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
    """Keypoint data reader."""


    def __init__(self, 
                 data_path, 
                 trainval_ratio=0.8, 
                 num_keypoints=2):
        """Initialization.
      
        Args:
            data_path: Path to hdf5 file.
            trainval_ratio: The ratio between the training
                and validation data.
            num_keypoints: 3 if the effect point is considered
                else 2.
        
        Returns:
            None.
        """
        logging.info('Loading {}'.format(data_path))
        f = h5py.File(data_path, 'r')
        self.pos_p = f['pos_point_cloud']
        self.pos_k = f['pos_keypoints']
        self.neg_p = f['neg_point_cloud']
        self.neg_k = f['neg_keypoints']
        self.num_funct_vect = self.pos_k.shape[1] - num_keypoints
        self.num_keypoints = num_keypoints

        self.trainval_ratio = trainval_ratio
        print('Number positive: {}'.format(
            self.pos_p.shape[0]))
        return

    def make_suitable(self, indices):
        """Removes the repeated indices."""
        indices = sorted(set(list(indices)))
        return indices

    def random_rotate(self, p, k):
        """Randomly rotates point cloud and keypoints
           for data augmentation.

        Args:
            p: The point cloud.
            k: The keypoints.

        Returns:
            p: The rotated point cloud.
            k: The rotated keypoints.
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
        nk = self.num_keypoints
        k[:, :nk] = np.matmul(k[:, :nk] - mean_p,
                              mat_drz_t) + mean_p
        k[:, nk:] = np.matmul(k[:, nk:], mat_drz_t)
        return p, k

    def random_disturb(self, p, k, scale_down=0.2):
        """Randomly disturbs keypoints for creating
           new negative examples.

        Args:
            p: The point cloud.
            k: The keypoints.

        Returns:
            p: The rotated point cloud.
            k: The rotated keypoints.
        """
        mean_p = np.mean(p, axis=1, keepdims=True)
        std_p = np.std(p - mean_p, axis=1, keepdims=True)
        noise = np.random.normal(
            size=np.shape(k)) * std_p * scale_down
        k = k + noise
        return p, k

    def sample_pos_train(self, size):
        """Randomly chooses postive examples for training.
        
        Args:
            size: The number of output examples.
         
        Returns:
            pos_p: The point cloud.
            pos_g: The positive keypoints.
        """
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
        """Randomly chooses negative examples for training.
        
        Args:
            size: The number of output examples.
         
        Returns:
            neg_p: The point cloud.
            neg_g: The negative keypoints.
        """
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
        """Randomly chooses postive examples for validation.
        
        Args:
            size: The number of output examples.
         
        Returns:
            pos_p: The point cloud.
            pos_g: The positive keypoints.
        """
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
        """Randomly chooses negative examples for validation.
        
        Args:
            size: The number of output examples.
         
        Returns:
            neg_p: The point cloud.
            neg_g: The negative keypoints.
        """
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
