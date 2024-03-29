import numpy as np
import random
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn import linear_model


def make_noisy(x, std=0.001):
    return x + np.random.normal(size=np.shape(x)) * std


def hammer_keypoints_heuristic(point_cloud,
                               n_clusters=12):
    p = np.squeeze(point_cloud)
    center = np.mean(p, axis=0)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=0).fit(p)
    centers = kmeans.cluster_centers_

    xs, ys, _ = np.split(centers, [1, 2], axis=1)
    ransac = linear_model.RANSACRegressor()
    ransac.fit(xs, np.squeeze(ys))

    centers_inlier = centers[ransac.inlier_mask_]
    grasp_point = random.choice(centers_inlier)
    
    centers_head = centers[
            np.logical_not(ransac.inlier_mask_)]

    if centers_head.shape[0] == 0:
        func_point = random.choice(centers)
    elif centers_head.shape[0] < 3:
        func_point = random.choice(centers_head)
    else:
        hull = ConvexHull(make_noisy(centers_head[:, :2]))
        hull = centers_head[hull.vertices]
        func_point = random.choice(hull)

    grasp_point = np.expand_dims(
        grasp_point, 0).astype(np.float32)
    func_point = np.expand_dims(
        func_point, 0).astype(np.float32)
    
    vect_grasp_func = np.squeeze(grasp_point - func_point)
    k = vect_grasp_func[1] / (vect_grasp_func[0] + 1e-6)
    
    func_vect = np.reshape(
        np.array([-k/(1 + k**2)**0.5,
                  1/(1 + k**2)**0.5, 0.0]), [1, 3])
    vect_center_func = func_point - center
    if vect_center_func.dot(func_vect.T) < 0:
        func_vect = -func_vect

    grasp_point = grasp_point.astype(np.float32)
    func_point = func_point.astype(np.float32)
    func_vect = func_vect.astype(np.float32)

    return grasp_point, func_point, func_vect


def push_keypoints_heuristic(point_cloud,
                             n_clusters=12):
    p = np.squeeze(point_cloud)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=0).fit(p)
    centers = kmeans.cluster_centers_
    hull = ConvexHull(make_noisy(centers[:, :2]))
    hull = centers[hull.vertices]

    func_point_init = random.choice(centers)
    dist = np.linalg.norm(centers - func_point_init, axis=1)
    argsort = np.argsort(dist)[:n_clusters//2]
    centers_near = centers[argsort]
    ransac = linear_model.RANSACRegressor()
    x, y, _ = np.split(centers_near - func_point_init, [1, 2], axis=1)
    ransac.fit(x, y)
    centers_inlier = centers_near[ransac.inlier_mask_]
    
    mean_inlier = np.mean(centers_inlier, axis=0, keepdims=True)
    dist = np.linalg.norm(centers - mean_inlier, axis=1)
    func_point = np.reshape(centers[np.argmin(dist)], [1, 3])
    
    grasp_point = random.choice(centers).reshape([1, 3])

    invalid = np.amin(np.linalg.norm(
              grasp_point - hull, axis=1)) == 0
    while invalid:
        grasp_point = random.choice(centers).reshape([1, 3])
        invalid = np.amin(np.linalg.norm(
                  grasp_point - hull, axis=1)) == 0

    k = np.squeeze(ransac.estimator_.coef_)
    
    func_vect = np.reshape(
        np.array([-k/(1 + k**2)**0.5,
                  1/(1 + k**2)**0.5, 0.0]), [1, 3])

    same_side = (func_vect[0, 0] * k - func_vect[0, 1]) * (
        (grasp_point[0, 0] - func_point[0, 0]) * k - 
        (grasp_point[0, 1] - func_point[0, 1])) > 0
    func_vect = -func_vect if same_side else func_vect
    
    grasp_point = grasp_point.astype(np.float32)
    func_point = func_point.astype(np.float32)
    func_vect = func_vect.astype(np.float32)
    
    return grasp_point, func_point, func_vect


def reach_keypoints_heuristic(point_cloud,
                              n_clusters=12):
    p = np.squeeze(point_cloud)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=0).fit(p)
    centers = kmeans.cluster_centers_
    hull = ConvexHull(make_noisy(centers[:, :2]))
    hull = centers[hull.vertices]
    grasp_point = random.choice(centers).reshape([1, 3])
    invalid = np.amin(np.linalg.norm(
              grasp_point - hull, axis=1)) == 0
    while invalid:
        grasp_point = random.choice(centers).reshape([1, 3])
        invalid = np.amin(np.linalg.norm(
                  grasp_point - hull, axis=1)) == 0
    
    xs, ys, zs = np.split(centers, [1, 2], axis=1)
    ransac = linear_model.RANSACRegressor()
    ransac.fit(xs, np.squeeze(ys))
    centers_inlier = centers[ransac.inlier_mask_]
    if np.random.uniform() > 0.75:
        inlier_mask = np.logical_not(ransac.inlier_mask_)
        if np.sum(inlier_mask) > 1:
            ransac = linear_model.RANSACRegressor()
            ransac.fit(xs[inlier_mask], np.squeeze(ys[inlier_mask]))
            centers_inlier = centers[inlier_mask]
            centers_inlier = centers_inlier[ransac.inlier_mask_]
        
    k = np.squeeze(ransac.estimator_.coef_)
    func_vect = np.reshape(
        [1/np.sqrt(1+np.square(k)),
         k/np.sqrt(1+np.square(k)), 0], [1, 3])
    proj = np.sum(func_vect * (
        centers_inlier - np.mean(centers_inlier, 
                                 axis=0, keepdims=True)), axis=1)
    indices = [np.argmax(proj), np.argmin(proj)]
    sides = centers_inlier[indices]
    dist = [None, None]
    for iside in [0, 1]:
        dist[iside] = np.linalg.norm(sides[iside] - centers, axis=1)
        dist[iside] = np.sum(np.square(
            dist[iside][np.argsort(dist[iside])[n_clusters//2]]))
    sides = sides if dist[0] > dist[1] else sides[[1, 0]]
    func_point = sides[0]
    approx_vect = sides[0] - sides[1]
    func_vect = func_vect if func_vect.dot(
        approx_vect) > 0 else -func_vect
    
    grasp_point = grasp_point.reshape([1, 3]).astype(np.float32)
    func_point = func_point.reshape([1, 3]).astype(np.float32)
    func_vect = func_vect.reshape([1, 3]).astype(np.float32)
    
    return grasp_point, func_point, func_vect


def pull_keypoints_heuristic(point_cloud,
                             n_clusters=12):
    p = np.squeeze(point_cloud)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=0).fit(p)
    centers = kmeans.cluster_centers_

    xs, ys, _ = np.split(centers, [1, 2], axis=1)
    ransac = linear_model.RANSACRegressor()
    ransac.fit(xs, np.squeeze(ys))

    centers_inlier = centers[ransac.inlier_mask_]
    grasp_point = random.choice(centers_inlier)
    
    centers_head = centers[
            np.logical_not(ransac.inlier_mask_)]

    if centers_head.shape[0] == 0:
        func_point = random.choice(centers)
    else:
        func_point = random.choice(centers_head)

    grasp_point = np.expand_dims(
        grasp_point, 0).astype(np.float32)
    func_point = np.expand_dims(
        func_point, 0).astype(np.float32)

    func_vect = np.squeeze(grasp_point - func_point)
    func_vect[2] = 0
    func_vect = func_vect / (1e-6 + np.linalg.norm(func_vect))
    s = np.random.choice([1, -1])
    rot_mat = np.array([[1/np.sqrt(2), -s/np.sqrt(2), 0],
                        [s/np.sqrt(2), 1/np.sqrt(2), 0],
                        [0, 0, 1]])
    func_vect = np.reshape(func_vect.dot(rot_mat), [1, 3])

    grasp_point = grasp_point.astype(np.float32)
    func_point = func_point.astype(np.float32)
    func_vect = func_vect.astype(np.float32)
    
    return grasp_point, func_point, func_vect


def combine_keypoints_heuristic(point_cloud,
                                n_clusters=16):
    p = np.squeeze(point_cloud)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=0).fit(p)
    centers = kmeans.cluster_centers_
    c = np.mean(centers, axis=0, keepdims=True)

    xs, ys, zs = np.split(centers, [1, 2], axis=1)
    ransac = linear_model.RANSACRegressor()
    ransac.fit(xs, np.squeeze(ys))

    centers_inlier = centers[ransac.inlier_mask_]
    grasp_point = np.mean(centers_inlier, axis=0, keepdims=True)

    func_point_push = np.mean(
            centers[np.logical_not(ransac.inlier_mask_)],
            axis=0, keepdims=True)

    k = np.squeeze(ransac.estimator_.coef_)
    func_vect_push = np.reshape(
        [1/np.sqrt(1+np.square(k)),
         k/np.sqrt(1+np.square(k)), 0], [1, 3])

    v_cf = func_point_push - c
    if np.dot(
            np.squeeze(v_cf), 
            np.squeeze(func_vect_push)) < 0:
        func_vect_push = -func_vect_push

    func_vect_reach = -func_vect_push
    func_vect_hammer = func_vect_push.dot(
            np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))

    vect_grasp_to_centers = (centers - grasp_point) * np.array([1, 1, 0])
    product_reach = np.sum(
            func_vect_reach * vect_grasp_to_centers, axis=1)
    func_point_reach = centers[np.argmax(product_reach)]

    product_hammer = np.sum(
            func_vect_hammer * vect_grasp_to_centers, axis=1)
    func_point_hammer = centers[np.argmax(product_hammer)]

    grasp_point = np.reshape(grasp_point, [1, 3]).astype(np.float32)
    func_point = np.concatenate(
            [np.reshape(func_point_push, [1, 3]),
             np.reshape(func_point_reach, [1, 3]),
             np.reshape(func_point_hammer, [1, 3])], 
            axis=0).astype(np.float32)

    func_vect = np.concatenate(
            [np.reshape(func_vect_push, [1, 3]),
             np.reshape(func_vect_reach, [1, 3]),
             np.reshape(func_vect_hammer, [1, 3])], 
            axis=0).astype(np.float32)

    return grasp_point, func_point, func_vect

