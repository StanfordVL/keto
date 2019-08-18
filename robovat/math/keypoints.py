import numpy as np
import random
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn import linear_model


def hammer_keypoints_heuristic(point_cloud,
                               n_clusters=12):
    p = np.squeeze(point_cloud)
    center = np.mean(p, axis=0)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=0).fit(p)
    centers = kmeans.cluster_centers_
    hull = ConvexHull(centers[:, :2])
    hull = centers[hull.vertices]
    success = False
    while not success:
        dist = 0.0
        while dist == 0:
            grasping_point = random.choice(centers)
            dist_from_hull = np.linalg.norm(
                grasping_point - hull, axis=1)
            dist = np.amin(dist_from_hull)
        vect_center_grasp = grasping_point - center
        vect_center_hull = hull - center
        cosine = vect_center_grasp * vect_center_hull
        mask = np.sum(cosine, axis=1) < 0
        success = True if np.sum(mask) > 0 else False
    func_point = random.choice(hull[mask])
    grasping_point = np.expand_dims(
        grasping_point, 0).astype(np.float32)
    func_point = np.expand_dims(
        func_point, 0).astype(np.float32)
    
    kmeans = KMeans(
        n_clusters=32,
        random_state=0).fit(p)
    centers_dense = kmeans.cluster_centers_
    hull = ConvexHull(centers_dense[:, :2])
    hull = centers_dense[hull.vertices]
    hull_index = np.argsort(
            np.linalg.norm(func_point - hull, axis=1))[0]
    func_point = hull[np.newaxis, hull_index]

    return grasping_point, func_point


def push_keypoints_heuristic(point_cloud,
                             n_clusters=12):
    p = np.squeeze(point_cloud)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=0).fit(p)
    centers = kmeans.cluster_centers_
    hull = ConvexHull(centers[:, :2])
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
    hull = ConvexHull(centers[:, :2])
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


def search_keypoints(point_cloud,
                     n_clusters=8,
                     n_collision_points=6):
    p = np.squeeze(point_cloud)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=0).fit(p)
    centers = kmeans.cluster_centers_
    p_center = np.mean(p,
                       axis=0,
                       keepdims=True)
    dist = np.linalg.norm(
        centers - p_center, axis=1)
    indices = np.argsort(dist)[::-1]
    collision_points = centers[
        indices[:n_collision_points]]
    farest_center = np.expand_dims(
        centers[indices[0]], axis=0)
    dist_farest = np.linalg.norm(
        centers - farest_center, axis=1)
    grasping_point = centers[
        np.argsort(dist_farest)[1]]
    grasping_point = np.expand_dims(
        grasping_point, axis=0)
    handle_axis = farest_center - grasping_point
    handle_axis = handle_axis / np.linalg.norm(
        handle_axis, axis=1, keepdims=True)
    func_candidates = centers[
        np.argsort(dist_farest)[1:]]
    vect = farest_center - func_candidates
    vect = vect / np.linalg.norm(
        vect, axis=1, keepdims=True)
    func_index = np.argsort(
        np.abs(np.sum(
            handle_axis * vect, axis=1)))[0]
    func_point = np.expand_dims(
        func_candidates[func_index], 0)

    kmeans = KMeans(
        n_clusters=32,
        random_state=0).fit(p)
    centers_dense = kmeans.cluster_centers_
    hull = ConvexHull(centers_dense[:, :2])
    hull = centers_dense[hull.vertices]
    hull_index = np.argsort(
            np.linalg.norm(func_point - hull, axis=1))[0]
    func_point = hull[np.newaxis, hull_index]

    keypoints = [grasping_point,
                 func_point,
                 collision_points]

    return keypoints
