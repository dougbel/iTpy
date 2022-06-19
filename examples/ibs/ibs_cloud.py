import time
import numpy as np
from os import remove

import trimesh
import open3d as o3d

import it.util as util
from it.training.ibs import IBS

if __name__ == '__main__':
    '''
    Shows the IBS calculated using point clouds. Due to inadequate sampling, IBS pierce in object and 
    environment boundaries. It seems that even with high-density sampling penetrations are presented.     
    '''
    influence_radio_ratio = 2

    ################################################################################################
    # ##   1.  EXECUTION WITH 1,000 POINTS IN OBJECT AND ENVIRONMENT RESPECTIVELY
    ################################################################################################
    od3_cloud_env_poisson = o3d.io.read_point_cloud('./data/ibs/cloud_env_1000_points.pcd')
    od3_cloud_obj_poisson = o3d.io.read_point_cloud('./data/ibs/cloud_obj_1000_points.pcd')

    np_cloud_env_poisson = np.asarray(od3_cloud_env_poisson.points)
    np_cloud_obj_poisson = np.asarray(od3_cloud_obj_poisson.points)

    print("IBS calculation ...")
    start = time.time()  # timing execution
    ibs_calculator = IBS(np_cloud_env_poisson, np_cloud_obj_poisson)
    end = time.time()  # timing execution
    print(end - start, " seconds on IBS calculation (2,000 points)")  # timing execution

    # ### VISUALIZATION

    tri_mesh_obj = trimesh.load_mesh("./data/interactions/table_bowl/bowl.ply")
    radio, np_pivot = util.influence_sphere(tri_mesh_obj, radio_ratio=influence_radio_ratio)

    [idx_extracted, np_ibs_vertices_extracted] = util.extract_cloud_by_sphere(ibs_calculator.vertices, np_pivot, radio)

    edges_from, edges_to = util.get_edges(ibs_calculator.vertices, ibs_calculator.ridge_vertices, idx_extracted)

    tri_cloud_obj = trimesh.points.PointCloud(np.asarray(od3_cloud_obj_poisson.points), colors=[0, 255, 0, 255])
    tri_cloud_env = trimesh.points.PointCloud(np.asarray(od3_cloud_env_poisson.points), colors=[100, 100, 100, 255])

    tri_cloud_ibs_vertices_extracted = trimesh.points.PointCloud(np_ibs_vertices_extracted, colors=[0, 0, 255, 255])
    tri_path_ibs_edges_extracted = trimesh.load_path(np.hstack((edges_from, edges_to)).reshape(-1, 2, 3))

    tri_mesh_ibs = ibs_calculator.get_trimesh()
    # tri_mesh_ibs = tri_mesh_ibs.subdivide()

    sphere_ro, sphere_center = util.influence_sphere(tri_mesh_obj, radio_ratio=influence_radio_ratio)

    tri_mesh_ibs_segmented = util.slide_mesh_by_sphere(tri_mesh_ibs, sphere_center, sphere_ro, 16)

    visualizer = trimesh.Scene([tri_cloud_ibs_vertices_extracted, tri_cloud_obj,
                                tri_cloud_env, tri_path_ibs_edges_extracted])#, tri_mesh_ibs_segmented])

    # display the environment with callback
    visualizer.show(flags={'cull': False, 'wireframe': False, 'axis': False})

    ################################################################################################
    # ##   2.  EXECUTION WITH 10,000 POINTS IN OBJECT AND ENVIRONMENT RESPECTIVELY
    ################################################################################################

    od3_cloud_env_poisson = o3d.io.read_point_cloud('./data/ibs/cloud_env_10000_points.pcd')
    od3_cloud_obj_poisson = o3d.io.read_point_cloud('./data/ibs/cloud_obj_10000_points.pcd')

    np_cloud_env_poisson = np.asarray(od3_cloud_env_poisson.points)
    np_cloud_obj_poisson = np.asarray(od3_cloud_obj_poisson.points)

    print("IBS calculation ...")
    start = time.time()  # timing execution
    ibs_calculator = IBS(np_cloud_env_poisson, np_cloud_obj_poisson)
    end = time.time()  # timing execution
    print(end - start, " seconds on IBS calculation (20,000 points)")  # timing execution

    # ### VISUALIZATION

    tri_mesh_obj = trimesh.load_mesh("./data/interactions/table_bowl/bowl.ply")
    radio, np_pivot = util.influence_sphere(tri_mesh_obj, radio_ratio=influence_radio_ratio)

    [idx_extracted, np_ibs_vertices_extracted] = util.extract_cloud_by_sphere(ibs_calculator.vertices, np_pivot, radio)

    edges_from, edges_to = util.get_edges(ibs_calculator.vertices, ibs_calculator.ridge_vertices, idx_extracted)

    tri_cloud_obj = trimesh.points.PointCloud(np.asarray(od3_cloud_obj_poisson.points), colors=[0, 255, 0, 255])
    tri_cloud_env = trimesh.points.PointCloud(np.asarray(od3_cloud_env_poisson.points), colors=[100, 100, 100, 255])

    tri_cloud_ibs_vertices_extracted = trimesh.points.PointCloud(np_ibs_vertices_extracted, colors=[0, 0, 255, 255])
    tri_path_ibs_edges_extracted = trimesh.load_path(np.hstack((edges_from, edges_to)).reshape(-1, 2, 3))

    tri_mesh_ibs = ibs_calculator.get_trimesh()
    # tri_mesh_ibs = tri_mesh_ibs.subdivide()

    sphere_ro, sphere_center = util.influence_sphere(tri_mesh_obj, radio_ratio=influence_radio_ratio)

    tri_mesh_ibs_segmented = util.slide_mesh_by_sphere(tri_mesh_ibs, sphere_center, sphere_ro, 16)

    visualizer = trimesh.Scene([tri_cloud_ibs_vertices_extracted, tri_cloud_obj,
                                tri_cloud_env, tri_path_ibs_edges_extracted])#, tri_mesh_ibs_segmented])

    # display the environment with callback
    visualizer.show()
