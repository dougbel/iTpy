import time
import numpy as np
from os import remove

import trimesh
from open3d import open3d as o3d

import it.util as util
from  it.training.ibs import IBS


if __name__ == '__main__':

    ################################################################################################
    ###   1.  EXECUTION WITH 1,000 POINTS IN OBJECT AND ENVIRONMENT RESPECTIVELY
    ################################################################################################
    od3_cloud_env_poisson = o3d.io.read_point_cloud('./data/ibs/cloud_env_1000_points.pcd')
    od3_cloud_obj_poisson = o3d.io.read_point_cloud('./data/ibs/cloud_obj_1000_points.pcd')

    np_cloud_env_poisson = np.asarray(od3_cloud_env_poisson.points)
    np_cloud_obj_poisson = np.asarray(od3_cloud_obj_poisson.points)


    print ("IBS calculation ..." )  
    start = time.time() ## timing execution
    ibs_calculator = IBS( np_cloud_env_poisson, np_cloud_obj_poisson )
    end = time.time() ## timing execution
    print (end - start , " seconds on IBS calculation (2,000 points)" )  ## timing execution


    ####VISUALIZATION
    mesh_obj_o3d = o3d.io.read_triangle_mesh("./data/bowl.ply")
    obj_min_bound = mesh_obj_o3d.get_min_bound()
    obj_max_bound = mesh_obj_o3d.get_max_bound()
    
    radio = np.linalg.norm( obj_max_bound - obj_min_bound )
    np_pivot = np.asarray( obj_max_bound + obj_min_bound ) / 2

    [ idx_extracted, np_ibs_vertices_extracted ]= util.extract_by_distance( ibs_calculator.vertices, np_pivot, radio )

    edges_from, edges_to = util.get_edges( ibs_calculator.vertices, ibs_calculator.ridge_vertices, idx_extracted )

    
    visualizer = trimesh.Scene( [ trimesh.points.PointCloud( np_ibs_vertices_extracted, colors=[0,255,0,255] ), 
                                  trimesh.points.PointCloud( np.asarray(od3_cloud_obj_poisson.points) , colors=[0,0,255,255] ),
                                  trimesh.points.PointCloud( np.asarray(od3_cloud_env_poisson.points), colors=[255,0,0,255] ),
                                  trimesh.load_path( np.hstack( ( edges_from, edges_to) ).reshape(-1, 2, 3))
                                  ] )

    # display the environment with callback
    visualizer.show()



    ################################################################################################
    ###   2.  EXECUTION WITH 10,000 POINTS IN OBJECT AND ENVIRONMENT RESPECTIVELY
    ################################################################################################

    od3_cloud_env_poisson = o3d.io.read_point_cloud('./data/ibs/cloud_env_10000_points.pcd')
    od3_cloud_obj_poisson = o3d.io.read_point_cloud('./data/ibs/cloud_obj_10000_points.pcd')

    np_cloud_env_poisson = np.asarray(od3_cloud_env_poisson.points)
    np_cloud_obj_poisson = np.asarray(od3_cloud_obj_poisson.points)


    print ("IBS calculation ..." )  
    start = time.time() ## timing execution
    ibs_calculator = IBS( np_cloud_env_poisson, np_cloud_obj_poisson )
    end = time.time() ## timing execution
    print (end - start , " seconds on IBS calculation (20,000 points)" )  ## timing execution
    
    ####VISUALIZATION
    mesh_obj_o3d = o3d.io.read_triangle_mesh("./data/bowl.ply")
    obj_min_bound = mesh_obj_o3d.get_min_bound()
    obj_max_bound = mesh_obj_o3d.get_max_bound()
    
    radio = np.linalg.norm( obj_max_bound - obj_min_bound )
    np_pivot = np.asarray( obj_max_bound + obj_min_bound ) / 2
    
    [ idx_extracted, np_ibs_vertices_extracted ]= util.extract_by_distance( ibs_calculator.vertices, np_pivot, radio )
    
    edges_from, edges_to = util.get_edges( ibs_calculator.vertices, ibs_calculator.ridge_vertices, idx_extracted )

    
    visualizer = trimesh.Scene( [ trimesh.points.PointCloud( np_ibs_vertices_extracted, colors=[0,255,0,255] ), 
                                  trimesh.points.PointCloud( np.asarray(od3_cloud_obj_poisson.points) , colors=[0,0,255,255] ),
                                  trimesh.points.PointCloud( np.asarray(od3_cloud_env_poisson.points), colors=[255,0,0,255] ),
                                  trimesh.load_path( np.hstack( ( edges_from, edges_to) ).reshape(-1, 2, 3))
                                  ] )

    # display the environment with callback
    visualizer.show()

