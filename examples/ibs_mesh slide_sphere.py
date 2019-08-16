import time
import numpy as np
from os import remove

import trimesh
from open3d import open3d as o3d

import it.util as util
from  it.training.ibs import IBSMesh


if __name__ == '__main__':

    tri_mesh_obj = trimesh.load_mesh("./data/bowl.ply")
    od3_mesh_obj = o3d.io.read_triangle_mesh("./data/bowl.ply")
    od3_cloud_obj_poisson = o3d.geometry.sample_points_poisson_disk( od3_mesh_obj, 400 )

    obj_min_bound = od3_mesh_obj.get_min_bound()
    obj_max_bound = od3_mesh_obj.get_max_bound()
    
    tri_mesh_env = trimesh.load_mesh('./data/table.ply')
    tri_mesh_env_segmented = util.slide_mesh_by_bounding_box(tri_mesh_env, obj_min_bound, obj_max_bound)  

    #it was not possible create a TriangleMesh on the fly, then save an open a ply file
    trimesh.exchange.export.export_mesh(tri_mesh_env_segmented,"segmented_environment.ply", "ply")
    od3_mesh_env_segmented = o3d.io.read_triangle_mesh('segmented_environment.ply')
    remove("segmented_environment.ply")
    od3_cloud_env_poisson = o3d.geometry.sample_points_poisson_disk( od3_mesh_env_segmented, 400 )

    np_cloud_env_poisson = np.asarray(od3_cloud_env_poisson.points)
    np_cloud_obj_poisson = np.asarray(od3_cloud_obj_poisson.points)


    start = time.time() ## timing execution
    ibs_calculator = IBSMesh( np_cloud_env_poisson,tri_mesh_env,  np_cloud_obj_poisson, tri_mesh_obj )
    end = time.time() ## timing execution
    print (end - start , " seconds on IBS calculation (400 original points)" )  ## timing execution
    
    ################################
    #GENERATING AND SEGMENTING IBS MESH
    ################################
    start = time.time() ## timing execution
    
    tri_mesh_ibs = ibs_calculator.get_trimesh()
    tri_mesh_ibs = tri_mesh_ibs.subdivide()
    
    sphere_ro = np.linalg.norm( obj_max_bound - obj_min_bound )
    sphere_center = np.asarray( obj_max_bound + obj_min_bound ) / 2
    
    tri_mesh_ibs_segmented = util.slide_mesh_by_sphere( tri_mesh_ibs, sphere_center, sphere_ro ) 
    
    end = time.time() ## timing execution
    print (end - start , " seconds on IBS MESH GENERATION AND SEGMENTATION" )  ## timing execution

    
    #tri_mesh_ibs_segmented.export("ibs_mesh_segmented.ply","ply")

    print( "is convex:" + str(tri_mesh_ibs.is_convex))
    print( "is empty:" + str(tri_mesh_ibs.is_empty))
    print( "is watertight:" + str(tri_mesh_ibs.is_watertight))
    print( "is widing consistent:" + str(tri_mesh_ibs.is_winding_consistent))

    
    ################################
    # VISUALIZATION
    ################################
    
    tri_mesh_obj.visual.face_colors = [0, 255, 0, 100]
    tri_mesh_env.visual.face_colors = [100, 100, 100, 100]
    tri_mesh_ibs_segmented.visual.face_colors = [0, 0, 150, 100]
    tri_mesh_ibs.visual.face_colors = [255, 0, 0, 50]

    visualizer3 = trimesh.Scene([ 
                                tri_mesh_env, 
                                tri_mesh_obj,
                                tri_mesh_ibs_segmented
                                ] )

    # display the environment with callback
    visualizer3.show()