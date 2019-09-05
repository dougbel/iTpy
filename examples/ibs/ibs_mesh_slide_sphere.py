import time
import numpy as np
from os import remove

import trimesh

import it.util as util
from  it.training.ibs import IBSMesh


if __name__ == '__main__':

    tri_mesh_obj = trimesh.load_mesh("./data/interactions/table_bowl/bowl.ply")
    
    obj_min_bound = np.asarray( tri_mesh_obj.vertices ).min(axis=0)
    obj_max_bound = np.asarray( tri_mesh_obj.vertices ).max(axis=0)
    
    tri_mesh_env = trimesh.load_mesh('./data/interactions/table_bowl/table.ply')


    extension = np.linalg.norm(obj_max_bound-obj_min_bound)
    middle_point = (obj_max_bound+obj_min_bound)/2
    
    tri_mesh_env_segmented = util.slide_mesh_by_bounding_box( tri_mesh_env, middle_point, extension )  

    start = time.time() ## timing execution
    ibs_calculator = IBSMesh( tri_mesh_env_segmented, tri_mesh_obj, 400, 4 )
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
    
    tri_mesh_ibs_segmented = util.slide_mesh_by_sphere( tri_mesh_ibs, sphere_center, sphere_ro, 16 ) 
    
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
    tri_mesh_ibs.visual.face_colors = [0, 0, 150, 100]

    visualizer3 = trimesh.Scene([ 
                                tri_mesh_env, 
                                tri_mesh_obj,
                                tri_mesh_ibs_segmented
                                ] )

    # display the environment with callback
    visualizer3.show()