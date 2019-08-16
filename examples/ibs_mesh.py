import time
import numpy as np
from os import remove

import trimesh

import it.util as util
from  it.training.ibs import IBSMesh


if __name__ == '__main__':

    tri_mesh_obj = trimesh.load_mesh("./data/bowl.ply")
    
    obj_min_bound = np.asarray( tri_mesh_obj.vertices ).min(axis=0)
    obj_max_bound = np.asarray( tri_mesh_obj.vertices ).max(axis=0)

    tri_mesh_env = trimesh.load_mesh('./data/table.ply')

    tri_mesh_env_segmented = util.slide_mesh_by_bounding_box(tri_mesh_env, obj_min_bound, obj_max_bound)  

    np_cloud_env_poisson = util.sample_points_poisson_disk( tri_mesh_env_segmented, 400 )
    np_cloud_obj_poisson = util.sample_points_poisson_disk( tri_mesh_obj, 400 )


    start = time.time() ## timing execution
    ibs_calculator = IBSMesh( np_cloud_env_poisson,tri_mesh_env,  np_cloud_obj_poisson, tri_mesh_obj )  
    end = time.time() ## timing execution
    print (end - start , " seconds on IBS calculation (400 original points)" )  ## timing execution

    
    ########################################################################################################################
    # 1. IBS VISUALIZATION
    edges_from, edges_to = util.get_edges( ibs_calculator.vertices, ibs_calculator.ridge_vertices )
    
    visualizer = trimesh.Scene( [ 
                                  trimesh.load_path( np.hstack( ( edges_from, edges_to ) ).reshape(-1, 2, 3) ),
                                  trimesh.points.PointCloud( np_cloud_obj_poisson , colors=[0,0,255,255] ),
                                  tri_mesh_obj,
                                ] )

    visualizer.show()


    ########################################################################################################################
    # 2. CROPPED VISUALIZATION MESH AND POINT CLOUD (IBS)

    #extracting point no farther than the principal sphere
    radio = np.linalg.norm( obj_max_bound - obj_min_bound )
    np_pivot = np.asarray( obj_max_bound + obj_min_bound ) / 2
    [idx_extracted, np_ibs_vertices_extracted ]= util.extract_by_distance( ibs_calculator.vertices, np_pivot, radio )

    #cutting edges in the polygon mesh
    edges_from, edges_to = util.get_edges( ibs_calculator.vertices, ibs_calculator.ridge_vertices, idx_extracted )

    
    visualizer2 = trimesh.Scene( [ #trimesh.points.PointCloud( ibs_calculator.cloud_ibs, colors=[255,255,0,255] ), 
                                  trimesh.points.PointCloud( np_ibs_vertices_extracted, colors=[0,255,0,255] ), 
                                  trimesh.points.PointCloud( np_cloud_obj_poisson , colors=[0,0,255,255] ),
                                  tri_mesh_obj,
                                  trimesh.points.PointCloud( ibs_calculator.points, colors=[255,0,0,255] ),
                                  trimesh.load_path( np.hstack(( edges_from, edges_to)).reshape(-1, 2, 3) )
                                  ] )

    # display the environment with callback
    visualizer2.show()

