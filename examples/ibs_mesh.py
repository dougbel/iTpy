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

    
    ########################################################################################################################
    # 1. IBS VISUALIZATION
    edges_from, edges_to = util.get_edges( ibs_calculator.vertices, ibs_calculator.ridge_vertices )
    
    visualizer = trimesh.Scene( [ 
                                  trimesh.load_path( np.hstack( ( edges_from, edges_to ) ).reshape(-1, 2, 3) ),
                                  trimesh.points.PointCloud( np.asarray(od3_cloud_obj_poisson.points) , colors=[0,0,255,255] ),
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
                                  trimesh.points.PointCloud( np.asarray(od3_cloud_obj_poisson.points) , colors=[0,0,255,255] ),
                                  tri_mesh_obj,
                                  trimesh.points.PointCloud( ibs_calculator.points, colors=[255,0,0,255] ),
                                  trimesh.load_path( np.hstack(( edges_from, edges_to)).reshape(-1, 2, 3) )
                                  ] )

    # display the environment with callback
    visualizer2.show()

