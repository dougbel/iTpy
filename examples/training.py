import numpy as np

import trimesh

import it.util as util
from it.training.ibs import IBSMesh
from it.training.trainer import Trainer
from it.training.agglomerator import Agglomerator


if __name__ == '__main__':
                                  
    tri_mesh_obj = trimesh.load_mesh("./data/bowl.ply")
    
    obj_min_bound = np.asarray( tri_mesh_obj.vertices ).min(axis=0)
    obj_max_bound = np.asarray( tri_mesh_obj.vertices ).max(axis=0)
    
    tri_mesh_env = trimesh.load_mesh('./data/table.ply')

    tri_mesh_env_segmented = util.slide_mesh_by_bounding_box(tri_mesh_env, obj_min_bound, obj_max_bound)  

    np_cloud_env_poisson = util.sample_points_poisson_disk( tri_mesh_env_segmented, 400 )
    np_cloud_obj_poisson = util.sample_points_poisson_disk( tri_mesh_obj, 400 )


    ibs_calculator = IBSMesh( np_cloud_env_poisson,tri_mesh_env,  np_cloud_obj_poisson, tri_mesh_obj )
    
    ################################
    #GENERATING AND SEGMENTING IBS MESH
    ################################
    
    tri_mesh_ibs = ibs_calculator.get_trimesh()
    tri_mesh_ibs = tri_mesh_ibs.subdivide()
    
    sphere_ro = np.linalg.norm( obj_max_bound - obj_min_bound )
    sphere_center = np.asarray( obj_max_bound + obj_min_bound ) / 2
    
    tri_mesh_ibs_segmented = util.slide_mesh_by_sphere( tri_mesh_ibs, sphere_center, sphere_ro ) 

    np_cloud_ibs = util.sample_points_poisson_disk(tri_mesh_ibs, Trainer.SAMPLE_SIZE*3)
    
    trainer = Trainer( np_cloud_ibs, tri_mesh_env )

    agg = Agglomerator(trainer)

    agg.save_agglomerated_iT( "Place", "table", "bowl" )