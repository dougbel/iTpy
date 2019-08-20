import numpy as np

import trimesh

import it.util as util
from it.training.ibs import IBSMesh
from it.training.trainer import Trainer
from it.training.trainer import MeshSamplingMethod as msm
from it.training.agglomerator import Agglomerator


if __name__ == '__main__':

    obj_name = "bowl"
    obj_mesh_file = "./data/bowl.ply"
    env_name = "table"
    env_mesh_file = "./data/table.ply"
    affordance_name = "Place"


                                  
    tri_mesh_obj = trimesh.load_mesh( obj_mesh_file)
    
    obj_min_bound = np.asarray( tri_mesh_obj.vertices ).min(axis=0)
    obj_max_bound = np.asarray( tri_mesh_obj.vertices ).max(axis=0)
    
    tri_mesh_env = trimesh.load_mesh( env_mesh_file )


    extension = np.linalg.norm(obj_max_bound-obj_min_bound)
    middle_point = (obj_max_bound+obj_min_bound)/2
    
    tri_mesh_env_segmented = util.slide_mesh_by_bounding_box(tri_mesh_env, middle_point, extension)  

    
    ibs_calculator = IBSMesh( tri_mesh_env_segmented,  tri_mesh_obj, 400, 4 )
    
    ################################
    #GENERATING AND SEGMENTING IBS MESH
    ################################
    
    tri_mesh_ibs = ibs_calculator.get_trimesh()
    tri_mesh_ibs = tri_mesh_ibs.subdivide()
    
    sphere_ro = np.linalg.norm( obj_max_bound - obj_min_bound )
    sphere_center = np.asarray( obj_max_bound + obj_min_bound ) / 2
    
    tri_mesh_ibs_segmented = util.slide_mesh_by_sphere( tri_mesh_ibs, sphere_center, sphere_ro ) 

    
    trainer = Trainer( tri_mesh_ibs = tri_mesh_ibs_segmented, 
                        tri_mesh_env = tri_mesh_env, 
                        sampling_method = msm.ON_MESH_BY_POISSON )

    agg = Agglomerator(trainer)

    agg.save_agglomerated_iT( affordance_name, env_name, obj_name )


    #VISUALIZATION
    provenance_vectors = trimesh.load_path( np.hstack(( trainer.pv_points, trainer.pv_points + trainer.pv_vectors )).reshape(-1, 2, 3) )
    
    tri_mesh_obj.visual.face_colors = [0, 255, 0, 200]
    tri_mesh_ibs_segmented.visual.face_colors = [0, 0, 255, 100]
    tri_mesh_env.visual.face_colors = [200, 200, 200, 150]

    scene = trimesh.Scene( [ 
                            tri_mesh_obj,
                            tri_mesh_env,
                            tri_mesh_ibs_segmented,
                            provenance_vectors
                            ])
    scene.show()