import numpy as np

import open3d as o3d
import trimesh

from  it.training.trainer import Trainer
from  it.training.trainer import MeshSamplingMethod as msm
import it.util as util


def get_camera(scene):
  np.set_printoptions(suppress=True)
  print(scene.camera_transform)


def visualize ( trainer, tri_mesh_env, tri_mesh_obj ):
    tri_cloud_ibs = trimesh.points.PointCloud( trainer.np_cloud_ibs, color=[255,0,0,100] )
    pv_origin     = trimesh.points.PointCloud( trainer.pv_points, color=[0,0,255,250] )

    tri_mesh_env.visual.face_colors = [100, 100, 100, 100]
    tri_mesh_obj.visual.face_colors = [0, 255, 0, 100]

    pv_3d_path = np.hstack(( trainer.pv_points,trainer.pv_points + trainer.pv_vectors)).reshape(-1, 2, 3)

    provenance_vectors = trimesh.load_path( pv_3d_path )
    
    scene = trimesh.Scene( [ 
                            provenance_vectors,
                            pv_origin,
                            tri_cloud_ibs,
                            tri_mesh_env,
                            tri_mesh_obj
                            ])
    scene.show(callback= get_camera)


if __name__ == '__main__':
    #RIDE A MOTORCYCLE
    tri_mesh_env = trimesh.load_mesh("./data/interactions/motorbike_rider/motorbike.ply")
    tri_mesh_obj = trimesh.load_mesh("./data/interactions/motorbike_rider/biker.ply")
    tri_mesh_ibs_segmented = trimesh.load_mesh("./data/interactions/motorbike_rider/ibs_motorbike_biker_sampled_3000_resamplings_2.ply")

    #PLACE BOWL TABLE
    '''tri_mesh_ibs_segmented = trimesh.load_mesh('./data/pv/ibs_mesh_segmented.ply')
    tri_mesh_env = trimesh.load_mesh('./data/table.ply')
    tri_mesh_obj = trimesh.load_mesh('./data/bowl.ply')'''

    trainer_poisson = Trainer( tri_mesh_ibs_segmented, tri_mesh_env, msm.ON_MESH_BY_POISSON )
    
    trainer_vertices = Trainer( tri_mesh_ibs_segmented, tri_mesh_env, msm.ON_MESH_VERTICES )

    trainer_weighted = Trainer( tri_mesh_ibs_segmented, tri_mesh_env, msm.ON_MESH_WEIGHTED, 10, 5 )



    #VISUALIZATION    

    visualize ( trainer_poisson, tri_mesh_env, tri_mesh_obj )

    visualize ( trainer_vertices, tri_mesh_env, tri_mesh_obj )

    visualize ( trainer_weighted, tri_mesh_env, tri_mesh_obj )