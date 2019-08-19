import numpy as np

import open3d as o3d
import trimesh

from  it.training.trainer import Trainer
from  it.training.trainer import MeshSamplingMethod as msm
import it.util as util

def visualize ( trainer, tri_mesh_env, tri_mesh_obj ):
    tri_cloud_ibs = trimesh.points.PointCloud( trainer.np_cloud_ibs, color=[255,0,0,150] )
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
    scene.show()


if __name__ == '__main__':

    tri_mesh_ibs = trimesh.load_mesh('./data/pv/ibs_mesh_segmented.ply')
    
    tri_mesh_env = trimesh.load_mesh('./data/table.ply')

    trainer_poisson = Trainer( tri_mesh_ibs, tri_mesh_env, msm.ON_MESH_BY_POISSON )
    
    trainer_vertices = Trainer( tri_mesh_ibs, tri_mesh_env, msm.ON_MESH_VERTICES )



    #VISUALIZATION
    tri_mesh_obj = trimesh.load_mesh('./data/bowl.ply')

    visualize ( trainer_poisson, tri_mesh_env, tri_mesh_obj )

    visualize ( trainer_vertices, tri_mesh_env, tri_mesh_obj )