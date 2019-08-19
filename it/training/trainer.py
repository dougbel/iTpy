import trimesh
import numpy as np
from open3d import open3d as o3d
from transforms3d.derivations.eulerangles import z_rotation
import math

class Trainer:

    SAMPLE_SIZE = 512

    def __init__(self, np_cloud_ibs, tri_mesh_env):

        ( closest_points, norms , __) = tri_mesh_env.nearest.on_surface( np_cloud_ibs )
       
        idx_sample = np.random.randint( 0, np_cloud_ibs.shape[0], self.SAMPLE_SIZE )

        self.__get_env_normal( tri_mesh_env )

        self.pv_points = np_cloud_ibs[ idx_sample ]

        self.pv_vectors = closest_points[ idx_sample ] - self.pv_points

        self.pv_norms = norms[ idx_sample ]

        self.pv_max_norm = self.pv_norms.max()
        
        self.pv_min_norm = self.pv_norms.min()
        
        self.pv_mapped_norms = np.asarray( [ self.__map_norm(norm) for norm in self.pv_norms ] )

    

    def __get_env_normal( self, tri_mesh_env ):

        ( _, __ , triangle_id) = tri_mesh_env.nearest.on_surface( np.array([0,0,0]).reshape(-1,3) )
        self.env_normal = tri_mesh_env.face_normals[triangle_id]

        

    def __map_norm(self, norm):
        return ( norm - self.pv_min_norm) * (0- 1) / (self.pv_max_norm - self.pv_min_norm) + 1



#TODO tranform random sampling to a random sampling without repetition