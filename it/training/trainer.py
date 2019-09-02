import math
from enum import Enum
import numpy as np

import trimesh
from transforms3d.derivations.eulerangles import z_rotation

import it.util as util


class MeshSamplingMethod(Enum):
    ON_MESH_VERTICES = 0    #Samplig using vertices 
    ON_MESH_BY_POISSON = 1  #Sampling using mesh based in poisson disk sample "http://www.cemyuksel.com/research/sampleelimination/"
    ON_MESH_WEIGHTED =2     #Using weighted sampling the smaller provenance vector the higher probability of been choosen



class Trainer:
    SAMPLE_SIZE = 512

    def __init__(self, tri_mesh_ibs, tri_mesh_env, sampling_method = MeshSamplingMethod.ON_MESH_BY_POISSON ):

        self.__getIBSCloud( tri_mesh_ibs,tri_mesh_env,  sampling_method )

        self.__get_env_normal( tri_mesh_env )

        self.pv_points = self.np_cloud_ibs[ self.idx_ibs_cloud_sample ]

        self.pv_vectors = self.closest_points[ self.idx_ibs_cloud_sample ] - self.pv_points

        self.pv_norms = self.norms[ self.idx_ibs_cloud_sample ]

        self.pv_max_norm = self.pv_norms.max()
        
        self.pv_min_norm = self.pv_norms.min()
        
        self.pv_mapped_norms = np.asarray( [ self.__map_norm(norm, self.pv_max_norm, self.pv_min_norm ) for norm in self.pv_norms ] )


    def __getIBSCloud(self, tri_mesh_ibs, tri_mesh_env, sampling_method ):
        
        if sampling_method == MeshSamplingMethod.ON_MESH_WEIGHTED:
            self.np_cloud_ibs = util.sample_points_poisson_disk(tri_mesh_ibs, self.SAMPLE_SIZE*20)
            ( self.closest_points, self.norms , __) = tri_mesh_env.nearest.on_surface( self.np_cloud_ibs )
            #TODO tranform random sampling to a random sampling without repetition
            max_norm = self.norms.max()
            min_norm = self.norms.min()
            mapped_norms =  [ self.__map_norm(norm, max_norm, min_norm) for norm in self.norms ]
            sum_mapped_norms = sum(mapped_norms)
            probabilities = [float(mapped)/sum_mapped_norms for mapped in mapped_norms]
            self.idx_ibs_cloud_sample = np.random.choice(self.np_cloud_ibs.shape[0], self.SAMPLE_SIZE, replace =False, p=probabilities)


        else:

            if sampling_method == MeshSamplingMethod.ON_MESH_VERTICES:
                self.np_cloud_ibs = np.asarray( tri_mesh_ibs.vertices )
                ( self.closest_points, self.norms , __) = tri_mesh_env.nearest.on_surface( self.np_cloud_ibs )

            elif sampling_method == MeshSamplingMethod.ON_MESH_BY_POISSON:
                self.np_cloud_ibs = util.sample_points_poisson_disk(tri_mesh_ibs, self.SAMPLE_SIZE*5)
                ( self.closest_points, self.norms , __) = tri_mesh_env.nearest.on_surface( self.np_cloud_ibs )
            
            self.idx_ibs_cloud_sample = np.random.randint( 0, self.np_cloud_ibs.shape[0], self.SAMPLE_SIZE )
       


    def __get_env_normal( self, tri_mesh_env ):

        ( _, __ , triangle_id) = tri_mesh_env.nearest.on_surface( np.array([0,0,0]).reshape(-1,3) )
        self.env_normal = tri_mesh_env.face_normals[triangle_id]

        

    def __map_norm(self, norm, max, min):
        return ( norm - min) * (0- 1) / (max - min) + 1
        #return ( value_in - min_original_range) * (max_mapped- min_mapped) / (max_original_range - min_original_range) + min_mapped;

