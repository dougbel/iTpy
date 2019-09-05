import math
from enum import Enum
from  collections import Counter
from abc import ABC, abstractmethod
import numpy as np

import trimesh

import it.util as util


class MeshSamplingMethod(Enum):
    ON_MESH_VERTICES = 0    #Samplig using vertices 
    ON_MESH_BY_POISSON = 1  #Sampling using mesh based in poisson disk sample "http://www.cemyuksel.com/research/sampleelimination/"
    ON_MESH_WEIGHTED =2     #Using weighted sampling the smaller provenance vector the higher probability of been choosen



class Sampler(ABC):
    SAMPLE_SIZE = 512

    def __init__(self, tri_mesh_ibs, tri_mesh_env ):
        super().__init__()

        self.tri_mesh_ibs = tri_mesh_ibs
        self.tri_mesh_env = tri_mesh_env

        self.getSample()

        self.pv_max_norm = self.pv_norms.max()
        self.pv_min_norm = self.pv_norms.min()
        
        self.pv_mapped_norms = np.asarray( [ self.map_norm(norm, self.pv_max_norm, self.pv_min_norm ) for norm in self.pv_norms ] )

        #TODO order by mapped norm

    def map_norm(self, norm, max, min):
        return ( norm - min) * (0- 1) / (max - min) + 1
        #return ( value_in - min_original_range) * (max_mapped- min_mapped) / (max_original_range - min_original_range) + min_mapped;
    
    @abstractmethod
    def getSample(self):
        pass




class PoissonDiscRandomSampler(Sampler):

    def getSample(self):
        self.np_cloud_ibs = util.sample_points_poisson_disk(self.tri_mesh_ibs, self.SAMPLE_SIZE*5)
        
        self.idx_ibs_cloud_sample = np.random.randint( 0, self.np_cloud_ibs.shape[0], self.SAMPLE_SIZE )
        
        self.pv_points = self.np_cloud_ibs[ self.idx_ibs_cloud_sample ]
        ( closest_points_in_env, self.pv_norms , __) = self.tri_mesh_env.nearest.on_surface( self.pv_points )
        self.pv_vectors = closest_points_in_env - self.pv_points

class OnVerticesRandomSampler(Sampler):

    def getSample(self):
        self.np_cloud_ibs = np.asarray( self.tri_mesh_ibs.vertices )

        self.idx_ibs_cloud_sample = np.random.randint( 0, self.np_cloud_ibs.shape[0], self.SAMPLE_SIZE )

        self.pv_points = self.np_cloud_ibs[ self.idx_ibs_cloud_sample ]
        ( closest_points_in_env, self.pv_norms , __) = self.tri_mesh_env.nearest.on_surface( self.pv_points )
        self.pv_vectors = closest_points_in_env - self.pv_points


class PoissonDiscWeightedSampler(Sampler):

    def __init__(self, tri_mesh_ibs, tri_mesh_env,  rate_ibs_samples=25, rate_generated_random_numbers=20  ):
        self.rate_ibs_samples = rate_ibs_samples
        self.rate_generated_random_numbers = rate_generated_random_numbers
        super().__init__(tri_mesh_ibs, tri_mesh_env)

    def getSample(self):
        n_ibs_samples = self.SAMPLE_SIZE*self.rate_ibs_samples
        self.np_cloud_ibs = util.sample_points_poisson_disk(self.tri_mesh_ibs, n_ibs_samples)
        lclosest = []
        lnorms = []
        for iterator in range(self.rate_ibs_samples):
            idx_from = self.SAMPLE_SIZE*iterator
            idx_to = idx_from + self.SAMPLE_SIZE
            ( closest_points, norms , __) = self.tri_mesh_env.nearest.on_surface( self.np_cloud_ibs[ idx_from : idx_to ] )
            lclosest.extend( closest_points )
            lnorms.extend( norms )
        closest_points = np.asarray( lclosest )
        norms = np.asarray( lnorms )

        max_norm = norms.max()
        min_norm = norms.min()
        mapped_norms =  [ self.map_norm(norm, max_norm, min_norm) for norm in norms ]
        sum_mapped_norms = sum(mapped_norms)
        probabilities = [float(mapped)/sum_mapped_norms for mapped in mapped_norms]
        self.chosen_ibs_points = np.random.choice(self.np_cloud_ibs.shape[0], n_ibs_samples*self.rate_generated_random_numbers, p=probabilities)
        votes = Counter(self.chosen_ibs_points).most_common(self.SAMPLE_SIZE)
        
        self.idx_ibs_cloud_sample = [tuple[0] for tuple in votes]
        self.pv_points = self.np_cloud_ibs[ self.idx_ibs_cloud_sample ]
        self.pv_vectors = closest_points[ self.idx_ibs_cloud_sample ] - self.pv_points
        self.pv_norms = norms[ self.idx_ibs_cloud_sample ]
