import sys
import math
from enum import Enum
from  collections import Counter
from abc import ABC, abstractmethod
import numpy as np

import trimesh

import it.util as util

#TODO  stablish default rates

class Sampler(ABC):
    
    SAMPLE_SIZE = 512

    np_cloud_ibs = np.array([])

    idx_ibs_cloud_sample = []


    pv_points = np.array([])
    pv_vectors = np.array([])
    pv_norms = np.array([])
    pv_max_norm = sys.float_info.min
    pv_min_norm = sys.float_info.max
    pv_mapped_norms = np.array([])

    def __init__(self, tri_mesh_ibs, tri_mesh_env ):
        super().__init__()

        self.tri_mesh_ibs = tri_mesh_ibs
        self.tri_mesh_env = tri_mesh_env

        self.get_clouds_to_sample()

        self.get_sample()

        self.set_pv_min_max_mapped_norms()

        #TODO order by mapped norm


    def set_pv_min_max_mapped_norms(self):
        
        self.pv_max_norm = self.pv_norms.max()
        self.pv_min_norm = self.pv_norms.min()
        
        self.pv_mapped_norms = np.asarray( [ self.map_norm(norm, self.pv_max_norm, self.pv_min_norm ) for norm in self.pv_norms ] )


    def map_norm(self, norm, max, min):
        return ( norm - min) * (0- 1) / (max - min) + 1
        #return ( value_in - min_original_range) * (max_mapped- min_mapped) / (max_original_range - min_original_range) + min_mapped;
    
    @abstractmethod
    def get_sample(self):
        pass

    @abstractmethod
    def get_clouds_to_sample(self):
        pass


class WeightedSampler(Sampler, ABC):
    BATCH_SIZE_FOR_CLSST_POINT = 1000
    
    np_cloud_env = np.array([])
    norms = np.array([])


    def __init__(self, tri_mesh_ibs, tri_mesh_env, rate_generated_random_numbers=500  ):
        self.rate_generated_random_numbers = rate_generated_random_numbers
        super().__init__(tri_mesh_ibs, tri_mesh_env)


    def get_sample(self):
        
        max_norm = self.norms.max()
        min_norm = self.norms.min()
        mapped_norms =  [ self.map_norm(norm, max_norm, min_norm) for norm in self.norms ]
        sum_mapped_norms = sum(mapped_norms)
        probabilities = [float(mapped)/sum_mapped_norms for mapped in mapped_norms]

        n_rolls = self.np_cloud_ibs.shape[0]*self.rate_generated_random_numbers
        
        result_rolls = np.random.choice( self.np_cloud_ibs.shape[0], n_rolls, p=probabilities )
        votes = Counter(result_rolls).most_common(self.SAMPLE_SIZE)
        
        self.idx_ibs_cloud_sample = [tuple[0] for tuple in votes]
        self.pv_points = self.np_cloud_ibs[ self.idx_ibs_cloud_sample ]
        self.pv_vectors = self.np_cloud_env[ self.idx_ibs_cloud_sample ] - self.pv_points
        self.pv_norms = self.norms[ self.idx_ibs_cloud_sample ]

    def choosig_with_other_rate(self, rate_generated_random_numbers):
        self.rate_generated_random_numbers = rate_generated_random_numbers
        self.get_sample()
        self.set_pv_min_max_mapped_norms()


class PoissonDiscRandomSampler(Sampler):
    rate_ibs_samples = 5 
    def __init__(self, tri_mesh_ibs, tri_mesh_env,  rate_ibs_samples=5 ):
        self.rate_ibs_samples = rate_ibs_samples
        super().__init__(tri_mesh_ibs, tri_mesh_env)
    
    def get_clouds_to_sample(self):
        self.np_cloud_ibs = util.sample_points_poisson_disk(self.tri_mesh_ibs, self.SAMPLE_SIZE*self.rate_ibs_samples)

    def get_sample(self):

        self.idx_ibs_cloud_sample = np.random.randint( 0, self.np_cloud_ibs.shape[0], self.SAMPLE_SIZE )
        self.pv_points = self.np_cloud_ibs[ self.idx_ibs_cloud_sample ]
        ( closest_points_in_env, norms , __) = self.tri_mesh_env.nearest.on_surface( self.pv_points )
        self.pv_vectors = closest_points_in_env - self.pv_points
        self.pv_norms = norms



class PoissonDiscWeightedSampler(WeightedSampler):

    def __init__(self, tri_mesh_ibs, tri_mesh_env,  rate_ibs_samples=25, rate_generated_random_numbers=500  ):
        self.rate_ibs_samples = rate_ibs_samples
        super().__init__(tri_mesh_ibs, tri_mesh_env, rate_generated_random_numbers)


    def get_clouds_to_sample(self):
        n_ibs_samples = self.SAMPLE_SIZE*self.rate_ibs_samples
        self.np_cloud_ibs = util.sample_points_poisson_disk( self.tri_mesh_ibs, n_ibs_samples)

        iterations = math.ceil( n_ibs_samples/ self.BATCH_SIZE_FOR_CLSST_POINT )

        lclosest = []
        lnorms = []
        for it in range(iterations):
            idx_from = it * self.BATCH_SIZE_FOR_CLSST_POINT
            idx_to = idx_from + self.BATCH_SIZE_FOR_CLSST_POINT
            ( closest_points, norms , __) = self.tri_mesh_env.nearest.on_surface( self.np_cloud_ibs[ idx_from : idx_to ] )
            lclosest.extend( closest_points )
            lnorms.extend( norms )
            
        self.np_cloud_env = np.asarray( lclosest )
        self.norms = np.asarray( lnorms )

        bad_idxs = np.argwhere(np.isnan(lnorms))

        self.np_cloud_env = np.delete(self.np_cloud_env, bad_idxs, 0)
        self.np_cloud_ibs = np.delete(self.np_cloud_ibs, bad_idxs, 0)
        self.norms = np.delete(self.norms, bad_idxs, 0 )

    


class OnVerticesRandomSampler(Sampler):

    def get_clouds_to_sample(self):
        self.np_cloud_ibs = np.asarray( self.tri_mesh_ibs.vertices )

    def get_sample(self):

        self.idx_ibs_cloud_sample = np.random.randint( 0, self.np_cloud_ibs.shape[0], self.SAMPLE_SIZE )

        self.pv_points = self.np_cloud_ibs[ self.idx_ibs_cloud_sample ]
        ( closest_points_in_env, norms , __) = self.tri_mesh_env.nearest.on_surface( self.pv_points )
        self.pv_vectors = closest_points_in_env - self.pv_points
        self.pv_norms = norms


class OnVerticesWeightedSampler(WeightedSampler):

    def get_clouds_to_sample(self):
        self.np_cloud_ibs = np.asarray( self.tri_mesh_ibs.vertices )
        size_input_cloud = self.np_cloud_ibs.shape[0]
        iterations = math.ceil( size_input_cloud/ self.BATCH_SIZE_FOR_CLSST_POINT )

        lclosest = []
        lnorms = []

        for it in range(iterations):
            idx_from = it * self.BATCH_SIZE_FOR_CLSST_POINT
            idx_to = idx_from + self.BATCH_SIZE_FOR_CLSST_POINT
            ( closest_points, norms , __) = self.tri_mesh_env.nearest.on_surface( self.np_cloud_ibs[ idx_from : idx_to ] )
            lclosest += list(closest_points)
            lnorms +=  list(norms)

        self.np_cloud_env = np.asarray( lclosest )
        self.norms = np.asarray( lnorms )

        bad_idxs = np.argwhere(np.isnan(lnorms))

        self.np_cloud_env = np.delete(self.np_cloud_env, bad_idxs, 0)
        self.np_cloud_ibs = np.delete(self.np_cloud_ibs, bad_idxs, 0)
        self.norms = np.delete(self.norms, bad_idxs, 0 )



class OnGivenPointCloudRandomSampler(Sampler):

    np_input_cloud = np.array([])

    BATCH_SIZE_FOR_CLSST_POINT = 1000

    def __init__(self, tri_mesh_ibs, tri_mesh_env, np_input_cloud):
        self.np_input_cloud = np_input_cloud
        super().__init__( tri_mesh_ibs, tri_mesh_env )

    def get_clouds_to_sample(self):

        #With environment points, find the nearest point in the IBS surfaces
        size_input_cloud = self.np_input_cloud.shape[0]
        iterations = math.ceil( size_input_cloud/ self.BATCH_SIZE_FOR_CLSST_POINT )
        lclosest = []
        lnorms = []
        for it in range(iterations):
            idx_from = it * self.BATCH_SIZE_FOR_CLSST_POINT
            idx_to = idx_from + self.BATCH_SIZE_FOR_CLSST_POINT
            ( closest_points, norms , __) = self.tri_mesh_ibs.nearest.on_surface( self.np_input_cloud[ idx_from : idx_to ] )
            lclosest += list(closest_points)
            lnorms +=  list(norms)

        self.np_cloud_ibs = np.asarray( lclosest )
        bad_idxs = np.argwhere(np.isnan(lnorms))
        self.np_cloud_ibs = np.delete(self.np_cloud_ibs, bad_idxs, 0)

        # Calculate all PROVENANCE VECTOR RELATED TO IBS SURFACE SAMPLES
        size_cloud_ibs =  self.np_cloud_ibs.shape[0]
        iterations = math.ceil( size_cloud_ibs/ self.BATCH_SIZE_FOR_CLSST_POINT )
        lclosest = []
        lnorms = []
        for it in range(iterations):
            idx_from = it * self.BATCH_SIZE_FOR_CLSST_POINT
            idx_to = idx_from + self.BATCH_SIZE_FOR_CLSST_POINT
            ( closest_points, norms , __) = self.tri_mesh_env.nearest.on_surface( self.np_cloud_ibs[ idx_from : idx_to ] )
            lclosest += list(closest_points)
            lnorms +=  list(norms)

        self.np_cloud_env = np.asarray( lclosest )
        self.norms = np.asarray( lnorms )
        bad_idxs = np.argwhere(np.isnan(lnorms))
        self.np_cloud_ibs = np.delete(self.np_cloud_ibs, bad_idxs, 0)
        self.np_cloud_env = np.delete(self.np_cloud_env, bad_idxs, 0)
        self.norms = np.delete(self.norms, bad_idxs, 0 )

    def get_sample(self):

        self.idx_ibs_cloud_sample = np.random.randint( 0, self.np_cloud_ibs.shape[0], self.SAMPLE_SIZE )

        self.pv_points = self.np_cloud_ibs[ self.idx_ibs_cloud_sample ]
        ( closest_points_in_env, norms , __) = self.tri_mesh_env.nearest.on_surface( self.pv_points )
        self.pv_vectors = closest_points_in_env - self.pv_points
        self.pv_norms = norms


class OnGivenPointCloudWeightedSampler(WeightedSampler):

    def __init__(self, tri_mesh_ibs, tri_mesh_env, np_input_cloud, rate_generated_random_numbers=500  ):
        self.np_input_cloud = np_input_cloud
        super().__init__(tri_mesh_ibs, tri_mesh_env, rate_generated_random_numbers)


    def get_clouds_to_sample(self):
        size_input_cloud = self.np_input_cloud.shape[0]
        iterations = math.ceil( size_input_cloud/ self.BATCH_SIZE_FOR_CLSST_POINT )

        lclosest = []
        lnorms = []

        for it in range(iterations):
            idx_from = it * self.BATCH_SIZE_FOR_CLSST_POINT
            idx_to = idx_from + self.BATCH_SIZE_FOR_CLSST_POINT
            ( closest_points, norms , __) = self.tri_mesh_ibs.nearest.on_surface( self.np_input_cloud[ idx_from : idx_to ] )
            lclosest += list(closest_points)
            lnorms +=  list(norms)

        self.np_cloud_ibs = np.asarray( lclosest )
        bad_idxs = np.argwhere(np.isnan(lnorms))
        self.np_cloud_ibs = np.delete(self.np_cloud_ibs, bad_idxs, 0)

        # Calculate all PROVENANCE VECTOR RELATED TO IBS SURFACE SAMPLES
        size_cloud_ibs =  self.np_cloud_ibs.shape[0]
        iterations = math.ceil( size_cloud_ibs/ self.BATCH_SIZE_FOR_CLSST_POINT )
        lclosest = []
        lnorms = []
        for it in range(iterations):
            idx_from = it * self.BATCH_SIZE_FOR_CLSST_POINT
            idx_to = idx_from + self.BATCH_SIZE_FOR_CLSST_POINT
            ( closest_points, norms , __) = self.tri_mesh_env.nearest.on_surface( self.np_cloud_ibs[ idx_from : idx_to ] )
            lclosest += list(closest_points)
            lnorms +=  list(norms)

        self.np_cloud_env = np.asarray( lclosest )
        self.norms = np.asarray( lnorms )
        bad_idxs = np.argwhere(np.isnan(lnorms))
        self.np_cloud_ibs = np.delete(self.np_cloud_ibs, bad_idxs, 0)
        self.np_cloud_env = np.delete(self.np_cloud_env, bad_idxs, 0)
        self.norms = np.delete(self.norms, bad_idxs, 0 )