import numpy as np
from scipy.spatial import Voronoi

import trimesh

import it.util as util



class IBS:

    def __init__( self, np_cloud_env, np_cloud_obj ):

        self.size_cloud_env = np_cloud_env.shape[0]
        size_cloud_obj = np_cloud_obj.shape[0]

        self.points = np.empty( ( self.size_cloud_env + size_cloud_obj, 3 ), np.float64 )
        self.points[:self.size_cloud_env] = np_cloud_env
        self.points[self.size_cloud_env:] = np_cloud_obj

        voro = Voronoi( self.points )

        env_idx_vertices = []
        obj_idx_vertices = []

        #check the region formed around point in the environment
        #point_region : Index of the Voronoi region for each input point
        for env_idx_region in voro.point_region[ :self.size_cloud_env ] :
            #voronoi region of environment point
            #regions: Indices of the Voronoi vertices forming each Voronoi region
            env_idx_vertices +=  voro.regions[ env_idx_region ]

        for obj_idx_region in voro.point_region[ self.size_cloud_env: ] :
            #voronoi region of object point
            obj_idx_vertices += voro.regions[ obj_idx_region ] 

        env_idx_vertices = list( set( env_idx_vertices ) )
        obj_idx_vertices = list( set( obj_idx_vertices ) )

        idx_ibs_vertices = [vertex for vertex in env_idx_vertices  if vertex in obj_idx_vertices]

        #avoid index "-1" for vertices extraction
        valid_index = [idx for idx in idx_ibs_vertices if  idx != -1]
        self.vertices = voro.vertices[ valid_index ]

        #generate ridge vertices lists
        self.ridge_vertices = []        #Indices of the Voronoi vertices forming each Voronoi ridge
        self.ridge_points = []          #Indices of the points between which each Voronoi ridge lie
        for i in range( len(voro.ridge_vertices) ):
            ridge = voro.ridge_vertices[i]
            ridge_points = voro.ridge_points[i] 
            #only process ridges in which all vertices are defined in ridges defined by Voronoi 
            if all(idx_vertice in idx_ibs_vertices for idx_vertice in ridge):
                mapped_idx_ridge = [ ( idx_ibs_vertices.index(idx_vertice) if idx_vertice != -1 else -1 ) for idx_vertice  in ridge]
                self.ridge_vertices.append( mapped_idx_ridge)
                self.ridge_points.append(ridge_points)


    def get_trimesh(self):
        trifaces = []
        for ridge in self.ridge_vertices:
            if  -1 in ridge:
                continue
            for pos in range( len( ridge )-2 ):
                trifaces.append( [ridge[-1], ridge[pos], ridge[pos+1] ] )

        mesh = trimesh.Trimesh( vertices = self.vertices, faces = trifaces )
        mesh.fix_normals()

        return mesh

class IBSMesh( IBS ):

    def __init__(self, tri_mesh_env, tri_mesh_obj, size_sampling = 400, resamplings = 4, improve_by_collission = True):

        np_cloud_env_poisson = util.sample_points_poisson_disk( tri_mesh_env, size_sampling )

        np_cloud_obj_poisson = util.sample_points_poisson_disk( tri_mesh_obj, size_sampling )

        np_cloud_obj = self.__project_points_in_sampled_mesh( tri_mesh_obj, np_cloud_obj_poisson, np_cloud_env_poisson )

        np_cloud_env = self.__project_points_in_sampled_mesh( tri_mesh_env, np_cloud_env_poisson, np_cloud_obj )

        for i in range(1,resamplings):
            
            np_cloud_obj = self.__project_points_in_sampled_mesh( tri_mesh_obj, np_cloud_obj, np_cloud_env )

            np_cloud_env = self.__project_points_in_sampled_mesh( tri_mesh_env, np_cloud_env, np_cloud_obj )
            
            
        if improve_by_collission:

            self.__improve_sampling_by_collision_test(tri_mesh_env, tri_mesh_obj, np_cloud_env, np_cloud_obj)

        else:
            
            super( IBSMesh, self ).__init__( np_cloud_env, np_cloud_obj )
                  

    def __improve_sampling_by_collision_test(self, tri_mesh_env, tri_mesh_obj, np_cloud_env, np_cloud_obj):
        
        collision_tester = trimesh.collision.CollisionManager()
        collision_tester.add_object('env',tri_mesh_env)
        collision_tester.add_object('obj',tri_mesh_obj)
        in_collision = True

        while in_collision :

            super( IBSMesh, self ).__init__( np_cloud_env, np_cloud_obj )

            tri_mesh_ibs = self.get_trimesh()

            in_collision, data = collision_tester.in_collision_single(tri_mesh_ibs, return_data=True)

            if not in_collision:
                break
            
            print(" ------------------ ")
            print("puntos de contacto: ", len(data) )

            contact_points_obj = []
            contact_points_env = []

            for i in range(len(data)):
                if "env" in data[i].names:
                    contact_points_env.append( data[i].point ) 
                if "obj" in data[i].names:
                    contact_points_obj.append( data[i].point ) 

            if( len( contact_points_env ) > 0 ):
                np_contact_points_env = np.unique( np.asarray ( contact_points_env ), axis=0)
                np_cloud_env = np.concatenate((np_cloud_env, np_contact_points_env ))

            if( len( contact_points_obj ) > 0 ):
                np_contact_points_obj = np.unique( np.asarray ( contact_points_obj ), axis=0)
                np_cloud_obj = np.concatenate((np_cloud_obj, np_contact_points_obj ))
            
            if( len( contact_points_env ) > 0 ):
                np_cloud_obj = self.__project_points_in_sampled_mesh( tri_mesh_obj, np_cloud_obj, np_contact_points_env )

            if( len( contact_points_obj ) > 0 ):
                np_cloud_env = self.__project_points_in_sampled_mesh( tri_mesh_env, np_cloud_env, np_contact_points_obj )
            
                  


    def __project_points_in_sampled_mesh(self, tri_mesh_sampled, np_sample, np_to_project):
        if np_to_project.shape[0] == 0:
            return np_sample

        ( nearest_points, __ , __) = tri_mesh_sampled.nearest.on_surface( np_to_project )
        
        np_new_sample = np.empty( (len(np_sample) + len(nearest_points) ,3) )

        np_new_sample[:len(np_sample)] = np_sample
        np_new_sample[len(np_sample):] = nearest_points
        np_new_sample = np.unique(np_new_sample ,axis=0 )
    
        return np_new_sample