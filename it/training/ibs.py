import trimesh
import numpy as np
from open3d import open3d as o3d
from scipy.spatial import Voronoi
from os import remove


class IBS:

    def __init__( self, np_cloud_env, np_cloud_obj ):

        size_cloud_env = np_cloud_env.shape[0]
        size_cloud_obj = np_cloud_obj.shape[0]

        self.points = np.empty( ( size_cloud_env + size_cloud_obj, 3 ), np.float64 )
        self.points[:size_cloud_env] = np_cloud_env
        self.points[size_cloud_env:] = np_cloud_obj

        voro = Voronoi( self.points )

        env_idx_vertices = []
        obj_idx_vertices = []

        #check the region formed around point in the environment
        #point_region : Index of the Voronoi region for each input point
        for env_idx_region in voro.point_region[ :size_cloud_env ] :
            #voronoi region of environment point
            #regions: Indices of the Voronoi vertices forming each Voronoi region
            env_idx_vertices +=  voro.regions[ env_idx_region ]

        for obj_idx_region in voro.point_region[ size_cloud_env: ] :
            #voronoi region of object point
            obj_idx_vertices += voro.regions[ obj_idx_region ] 

        env_idx_vertices = list( set( env_idx_vertices ) )
        obj_idx_vertices = list( set( obj_idx_vertices ) )

        idx_ibs_vertices = [vertex for vertex in env_idx_vertices  if vertex in obj_idx_vertices]

        #avoid index "-1" for vertices extraction
        valid_index = [idx for idx in idx_ibs_vertices if  idx != -1]
        self.vertices = voro.vertices[ valid_index ]

        #generate ridge vertices lists
        self.ridge_vertices = []
        for ridge in voro.ridge_vertices:
            #only process ridges in which all vertices are defined in ridges defined by Voronoi 
            if all(idx_vertice in idx_ibs_vertices for idx_vertice in ridge):
                mapped_idx_ridge = [ ( idx_ibs_vertices.index(idx_vertice) if idx_vertice != -1 else -1 ) for idx_vertice  in ridge]
                self.ridge_vertices.append( mapped_idx_ridge)


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

    def __init__(self, cloud_env, tri_mesh_env, cloud_obj, tri_mesh_obj):
        
        #####
        ( obj_points_in_env, __ , __) = tri_mesh_env.nearest.on_surface( cloud_obj )
        
        np_cloud_env = np.empty( (len(cloud_env) + len(obj_points_in_env) ,3) )

        np_cloud_env[:len(cloud_env)] = cloud_env
        np_cloud_env[len(cloud_env):] = obj_points_in_env

        #####
        ( env_points_in_obj, __ , __) = tri_mesh_obj.nearest.on_surface( np_cloud_env )

        np_cloud_obj = np.empty( (len(cloud_obj) + len(env_points_in_obj) ,3) )

        np_cloud_obj[:len(cloud_obj)] = cloud_obj
        np_cloud_obj[len(cloud_obj):] = env_points_in_obj
        np_cloud_obj = np.unique(np_cloud_obj ,axis=0 )

        #####
        ( obj_points_in_env, __ , __) = tri_mesh_env.nearest.on_surface( np_cloud_obj )
        
        np_cloud_env_final = np.empty( (len(np_cloud_env) + len(obj_points_in_env) ,3) )

        np_cloud_env_final[:len(np_cloud_env)] = np_cloud_env
        np_cloud_env_final[len(np_cloud_env):] = obj_points_in_env
        
        #####

        super( IBSMesh, self ).__init__( np_cloud_env_final, np_cloud_obj )
