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

'''class IBS_mesh:

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
        size_cloud_env = np_cloud_env_final.shape[0]
        size_cloud_obj = np_cloud_obj.shape[0]
        
        self.points = np.empty( ( size_cloud_env + size_cloud_obj, 3 ), np.float64 )
        self.points[:size_cloud_env] = np_cloud_env_final
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
        
        self.vertices = voro.vertices[ idx_ibs_vertices ]

        #with new indexation based on cloud_ibs
        self.ridge_vertices = []
        for ridge in voro.ridge_vertices:
            if all(idx_vertice in idx_ibs_vertices for idx_vertice in ridge):
                mapped_idx_ridge = [idx_ibs_vertices.index(idx_vertice) if idx_vertice != -1 else -1 for idx_vertice  in ridge]
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
        

if __name__ == '__main__':

    import time
    import util


    tri_mesh_obj = trimesh.load_mesh("./data/bowl.ply")
    od3_mesh_obj = o3d.io.read_triangle_mesh("./data/bowl.ply")
    od3_cloud_obj_poisson = o3d.geometry.sample_points_poisson_disk( od3_mesh_obj, 400 )

    obj_min_bound = od3_mesh_obj.get_min_bound()
    obj_max_bound = od3_mesh_obj.get_max_bound()
    
    tri_mesh_env = trimesh.load_mesh('./data/table.ply')
    tri_mesh_env_segmented = util.slide_mesh_by_bounding_box(tri_mesh_env, obj_min_bound, obj_max_bound)  

    #it was not possible create a TriangleMesh on the fly, then save an open a ply file
    trimesh.exchange.export.export_mesh(tri_mesh_env_segmented,"segmented_environment.ply", "ply")
    od3_mesh_env_segmented = o3d.io.read_triangle_mesh('segmented_environment.ply')
    remove("segmented_environment.ply")
    od3_cloud_env_poisson = o3d.geometry.sample_points_poisson_disk( od3_mesh_env_segmented, 400 )

    np_cloud_env_poisson = np.asarray(od3_cloud_env_poisson.points)
    np_cloud_obj_poisson = np.asarray(od3_cloud_obj_poisson.points)


    start = time.time() ## timing execution

    ibs_calculator = IBS_mesh( np_cloud_env_poisson,tri_mesh_env,  np_cloud_obj_poisson, tri_mesh_obj )   # poisson disk sampling

    
    ########################################################################################################################
    # COMPILATION OF EDGES FOR VISUALIZATION
    idx_ridges_from = []
    idx_ridges_to = []
    
    for ridge in ibs_calculator.ridge_vertices:
        if  -1 in ridge:
            continue
        #ridge.insert(0,ridge[-1])
        for i in range( -1, len(ridge)-1 ) :
            idx_ridges_from.append(ridge[i])
            idx_ridges_to.append(ridge[i+1])
            
    edges_from = ibs_calculator.vertices[ idx_ridges_from ]
    edges_to = ibs_calculator.vertices[ idx_ridges_to ]
    
    ridges = trimesh.load_path( np.hstack(( edges_from, 
                                            edges_to
                                            )).reshape(-1, 2, 3) )


   
    # create a visualization environment with rays, hits, and mesh
    visualizer = trimesh.Scene( [ #trimesh.points.PointCloud( ibs_calculator.cloud_ibs, colors=[255,255,0,255] ), 
                                  #trimesh.points.PointCloud( np_ibs_vertices_extracted, colors=[0,255,0,255] ), 
                                  trimesh.points.PointCloud( np.asarray(od3_cloud_obj_poisson.points) , colors=[0,0,255,255] ),
                                  tri_mesh_obj,
                                  #trimesh.points.PointCloud( np.asarray(od3_cloud_env_poisson.points), colors=[255,0,0,255] ),
                                  ridges
                                  #ridges_recorted
                                  ] )

    # display the environment with callback
    visualizer.show()


        ########################################################################################################################
    # CROPPING MESH AND IBS POINT CLOUD

    #extracting point no farther than the principal sphere
    radio = np.linalg.norm( obj_max_bound - obj_min_bound )
    np_pivot = np.asarray( obj_max_bound + obj_min_bound ) / 2
    [idx_extracted, np_ibs_vertices_extracted ]= util.extract_by_distance( ibs_calculator.vertices, np_pivot, radio )

    #cutting faces in the polygon mesh
    faces_idx_ridges_from = []
    faces_idx_ridges_to = []
    for ridge in ibs_calculator.ridge_vertices:
        if  -1 in ridge:
            continue
        face_in_boundary = True
        for idx_vertice in ridge:
            if idx_vertice not in idx_extracted:
                face_in_boundary = False
                break
        if not face_in_boundary:
            continue
        #ridge.insert(0,ridge[-1])
        for i in range( -1, len(ridge)-1 ):
            faces_idx_ridges_from.append(ridge[i])
            faces_idx_ridges_to.append(ridge[i+1])
    
    edges_from = ibs_calculator.vertices[ faces_idx_ridges_from ]
    edges_to = ibs_calculator.vertices[ faces_idx_ridges_to ]

    ridges_recorted = trimesh.load_path( np.hstack(( edges_from, 
                                            edges_to
                                            )).reshape(-1, 2, 3) )



    
    visualizer2 = trimesh.Scene( [ #trimesh.points.PointCloud( ibs_calculator.cloud_ibs, colors=[255,255,0,255] ), 
                                  trimesh.points.PointCloud( np_ibs_vertices_extracted, colors=[0,255,0,255] ), 
                                  trimesh.points.PointCloud( np.asarray(od3_cloud_obj_poisson.points) , colors=[0,0,255,255] ),
                                  tri_mesh_obj,
                                  trimesh.points.PointCloud( ibs_calculator.points, colors=[255,0,0,255] ),
                                  #ridges
                                  ridges_recorted
                                  ] )

    # display the environment with callback
    visualizer2.show()


    tri_mesh_ibs = ibs_calculator.get_trimesh()
    

    tri_mesh_ibs = tri_mesh_ibs.subdivide()

    sphere_ro = np.linalg.norm( obj_max_bound - obj_min_bound )
    sphere_center = np.asarray( obj_max_bound + obj_min_bound ) / 2
    tri_mesh_ibs_segmented = util.slide_mesh_by_sphere( tri_mesh_ibs, sphere_center, sphere_ro ) 

    #tri_mesh_ibs_segmented.export("ibs_mesh_segmented.ply","ply")

    print( "is convex:" + str(tri_mesh_ibs.is_convex))
    print( "is empty:" + str(tri_mesh_ibs.is_empty))
    print( "is watertight:" + str(tri_mesh_ibs.is_watertight))
    print( "is widing consistent:" + str(tri_mesh_ibs.is_winding_consistent))

    
    
    tri_mesh_obj.visual.face_colors = [0, 255, 0, 100]
    tri_mesh_env.visual.face_colors = [100, 100, 100, 100]
    tri_mesh_ibs_segmented.visual.face_colors = [0, 0, 150, 100]
    tri_mesh_ibs.visual.face_colors = [255, 0, 0, 50]

    visualizer3 = trimesh.Scene([ 
                                tri_mesh_env, 
                                tri_mesh_obj,
                                tri_mesh_ibs_segmented,
                                #tri_mesh_ibs,
                                ridges_recorted
                                ] )

    # display the environment with callback
    visualizer3.show()

'''