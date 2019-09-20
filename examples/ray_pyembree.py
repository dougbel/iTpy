import trimesh
import numpy as np
from open3d import open3d as o3d
from transforms3d.affines import compose
import time
from it.testing.tester import Tester


last_position = np.array( [ 0, 0, 0 ] )
index = 0 
R = np.eye(3) #rotation matrix
Z = np.ones(3) #zooms

def move_object(scene):
    """
    A callback passed to a scene viewer which will update
    transforms in the viewer periodically.
    Parameters
    -------------
    scene : trimesh.Scene
      Scene containing geometry
    """
    global last_position
    global index

    translation = points_to_test[index] - last_position
    index += 1

    T = translation #translations
   
    A = compose(T, R, Z)

    # take one of the two spheres arbitrarily
    node = environment.graph.nodes_geometry[1]
    # apply the transform to the node
    scene.graph.update(node, matrix=A)
    

def test_collision(points_to_test):
    collision_tester = trimesh.collision.CollisionManager() #TODO check if exists a faster lib
    collision_tester.add_object('scene',scene)
    no_collided=[]
    last_position = np.array( [ 0, 0, 0 ] )
    index=0
    progress = 0
    period =[]

    for point in points_to_test:
        start = time.time() ## timing execution
        translation = point - last_position
        
        bowl.apply_translation(translation)
        in_collision = collision_tester.in_collision_single(bowl)   ## TEST WITH Open3D is_intersecting(self, arg0)
        if not in_collision:
            no_collided.append( point )
        last_position = point

        current_percent =  int(100*index/points_to_test.shape[0])
        if current_percent-progress >  0 :
            progress = current_percent
            print( progress , "%")
        index+= 1
        end = time.time()   ## timing execution
        period.append(end - start)   ## timing execution

    print (np.asarray(period).mean() , " seconds on each collision test" )  ## timing execution
    print (60/np.asarray(period).mean() , " collision test each min" )  ## timing execution
    return no_collided

def test_it(points_to_test):
    
    tester = It_Tester("./data/", "./data/testing.json")
    
    output = []

    index=0
    progress = 0
    
    period =[]

    for point in points_to_test:
        start = time.time() ## timing execution

        angles_with_best_scores = tester.best_angle_by_distance_by_affordance(scene, point)
        
        output.append( list(point) + list(angles_with_best_scores[0]) )
        
        current_percent =  int(100*index/points_to_test.shape[0])
        if current_percent-progress >  0 :
            progress = current_percent
            print( progress , "%")
        index+= 1

        end = time.time()   ## timing execution
        period.append(end - start)   ## timing execution


    print (np.asarray(period).mean() , " seconds on iT test" )  ## timing execution
    print (60/np.asarray(period).mean() , " iT tests each min" )  ## timing execution

    return np.asarray(output).reshape(-1,5)



if __name__ == '__main__':

    # test on a sphere mesh
    scene = trimesh.load_mesh('./data/it/gates400d.ply',process=False)
    # make mesh transparent- ish
    scene.visual.face_colors = [100, 100, 100, 100]


    #Read points to test
    global points_to_test
    points_to_test  = np.asarray(o3d.io.read_point_cloud("./data/Scenegrook_gates400_1611002_5 percent/1564147275_samplePoints.pcd").points)

    
    #Read information from test
    bowl = trimesh.load_mesh('./data/Place/bowl.ply',process=False)
    bowl.visual.face_colors = [0, 255, 0, 255]

    results_it_test = test_it(points_to_test[:10000]) ###############################################################################################################
    np.savetxt('output_it_test.csv', results_it_test, delimiter=',', fmt='%f')

    no_collission = test_collision(points_to_test[:10000]) ###############################################################################################################
    output = trimesh.points.PointCloud(np.asarray(no_collission))
    #no_collission_pcd = o3d.geometry.PointCloud()
    #no_collission_pcd.points = o3d.utility.Vector3dVector(no_collided)
    #o3d.io.write_point_cloud("no_collission_points.pcd",no_collission_pcd )


    # create a visualization scene with rays, hits, and mesh
    environment = trimesh.Scene( [ scene, 
                            bowl, 
                            output] )

    # display the scene with callback
    environment.show()

    #environment.show(callback=move_object)



    #TODO not permit too far points 
    #TODO measure performance
    #TODO cut partially the mesh aorund the testing point
    #TODO use libraries as cupy to work with numpy arrays
    #TODO fin alternative ray tracing approaches that uses the gpu
    #TODO generate and try scores
