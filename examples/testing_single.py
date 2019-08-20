import numpy as np

import trimesh

from it.testing.tester import Tester



if __name__ == '__main__':



    tester = Tester("./data/it", "./data/it/single_testing.json")
    scene = trimesh.load_mesh('./data/it/gates400.ply',process=False)
    scene.visual.face_colors = [100, 100, 100, 100]


    idx_ray, intersections = tester.intersections_with_scene(scene, [-0.97178262, -0.96805501,  0.82638292])
    angles_with_best_scores = tester.best_angle_by_affordance(scene, [-0.97178262, -0.96805501,  0.82638292])
    all_distances, resumed_distances = tester.measure_scores(scene, [-0.97178262, -0.96805501,  0.82638292])

    
    affordance_name = tester.affordances[0][0]
    affordance_object = tester.affordances[0][1]
    tri_mesh_object_file = "./data/it/" + affordance_name + "/" + affordance_name + "_" + affordance_object + "_object.ply"

    bowl        = trimesh.load_mesh( tri_mesh_object_file, process=False)
    bowl.visual.face_colors = [10, 255, 10, 100]
    bowl.apply_translation( [-0.97178262, -0.96805501,  0.82638292] )


    collision_tester = trimesh.collision.CollisionManager()
    collision_tester.add_object('scene',scene)
    in_collision = collision_tester.in_collision_single(bowl)


    print("In collission: " + str(in_collision))


    pv_begin = tester.compiled_pv_begin[:512]
    pv_direction = tester.compiled_pv_direction[:512]
    provenance_vectors = trimesh.load_path( np.hstack(( pv_begin, pv_begin + pv_direction)).reshape(-1, 2, 3) )
    pv_intersections = intersections[:512]


    scene = trimesh.Scene( [ #trimesh.points.PointCloud(tests.compiled_sampled_points), 
                            #trimesh.points.PointCloud(tests.compiled_pv_end), 
                            provenance_vectors,
                            trimesh.points.PointCloud(pv_intersections),
                            scene,
                            bowl ])
    scene.show()