import numpy as np

import trimesh

from it.testing.tester import Tester

if __name__ == '__main__':

    working_directory = "./data/it/IBSMesh_400_4_OnGivenPointCloudWeightedSampler_5_500"

    tester = Tester(working_directory, working_directory+"/single_testing.json")
    environment = trimesh.load_mesh('./data/it/gates400.ply', process=False)

    idx_ray, intersections = tester.intersections_with_scene(environment, [-0.48689266781021423 , -0.15363679409350514 , 0.8177121144402457])
    angles_with_best_scores = tester.best_angle_by_affordance(environment, [-0.48689266781021423 , -0.15363679409350514 , 0.8177121144402457])
    all_distances, resumed_distances = tester.measure_scores(environment, [-0.48689266781021423 , -0.15363679409350514 , 0.8177121144402457])

    affordance_name = tester.affordances[0][0]
    affordance_object = tester.affordances[0][1]
    tri_mesh_object_file = working_directory + "/"+affordance_name + "/" + affordance_name + "_" + affordance_object + "_object.ply"

    bowl = trimesh.load_mesh(tri_mesh_object_file, process=False)
    bowl.apply_translation([-0.48689266781021423 , -0.15363679409350514 , 0.8177121144402457])

    collision_tester = trimesh.collision.CollisionManager()
    collision_tester.add_object('scene', environment)
    in_collision = collision_tester.in_collision_single(bowl)

    print("In collission: " + str(in_collision))

    scores = resumed_distances[0]
    for i in range(tester.num_orientations):
        print("distances score = ", scores[i])
        idx_from = i * 512
        idx_to = idx_from + 512
        pv_begin = tester.compiled_pv_begin[idx_from:idx_to]
        pv_direction = tester.compiled_pv_direction[idx_from:idx_to]
        provenance_vectors = trimesh.load_path(np.hstack((pv_begin, pv_begin + pv_direction)).reshape(-1, 2, 3))
        pv_intersections = intersections[idx_from:idx_to]

        environment.visual.face_colors = [100, 100, 100, 100]
        bowl.visual.face_colors = [0, 255, 0, 100]
        scene = trimesh.Scene([
            provenance_vectors,
            trimesh.points.PointCloud(pv_intersections),
            environment,
            bowl])
        scene.show()
