import math
import numpy as np
import os

import trimesh
from scipy import linalg

from transforms3d.derivations.eulerangles import z_rotation
from transforms3d.affines import compose

from it.testing.tester import Tester

if __name__ == '__main__':

    working_directory = "./data/it/IBSMesh_400_4_OnGivenPointCloudWeightedSampler_5_500"

    tester = Tester(working_directory, working_directory + "/single_testing.json")

    environment = trimesh.load_mesh('./data/it/gates400.ply', process=False)

    #testing_point = [-0.48689266781021423, -0.15363679409350514,0.8177121144402457]
    #testing_point = [-0.97178262, -0.96805501, 0.82738298] #in the edge of table, but with floor
    testing_point = [-2.8, 1., 0.00362764] # half inside the scene, half outside

    idx_ray, intersections = tester.intersections_with_scene(environment, testing_point)
    angles_with_best_score = tester.best_angle_by_affordance(environment, testing_point)
    all_distances, resumed_distances, missed = tester.measure_scores(environment, testing_point)

    # as this is a run with only one affordance to test, only get the first row of results
    first_affordance_scores =angles_with_best_score[0]
    score = first_affordance_scores[2]
    angle = first_affordance_scores[1]
    agglo_position = int(first_affordance_scores[0])

    affordance_name = tester.affordances[0][0]
    affordance_object = tester.affordances[0][1]
    tri_mesh_object_file = tester.objs_filenames[0]

    # visualizing
    bowl = trimesh.load_mesh(tri_mesh_object_file, process=False)
    print("distances score = ", score)
    idx_from = agglo_position * 512
    idx_to = idx_from + 512
    pv_begin = tester.compiled_pv_begin[idx_from:idx_to]
    pv_direction = tester.compiled_pv_direction[idx_from:idx_to]
    provenance_vectors = trimesh.load_path(np.hstack((pv_begin, pv_begin + pv_direction)).reshape(-1, 2, 3))
    pv_intersections = intersections[idx_from:idx_to]

    R = z_rotation(angle)  # rotation matrix
    Z = np.ones(3)  # zooms
    T = testing_point
    A = compose(T, R, Z)
    bowl.apply_transform(A)
    environment.visual.face_colors = [100, 100, 100, 100]
    bowl.visual.face_colors = [0, 255, 0, 100]
    scene = trimesh.Scene([provenance_vectors, trimesh.points.PointCloud(pv_intersections), environment, bowl])
    scene.show()
    #bowl.apply_transform(linalg.inv(A))
