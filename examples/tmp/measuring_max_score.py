import math
import numpy as np
import os

import trimesh
from scipy import linalg

from transforms3d.derivations.eulerangles import z_rotation
from transforms3d.affines import compose

from it import util
from it.testing.results import Analyzer
from it.testing.tester import Tester

if __name__ == '__main__':
    working_directory = "./data/it/IBSMesh_400_4_OnGivenPointCloudWeightedSampler_5_500"

    tester = Tester(working_directory, working_directory + "/single_testing.json")

    environment = trimesh.load_mesh('./data/it/gates400.ply', process=False)

    # testing_point = [-0.48689266781021423, -0.15363679409350514,0.8177121144402457]
    testing_point = [-0.97178262, -0.96805501, 0.82738298] #in the edge of table, but with floor
    # testing_point = [-2.8, 1., 0.00362764]  # half inside the scene, half outside

    testing_point = [0,0,0]

    influence_radio_ratio = 1.0

    # visualizing
    tri_mesh_object_file = tester.objs_filenames[0]
    tri_mesh_obj = trimesh.load_mesh(tri_mesh_object_file, process=False)
    ro, center = util.influence_sphere(tri_mesh_obj, radio_ratio=influence_radio_ratio)

    influence_sphere = trimesh.primitives.Sphere(radius=ro, center=center, subdivisions=5)

    pv_points = tester.compiled_pv_begin[0:512]
    pv_vectors = tester.compiled_pv_direction[0:512]

    expected_pv_end = pv_points + pv_vectors

    # looking for the nearest ray intersections
    (__,
     idx_ray,
     intersections) = influence_sphere.ray.intersects_id(
        ray_origins=pv_points,
        ray_directions=pv_vectors,
        return_locations=True,
        multiple_hits=False)

    analyzer = Analyzer(idx_ray, intersections, 1, [ro], 1, expected_pv_end)

    __, max_distance, __ = analyzer.measure_scores()

    print(max_distance)

    pv_origin = trimesh.points.PointCloud(pv_points, color=[255, 0, 0, 255])
    pv_calculated_pv_end = trimesh.points.PointCloud(expected_pv_end, color=[255, 0, 0, 250])
    pv_intersections = trimesh.points.PointCloud(intersections, color=[0, 0, 255, 250])


    tri_mesh_obj.visual.face_colors = [0, 255, 0, 255]
    influence_sphere.visual.face_colors = [0, 0, 255, 25]

    pv_3d_path = np.hstack(
        (pv_points, pv_points + pv_vectors)).reshape(-1, 2, 3)
    pv_max_path = np.hstack((pv_points, intersections)).reshape(-1, 2, 3)




    provenance_vectors = trimesh.load_path(pv_3d_path)
    provenance_vectors_max_path = trimesh.load_path(pv_max_path)



    scene = trimesh.Scene([
        provenance_vectors_max_path,
        pv_origin,
        pv_intersections,
        influence_sphere,
        pv_calculated_pv_end,
        tri_mesh_obj
    ])
    scene.show()