import trimesh

import numpy as np
import open3d as o3d
import pandas as pd

from it import util
from it.testing.results import Analyzer
from it.training.sampler import *
from it.training.trainer import Trainer


class MaxDistance:

    def __init__(self, trainer, tri_mesh_obj):
        '''
        This is a replic of method Trainer._get_max_distance_score but is presented here only to show the way of calculation
        :param trainer:
        :param tri_mesh_obj:
        '''

        ro, center = util.influence_sphere(tri_mesh_obj)

        self.influence_sphere = trimesh.primitives.Sphere(radius=ro, center=center, subdivisions=5)

        calculated_pv_end = trainer.pv_points + trainer.pv_vectors

        # looking for the nearest ray intersections
        (__,
         self.idx_ray,
         self.intersections) = self.influence_sphere.ray.intersects_id(
            ray_origins=trainer.pv_points,
            ray_directions=trainer.pv_vectors,
            return_locations=True,
            multiple_hits=False)

        analyzer = Analyzer(self.idx_ray, self.intersections, 1, [ro], 1, calculated_pv_end)

        self.all_max_distances, self.max_distance, missed = analyzer.measure_scores()



if __name__ == '__main__':
    interactions_data = pd.read_csv("./data/interactions/interaction.csv")

    to_test = 'hang'
    interaction = interactions_data[interactions_data['interaction'] == to_test]

    tri_mesh_env = trimesh.load_mesh(interaction.iloc[0]['tri_mesh_env'])
    tri_mesh_obj = trimesh.load_mesh(interaction.iloc[0]['tri_mesh_obj'])
    tri_mesh_ibs_segmented = trimesh.load_mesh(interaction.iloc[0]['tri_mesh_ibs_segmented'])
    np_cloud_env = np.asarray(o3d.io.read_point_cloud(interaction.iloc[0]['o3d_cloud_sources_ibs']).points)

    rate_generated_random_numbers = 500

    sampler_ibs_srcs_weighted = OnGivenPointCloudWeightedSampler(np_cloud_env, rate_generated_random_numbers)

    trainer_weighted = Trainer(tri_mesh_ibs_segmented, tri_mesh_env, sampler_ibs_srcs_weighted)

    max_d = MaxDistance(trainer_weighted, tri_mesh_obj)

    print(max_d.max_distance)

    pv_origin = trimesh.points.PointCloud(trainer_weighted.pv_points, color=[255, 0, 0, 255])

    tri_mesh_env.visual.face_colors = [100, 100, 100, 255]
    tri_mesh_obj.visual.face_colors = [0, 255, 0, 255]
    tri_mesh_ibs_segmented.visual.face_colors = [0, 0, 255, 100]
    max_d.influence_sphere.visual.face_colors = [0, 0, 255, 25]

    pv_3d_path = np.hstack((trainer_weighted.pv_points, trainer_weighted.pv_points + trainer_weighted.pv_vectors)).reshape(-1, 2, 3)
    pv_max_path = np.hstack((trainer_weighted.pv_points, max_d.intersections)).reshape(-1, 2, 3)

    pv_intersections = trimesh.points.PointCloud(max_d.intersections, color=[0, 0, 255, 250])


    provenance_vectors = trimesh.load_path(pv_3d_path)
    provenance_vectors_max_path = trimesh.load_path(pv_max_path)

    scene = trimesh.Scene([
        provenance_vectors,
        pv_origin,
        tri_mesh_ibs_segmented,
        tri_mesh_obj,
        tri_mesh_env
    ])
    scene.show()

    scene = trimesh.Scene([
        provenance_vectors_max_path,
        pv_origin,
        pv_intersections,
        max_d.influence_sphere,
        tri_mesh_obj
    ])
    scene.show()