import open3d as o3d
import pandas as pd
import json

from it.training.sampler import *
from it.training.trainer import Trainer


class MaxDistancesCalculator:
    #influence_radio
    #sum_max_distances

    def __init__(self, pv_points, pv_vectors, tri_mesh_obj):
        '''
        This is a replic of method Trainer._get_max_distance_score but is presented here only to show the way
        of calculation
        :param trainer:
        :param tri_mesh_obj:
        '''
        self.influence_radio, self.influence_center = util.influence_sphere(tri_mesh_obj)

        self.sphere_of_influence = trimesh.primitives.Sphere(radius=self.influence_radio,
                                                             center=self.influence_center, subdivisions=5)

        expected_intersections = pv_points + pv_vectors

        # looking for the nearest ray intersections
        (__,
         idx_ray,
         calculated_intersections) = self.sphere_of_influence.ray.intersects_id(
            ray_origins=pv_points,
            ray_directions=pv_vectors,
            return_locations=True,
            multiple_hits=False)

        self.max_distances = np.linalg.norm(calculated_intersections - expected_intersections, axis=1)
        self.sum_max_distances = np.sum(self.max_distances)

    def get_info(self):
        info = {}
        info['obj_influence_radio'] = self.influence_radio
        info['sum_max_distances'] = self.sum_max_distances
        info['max_distances'] = self.max_distances.tolist()
        return info


if __name__ == '__main__':
    from sklearn import preprocessing

    interactions_data = pd.read_csv("./data/interactions/interaction.csv")

    to_test = 'ride'
    interaction = interactions_data[interactions_data['interaction'] == to_test]

    tri_mesh_env = trimesh.load_mesh(interaction.iloc[0]['tri_mesh_env'])
    tri_mesh_obj = trimesh.load_mesh(interaction.iloc[0]['tri_mesh_obj'])
    tri_mesh_ibs_segmented = trimesh.load_mesh(interaction.iloc[0]['tri_mesh_ibs_segmented'])
    np_cloud_env = np.asarray(o3d.io.read_point_cloud(interaction.iloc[0]['o3d_cloud_sources_ibs']).points)

    rate_generated_random_numbers = 500

    sampler_ibs_srcs_weighted = OnGivenPointCloudWeightedSampler(np_cloud_env, rate_generated_random_numbers)

    trainer_weighted = Trainer(tri_mesh_ibs_segmented, tri_mesh_env, sampler_ibs_srcs_weighted)
    max_d = MaxDistancesCalculator(trainer_weighted.pv_points, trainer_weighted.pv_vectors, tri_mesh_obj)

    print(max_d.sum_max_distances)

    pv_origin = trimesh.points.PointCloud(trainer_weighted.pv_points, color=[255, 0, 0, 255])

    tri_mesh_env.visual.face_colors = [100, 100, 100, 255]
    tri_mesh_obj.visual.face_colors = [0, 255, 0, 255]
    tri_mesh_ibs_segmented.visual.face_colors = [0, 0, 255, 100]
    max_d.sphere_of_influence.visual.face_colors = [0, 0, 255, 25]

    pv_3d_path = np.hstack((trainer_weighted.pv_points,
                            trainer_weighted.pv_points + trainer_weighted.pv_vectors)).reshape(-1, 2, 3)

    #pv_normalized_vectors = preprocessing.normalize(trainer_weighted.pv_vectors)
    pv_max_vectors = preprocessing.normalize(trainer_weighted.pv_vectors) * max_d.max_distances.reshape(-1, 1)
    calculated_max_intersections = trainer_weighted.pv_points + pv_max_vectors

    pv_max_path = np.hstack((trainer_weighted.pv_points,
                             calculated_max_intersections)).reshape(-1, 2, 3)

    pv_intersections = trimesh.points.PointCloud(calculated_max_intersections, color=[0, 0, 255, 250])

    provenance_vectors = trimesh.load_path(pv_3d_path)
    provenance_vectors_max_path = trimesh.load_path(pv_max_path)

    scene = trimesh.Scene([
        provenance_vectors,
        pv_origin,
        tri_mesh_ibs_segmented,
        tri_mesh_obj,
        tri_mesh_env
    ])

    scene.show(flags={'cull': False, 'wireframe':False, 'axis': False})


    scene = trimesh.Scene([
        provenance_vectors_max_path,
        pv_origin,
        pv_intersections,
        max_d.sphere_of_influence,
        tri_mesh_obj
    ])
    scene.show(flags={'cull': False})
