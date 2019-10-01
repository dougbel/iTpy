import pandas as pd

from it.training.maxdistancescalculator import MaxDistancesCalculator
from it.training.ibs import IBSMesh
from it.training.sampler import *
from it.training.trainer import Trainer
from it.training.agglomerator import Agglomerator
from it.training.saver import Saver

if __name__ == '__main__':
    interactions_data = pd.read_csv("./data/interactions/interaction.csv")

    to_test = 'hang'
    interaction = interactions_data[interactions_data['interaction'] == to_test]

    tri_mesh_env = trimesh.load_mesh(interaction.iloc[0]['tri_mesh_env'])
    tri_mesh_obj = trimesh.load_mesh(interaction.iloc[0]['tri_mesh_obj'])
    obj_name = interaction.iloc[0]['obj']
    env_name = interaction.iloc[0]['env']
    affordance_name = interaction.iloc[0]['interaction']

    obj_min_bound = np.asarray(tri_mesh_obj.vertices).min(axis=0)
    obj_max_bound = np.asarray(tri_mesh_obj.vertices).max(axis=0)

    extension = np.linalg.norm(obj_max_bound - obj_min_bound)
    middle_point = (obj_max_bound + obj_min_bound) / 2

    tri_mesh_env_segmented = util.slide_mesh_by_bounding_box(tri_mesh_env, middle_point, extension)

    init_size_sampling = 400
    resamplings = 4

    ibs_calculator = IBSMesh(init_size_sampling, resamplings)
    ibs_calculator.execute(tri_mesh_env_segmented, tri_mesh_obj)

    ################################
    # GENERATING AND SEGMENTING IBS MESH
    ################################

    tri_mesh_ibs = ibs_calculator.get_trimesh()
    # tri_mesh_ibs = tri_mesh_ibs.subdivide()

    sphere_ro = np.linalg.norm(obj_max_bound - obj_min_bound)
    sphere_center = np.asarray(obj_max_bound + obj_min_bound) / 2

    tri_mesh_ibs_segmented = util.slide_mesh_by_sphere(tri_mesh_ibs, sphere_center, sphere_ro)

    rate_ibs_samples = 5
    rate_generated_random_numbers = 500
    np_cloud_env = ibs_calculator.points[: ibs_calculator.size_cloud_env]

    # sampler = PoissonDiscRandomSampler( rate_ibs_samples )
    # sampler = PoissonDiscWeightedSampler( rate_ibs_samples=rate_ibs_samples, rate_generated_random_numbers=rate_generated_random_numbers)
    # sampler =  OnVerticesRandomSampler()
    # sampler =  OnVerticesWeightedSampler( rate_generated_random_numbers=rate_generated_random_numbers )
    # sampler =  OnGivenPointCloudRandomSampler( np_input_cloud = np_cloud_env )
    sampler = OnGivenPointCloudWeightedSampler(np_input_cloud=np_cloud_env,
                                               rate_generated_random_numbers=rate_generated_random_numbers)

    trainer = Trainer(tri_mesh_ibs=tri_mesh_ibs_segmented, tri_mesh_env=tri_mesh_env, sampler=sampler)

    agglomerator = Agglomerator(trainer)

    max_distances = MaxDistancesCalculator(trainer, tri_mesh_obj)

    output_subdir = "IBSMesh_" + str(init_size_sampling) + "_" + str(resamplings)+ "_"
    output_subdir += sampler.__class__.__name__ + "_"+ str(rate_ibs_samples) + "_" + str(rate_generated_random_numbers)



    Saver(affordance_name, env_name, obj_name, agglomerator,max_distances, ibs_calculator, tri_mesh_obj, output_subdir)

    # VISUALIZATION
    provenance_vectors = trimesh.load_path(
        np.hstack((trainer.pv_points, trainer.pv_points + trainer.pv_vectors)).reshape(-1, 2, 3))

    tri_mesh_obj.visual.face_colors = [0, 255, 0, 200]
    tri_mesh_ibs_segmented.visual.face_colors = [0, 0, 255, 100]
    tri_mesh_env.visual.face_colors = [200, 200, 200, 150]

    scene = trimesh.Scene([
        tri_mesh_obj,
        tri_mesh_env,
        tri_mesh_ibs_segmented,
        provenance_vectors
    ])
    scene.show()
