import time
import os
import numpy as np
import matplotlib.pyplot as plt

import trimesh

import it.util as util
from it.training.ibs import IBS

if __name__ == '__main__':
    '''
    Generates information to establish the correlation between IBS calculation time and point clouds sampling density.
    Using point cloud to create an IBS which no pierces surfaces requires a high-density point sampling, 
    thereon generate it is highly time-consuming.
    '''

    tri_mesh_obj = trimesh.load_mesh("./data/interactions/table_bowl/bowl.ply")
    tri_mesh_env = trimesh.load_mesh('./data/interactions/table_bowl/table_segmented.ply')

    samples = []
    timing = []

    directory = './output/ibs_timing_cloud_generation/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    last_iteration = 1

    for i in range(1, 40):
        size_sampling = i * 100
        np_cloud_env_poisson = util.sample_points_poisson_disk(tri_mesh_env, size_sampling)
        np_cloud_obj_poisson = util.sample_points_poisson_disk(tri_mesh_obj, size_sampling)

        start = time.time()  # timing execution
        ibs_calculator = IBS(np_cloud_env_poisson, np_cloud_obj_poisson)
        end = time.time()  # timing execution

        samples.append(size_sampling * 2)
        timing.append(end - start)
        np_results = np.hstack((np.array(samples).reshape(-1, 1), np.array(timing).reshape(-1, 1)))
        np_results.tofile(directory + 'ibs_cloud_timing' + str(i) + '.dat')
        last_iteration = i

    np_results = np.fromfile(directory + 'ibs_cloud_timing' + str(last_iteration) + '.dat')
    np_results = np_results.reshape(-1, 2)

    plt.plot(np_results[:, 0], np_results[:, 1])

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.title('Sampled points vs Calculation time [s]')
    plt.ylabel('Calculation time [s]')
    plt.xlabel('Sampled points')

    plt.savefig(directory + 'ibs_cloud_timing.png')

    plt.show()
