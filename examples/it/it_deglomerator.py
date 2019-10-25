import numpy as np

import trimesh

from it.testing.deglomerator import Deglomerator

if __name__ == '__main__':
    tests = Deglomerator("./data/it/IBSMesh_400_4_OnGivenPointCloudWeightedSampler_5_500/hang", "hang", "umbrella")
    print(tests.pv_points)

    # VISUALIZATION
    provenance_vectors = trimesh.load_path(
        np.hstack((tests.pv_points, tests.pv_points + tests.pv_vectors)).reshape(-1, 2, 3))

    pv_origin = trimesh.points.PointCloud(tests.pv_points, color=[0, 0, 255, 250])

    scene = trimesh.Scene([
        provenance_vectors,
        pv_origin
    ])
    scene.show()
