import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os

import open3d as o3d
import trimesh

from  it.training.trainer import Trainer
from  it.training.trainer import MeshSamplingMethod as msm
import it.util as util

def get_camera(scene):
  np.set_printoptions(suppress=True)
  print(scene.camera_transform)


if __name__ == '__main__':
    #RIDE A MOTORCYCLE
    env = "motorbike"
    obj = "rider"
    tri_mesh_env = trimesh.load_mesh("./data/interactions/motorbike_rider/motorbike.ply")
    tri_mesh_obj = trimesh.load_mesh("./data/interactions/motorbike_rider/biker.ply")
    tri_mesh_ibs_segmented = trimesh.load_mesh("./data/interactions/motorbike_rider/ibs_motorbike_biker_sampled_3000_resamplings_2.ply")

    #PLACE BOWL TABLE
    '''env = "table"
    obj = "bowl"
    tri_mesh_ibs_segmented = trimesh.load_mesh('./data/pv/ibs_mesh_segmented.ply')
    tri_mesh_env = trimesh.load_mesh('./data/table.ply')
    tri_mesh_obj = trimesh.load_mesh('./data/bowl.ply')'''

    tri_mesh_env.visual.face_colors = [100, 100, 100, 255]
    tri_mesh_obj.visual.face_colors = [0, 255, 0, 100]
  

    directory = './output/pv_selection_weighted/'

    if not os.path.exists( directory ):
            os.makedirs( directory )


    for rate_s in range(2,40,2):
        for rate_r in range(2,40,2):

            rate_samples_in_ibs = rate_s
            rate_random_samples = rate_r

            trainer_weighted = Trainer( tri_mesh_ibs_segmented, tri_mesh_env, msm.ON_MESH_WEIGHTED, rate_samples_in_ibs, rate_random_samples )
            #VISUALIZATION    
            tri_cloud_ibs = trimesh.points.PointCloud( trainer_weighted.np_cloud_ibs, color=[0,0,255,150] )
            pv_origin     = trimesh.points.PointCloud( trainer_weighted.pv_points, color=[0,0,255,250] )
            pv_3d_path = np.hstack(( trainer_weighted.pv_points,trainer_weighted.pv_points + trainer_weighted.pv_vectors)).reshape(-1, 2, 3)
            provenance_vectors = trimesh.load_path( pv_3d_path )
            scene = trimesh.Scene( [ provenance_vectors, pv_origin, tri_cloud_ibs, tri_mesh_env, tri_mesh_obj ])
            
            scene.camera_transform = [[ 1.,         0.,         0.,         0.04396537],
                                            [ 0.,         1.,         0.,         0.23108601],
                                            [-0.,         0.,         1.,         3.35716683],
                                            [ 0.,         0.,         0.,         1.        ]]

            png = scene.save_image()
            with open(directory+"weighted_ibssamples_"+str(rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+
                                "_randomchoiced_"+str(rate_random_samples*rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+"_1.png", 'wb') as f:
                f.write(png)
                f.close()


            scene.camera_transform = [[ 0.99948765, 0.02547537, -0.01937649, -0.00579407],
                                    [-0.02175125, 0.09651196, -0.99509413, -2.77415963],
                                    [-0.02348033, 0.99500575, 0.09701663, 0.41565603],
                                    [ 0.,         0.,         0.,         1.        ]]

            png = scene.save_image()
            with open(directory+"weighted_ibssamples_"+str(rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+
                                "_randomchoiced_"+str(rate_random_samples*rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+"_2.png", 'wb') as f:
                f.write(png)
                f.close()


            scene.camera_transform = [[-0.16499592, -0.05685685, 0.98465407, 3.39108636],
                                    [ 0.9861865,  0.00524637, 0.16555564, 0.71746308],
                                    [-0.01457883, 0.99836856, 0.05520583, 0.11499131],
                                    [ 0.,         0.,         0.,         1.        ]]
            png = scene.save_image()
            with open(directory+"weighted_ibssamples_"+str(rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+
                                "_randomchoiced_"+str(rate_random_samples*rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+"_3.png", 'wb') as f:
                f.write(png)
                f.close()


            scene.camera_transform = [[-0.53874073, -0.34343771, 0.76929121, 2.63196857],
                                    [ 0.8423785, -0.20601971, 0.49795014, 1.92104601],
                                    [-0.0125257,  0.9163004,  0.40029575, 1.34028108],
                                    [ 0.,         0.,         0.,         1.        ]]
            png = scene.save_image()
            with open(directory+"weighted_ibssamples_"+str(rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+
                                "_randomchoiced_"+str(rate_random_samples*rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+"_4.png", 'wb') as f:
                f.write(png)
                f.close()


            scene.camera_transform =  [[-0.34202215, -0.63235448, 0.69508896, 0.64184869],
                                    [ 0.92533648, -0.09784244, 0.36630486, 0.42891027],
                                    [-0.16362532, 0.76847555, 0.61860495, 0.41507145],
                                    [ 0.,         0.,         0.,         1.        ]]
            png = scene.save_image()
            with open(directory+"weighted_ibssamples_"+str(rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+
                                "_randomchoiced_"+str(rate_random_samples*rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+"_5.png", 'wb') as f:
                f.write(png)
                f.close()

            num_bins = trainer_weighted.SAMPLE_SIZE*rate_samples_in_ibs
            n, bins, patches = plt.hist(trainer_weighted.chosen_ibs_points, num_bins, facecolor='gray', alpha=0.5)
            plt.savefig(directory+"weighted_ibssamples_"+str(rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+
                                "_randomchoiced_"+str(rate_random_samples*rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+"_histogram.png")


 
    #PLACE BOWL TABLE
    env = "table"
    obj = "bowl"
    tri_mesh_ibs_segmented = trimesh.load_mesh('./data/pv/ibs_mesh_segmented.ply')
    tri_mesh_env = trimesh.load_mesh('./data/table.ply')
    tri_mesh_obj = trimesh.load_mesh('./data/bowl.ply')

    tri_mesh_env.visual.face_colors = [100, 100, 100, 255]
    tri_mesh_obj.visual.face_colors = [0, 255, 0, 100]
  

    directory = './output/pv_selection_weighted/'

    if not os.path.exists( directory ):
            os.makedirs( directory )


    for rate_s in range(2,40,2):
        for rate_r in range(2,40,2):

            rate_samples_in_ibs = rate_s
            rate_random_samples = rate_r

            trainer_weighted = Trainer( tri_mesh_ibs_segmented, tri_mesh_env, msm.ON_MESH_WEIGHTED, rate_samples_in_ibs, rate_random_samples )
            #VISUALIZATION    
            tri_cloud_ibs = trimesh.points.PointCloud( trainer_weighted.np_cloud_ibs, color=[0,0,255,150] )
            pv_origin     = trimesh.points.PointCloud( trainer_weighted.pv_points, color=[0,0,255,250] )
            pv_3d_path = np.hstack(( trainer_weighted.pv_points,trainer_weighted.pv_points + trainer_weighted.pv_vectors)).reshape(-1, 2, 3)
            provenance_vectors = trimesh.load_path( pv_3d_path )
            scene = trimesh.Scene( [ provenance_vectors, pv_origin, tri_cloud_ibs, tri_mesh_env, tri_mesh_obj ])
            
            scene.camera_transform = [[ 1.,         0.,         0.,         0.04396537],
                                            [ 0.,         1.,         0.,         0.23108601],
                                            [-0.,         0.,         1.,         3.35716683],
                                            [ 0.,         0.,         0.,         1.        ]]

            png = scene.save_image()
            with open(directory+"weighted_ibssamples_"+str(rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+
                                "_randomchoiced_"+str(rate_random_samples*rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+"_1.png", 'wb') as f:
                f.write(png)
                f.close()


            scene.camera_transform = [[ 0.99948765, 0.02547537, -0.01937649, -0.00579407],
                                    [-0.02175125, 0.09651196, -0.99509413, -2.77415963],
                                    [-0.02348033, 0.99500575, 0.09701663, 0.41565603],
                                    [ 0.,         0.,         0.,         1.        ]]

            png = scene.save_image()
            with open(directory+"weighted_ibssamples_"+str(rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+
                                "_randomchoiced_"+str(rate_random_samples*rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+"_2.png", 'wb') as f:
                f.write(png)
                f.close()


            scene.camera_transform = [[-0.16499592, -0.05685685, 0.98465407, 3.39108636],
                                    [ 0.9861865,  0.00524637, 0.16555564, 0.71746308],
                                    [-0.01457883, 0.99836856, 0.05520583, 0.11499131],
                                    [ 0.,         0.,         0.,         1.        ]]
            png = scene.save_image()
            with open(directory+"weighted_ibssamples_"+str(rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+
                                "_randomchoiced_"+str(rate_random_samples*rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+"_3.png", 'wb') as f:
                f.write(png)
                f.close()


            scene.camera_transform = [[-0.53874073, -0.34343771, 0.76929121, 2.63196857],
                                    [ 0.8423785, -0.20601971, 0.49795014, 1.92104601],
                                    [-0.0125257,  0.9163004,  0.40029575, 1.34028108],
                                    [ 0.,         0.,         0.,         1.        ]]
            png = scene.save_image()
            with open(directory+"weighted_ibssamples_"+str(rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+
                                "_randomchoiced_"+str(rate_random_samples*rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+"_4.png", 'wb') as f:
                f.write(png)
                f.close()


            scene.camera_transform =  [[-0.34202215, -0.63235448, 0.69508896, 0.64184869],
                                    [ 0.92533648, -0.09784244, 0.36630486, 0.42891027],
                                    [-0.16362532, 0.76847555, 0.61860495, 0.41507145],
                                    [ 0.,         0.,         0.,         1.        ]]
            png = scene.save_image()
            with open(directory+"weighted_ibssamples_"+str(rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+
                                "_randomchoiced_"+str(rate_random_samples*rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+"_5.png", 'wb') as f:
                f.write(png)
                f.close()

            num_bins = trainer_weighted.SAMPLE_SIZE*rate_samples_in_ibs
            n, bins, patches = plt.hist(trainer_weighted.chosen_ibs_points, num_bins, facecolor='gray', alpha=0.5)
            plt.savefig(directory+"weighted_ibssamples_"+str(rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+
                                "_randomchoiced_"+str(rate_random_samples*rate_samples_in_ibs*trainer_weighted.SAMPLE_SIZE)+"_histogram.png")

    
    
    
    #scene.show(callback=get_camera)