import time
import math

import pandas as pd
import numpy as np
from tqdm import trange

from it import util
from it.testing.tester import Tester


class EnviroTester(Tester):


    def start_restricted_test(self, environment, points_to_test, np_env_normals):

        data_frame = pd.DataFrame(columns=['point_x', 'point_y', 'point_z','point_nx', 'point_ny', 'point_nz',
                                           'diff_ns' , 'best_score', 'missings', 'best_angle',
                                           'best_orientation', 'calculation_time'])


        #for testing_point in points_to_test:
        for i in trange(points_to_test.shape[0], desc='Testing on environment'):
            testing_point = points_to_test[i]
            env_normal = np_env_normals[i]

            start = time.time()  # timing execution
            # TODO only work with one affordance
            normals_angle = util.angle_between(self.envs_normals[0], env_normal)

            if normals_angle > math.pi/3:
                orientation = math.nan
                angle = math.nan
                score = math.nan
                missing = math.nan
            else:
                analyzer = self.get_analyzer(environment, testing_point)
                angle_with_best_score = analyzer.best_angle_by_distance_by_affordance()

                # TODO permit work with multiple affordances
                first_affordance_scores = angle_with_best_score[0]
                orientation = int(first_affordance_scores[0])
                angle = first_affordance_scores[1]
                score = first_affordance_scores[2]
                missing = int(first_affordance_scores[3])

            end = time.time()  # timing execution
            calculation_time = end - start

            data_frame.loc[len(data_frame)] = [testing_point[0], testing_point[1], testing_point[2],
                                               env_normal[0], env_normal[1], env_normal[2],
                                               normals_angle, score, missing, angle,
                                               orientation, calculation_time]

        return data_frame


    def start_full_test(self, environment, points_to_test, np_env_normals):
        full_data_frame = pd.DataFrame(columns=['interaction',
                                                'point_x', 'point_y', 'point_z',
                                                'point_nx', 'point_ny', 'point_nz', 'diff_ns',
                                                'score', 'missings',
                                                'angle', 'orientation'])

        # for testing_point in points_to_test:
        for i in trange(points_to_test.shape[0]):
            testing_point = points_to_test[i]
            env_normal = np_env_normals[i]

            analyzer = None
            for idx_aff in range(len(self.affordances)):
                affordance = self.affordances[idx_aff][0] # it only get the interaction not the OBJECT's NAME
                affordance_env_normal = self.envs_normals[idx_aff]
                normals_angle = util.angle_between(affordance_env_normal, env_normal)

                if normals_angle > math.pi / 3:
                    score = math.nan
                    missing = math.nan
                    angle = math.nan
                    orientation = math.nan
                    full_data_frame.loc[len(full_data_frame)] = [affordance,
                                                             testing_point[0], testing_point[1], testing_point[2],
                                                             env_normal[0], env_normal[1], env_normal[2], normals_angle,
                                                             score, missing,
                                                             angle, orientation]
                else:
                    if analyzer is None:
                        analyzer = self.get_analyzer(environment, testing_point)
                        analyzer.measure_scores()
                    for orientation in range(self.num_orientations):
                        full_data_frame.loc[len(full_data_frame)] = [affordance,
                                                             testing_point[0], testing_point[1], testing_point[2],
                                                             env_normal[0], env_normal[1], env_normal[2],
                                                             normals_angle,
                                                             analyzer.results.distances_summary[idx_aff][orientation],
                                                             analyzer.results.missed[idx_aff][orientation],
                                                             (2 * math.pi / analyzer.num_orientations) * orientation,
                                                             orientation]





        return full_data_frame