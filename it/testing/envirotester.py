import time
import math

import pandas as pd
import numpy as np
from tqdm import trange

from it import util
from it.testing.tester import Tester


class EnviroTester(Tester):


    def start(self, environment, points_to_test, np_env_normals):

        data_frame = pd.DataFrame(columns=['point_x', 'point_y', 'point_z','point_nx', 'point_ny', 'point_nz',
                                           'diff_ns' , 'best_score', 'missings', 'best_angle',
                                           'best_orientation', 'calculation_time'])


        #for testing_point in points_to_test:
        for i in trange(points_to_test.shape[0], desc='Testing on environment'):
            testing_point = points_to_test[i]
            env_normal = np_env_normals[i]

            start = time.time()  # timing execution

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