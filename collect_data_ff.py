import pytry
import numpy as np
import nengo
import gather_bee
import neuron_bee_ff
from nengo_bee import NengoBee
import os
from scipy import io
from multiprocessing import Pool
from functools import partial
import datetime

# Do longitudinal gain scheduled flight

if __name__ == "__main__":
    bee_model = neuron_bee_ff.GatherDataTrial()

    folder = 'gather_ff_1_8'

    if not os.path.exists(folder):
        os.makedirs(folder)

    att_max = 0.4
    att_rate_max = 6
    vel_var = 0.5
    t_max = 0.5

    n_trials = 3

    # vel_choices = np.arange(0, vel_max + 0.1, 0.1)
    # angle_choices = np.arange(-angle_max, angle_max + 90, 90)
    # turn_rate_choices = np.arange(0, turn_rate_max + 90, 90)
    vel_choices = [0.5]
    angle_choices = [90]
    turn_rate_choices = [0]
    vel_prob = [1.0]
    angle_prob = [1.0]
    turn_rate_prob = [1.0]
    # vel_choices = [0.5]
    # angle_choices = [-90, -60, -30, 0, 30, 60, 90]
    # turn_rate_choices = [0]
    # vel_prob = [1.0]
    # angle_prob = [0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2]
    # turn_rate_prob = [1.0]

    vels = np.random.choice(vel_choices, n_trials, p=vel_prob)
    angles = np.random.choice(angle_choices, n_trials, p=angle_prob)
    turn_rates = np.random.choice(turn_rate_choices, n_trials, p=turn_rate_prob)

    for i in range(n_trials):
        print('Working on Iteration {0}/{1}'.format(i+1, n_trials))
        vel = vels[i]
        turn_rate = turn_rates[i]
        angle = angles[i]

        bee_model.run(data_dir=folder,
                      pose_var=att_max,
                      dpose_var=att_rate_max,
                      vel_var=vel_var,
                      velocity=vel,
                      # velocity=0,
                      angle=angle,
                      turn_rate=turn_rate,
                      use_pif=True,
                      adapt=False,
                      wing_bias=False,
                      T=t_max,
                      # phi_0=np.random.uniform(-np.pi, np.pi, 1)[0],
                      data_format='npz',
                      seed=np.random.randint(10000),
                      verbose=False,
                      ctrl_filename='gather-ff_1_8.npz',
                      init_with_y_star=True,
                      dt=1e-3)