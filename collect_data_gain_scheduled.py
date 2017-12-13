import pytry
import numpy as np
import nengo
import gather_bee
import neuron_bee
from nengo_bee import NengoBee
import os
from scipy import io
from multiprocessing import Pool
from functools import partial
import datetime

def run_model(i, n_trials, folder, att_max, att_rate_max, vel_max, angle_max, turn_rate_max, bee_model, t_max):
    print('Working on Iteration {0}/{1}'.format(i+1, n_trials))
    bee_model.run(data_dir=folder,
                  data_filename='GatherDataTrial#{0}#{1}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), i),
                  pose_var=att_max,
                  dpose_var=att_rate_max,
                  velocity=np.random.uniform(0, vel_max),
                  # velocity=0,
                  angle=np.random.uniform(-angle_max, angle_max),
                  turn_rate=np.random.uniform(0, turn_rate_max),
                  use_pif=True,
                  adapt=False,
                  wing_bias=False,
                  T=t_max,
                  # phi_0=np.random.uniform(-np.pi, np.pi, 1)[0],
                  data_format='npz',
                  seed=i,
                  verbose=False,
                  ctrl_filename='gather-gain_scheduled_12_11.npz')

# Do longitudinal gain scheduled flight

if __name__ == "__main__":
    bee_model = neuron_bee.GatherDataTrial()

    folder = 'gather_gain_scheduled_12_13'

    if not os.path.exists(folder):
        os.makedirs(folder)

    FIXED_ANGLE=True

    att_max = 0.4
    att_rate_max = 12
    vel_max = 0
    angle_max = 0
    turn_rate_max = 0
    t_max = 2.0

    n_trials = 10
    #
    # traj_data = NengoBee.get_initial_set_point([0, 0, 0, 0])
    # u_0 = traj_data['u'][0]
    # u_a_mean = u_0[0]

    # iters = np.arange(n_trials)
    # p = Pool(16)
    # func = partial(run_model,
    #                n_trials=n_trials,
    #                folder=folder,
    #                att_max=att_max,
    #                att_rate_max=att_rate_max,
    #                vel_max=vel_max,
    #                angle_max=angle_max,
    #                turn_rate_max=turn_rate_max,
    #                bee_model=bee_model,
    #                t_max=t_max)
    # results = p.map(func, iters)

    for i in range(n_trials):
        print('Working on Iteration {0}/{1}'.format(i+1, n_trials))
        if FIXED_ANGLE:
            angle = angle_max
        else:
            angle = np.random.uniform(-angle_max, angle_max)
        bee_model.run(data_dir=folder,
                      pose_var=att_max,
                      dpose_var=att_rate_max,
                      velocity=np.random.uniform(0, vel_max),
                      # velocity=0,
                      angle=angle,
                      turn_rate=np.random.uniform(0, turn_rate_max),
                      use_pif=True,
                      adapt=False,
                      wing_bias=False,
                      T=t_max,
                      # phi_0=np.random.uniform(-np.pi, np.pi, 1)[0],
                      data_format='npz',
                      seed=np.random.randint(10000),
                      verbose=False,
                      ctrl_filename='gather-gain_scheduled_12_11.npz')