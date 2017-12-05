import pytry
import numpy as np
import nengo
import gather_bee
import neuron_bee
import os
from scipy import io

# Do longitudinal gain scheduled flight

bee_model = neuron_bee.GatherDataTrial()

folder = 'gather_gain_scheduled_12_1'

if not os.path.exists(folder):
    os.makedirs(folder)

att_max = 1.0
att_rate_max = 30
vel_max = 0.5

n_trials = 10

u_a_mean = 1.301081E2

for i in range(n_trials):
    bee_model.run(data_dir=folder,
                  pose_var=att_max,
                  dpose_var=att_rate_max,
                  # velocity=np.random.uniform(0, vel_max),
                  velocity=0,
                  use_pif=True,
                  adapt=False,
                  wing_bias=False,
                  T=1.0,
                  # phi_0=np.random.uniform(-np.pi, np.pi, 1)[0],
                  data_format='npz',
                  seed=i)