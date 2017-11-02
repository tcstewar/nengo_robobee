import pytry
import numpy as np
import nengo
import gather_bee
import neuron_bee
import os
import glob

bee_model = gather_bee.GatherDataTrial()
# bee_model = neuron_bee.GatherDataTrial()

folder = 'gather_pif_update'

if not os.path.exists(folder):
    os.makedirs(folder)

# Clean up old trials
files = glob.glob(os.path.join(folder, '*.*'))
for f in files:
    os.remove(f)

att_max = 0.8
att_rate_max = 50
# vel_max = 0.5
# vel_int_max = 0

n_trials = 20
t_max = 1.0

# u_a_mean = 1.301081E2

for i in range(n_trials):
    # Create random points in augmented state space
    # att = [theta, psi]
    # att_rate = [theta_dot, psi_dot]
    # vel = [v_x, v_y, v_z]
    # u = [u_a, u_p, u_r]
    # vel_int = time integral of [v_x, v_y, v_z]

    print('Trial {0}/{1}'.format(i+1, n_trials))
    bee_model.run(data_dir=folder,
                  att_var=att_max,
                  att_rate_var=att_rate_max,
                  # pose_var=att_max,
                  # dpose_var=att_rate_max,
                  T=t_max,
                  verbose=False,
                  data_format='npz',
                  seed=np.random.randint(n_trials * 100))