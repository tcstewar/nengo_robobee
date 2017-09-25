import pytry
import numpy as np
import nengo
import gather_bee
import os
import glob

bee_model = gather_bee.GatherDataTrial()

folder = 'gather_pif_update'

if not os.path.exists(folder):
    os.makedirs(folder)

# Clean up old trials
files = glob.glob(os.path.join(folder, '*.txt'))
for f in files:
    os.remove(f)

att_max = 0.5
att_rate_max = 30
vel_max = 0.5
vel_int_max = 0

n_trials = 5000

u_a_mean = 1.301081E2

for i in range(n_trials):
    # Create random points in augmented state space
    # att = [theta, psi]
    # att_rate = [theta_dot, psi_dot]
    # vel = [v_x, v_y, v_z]
    # u = [u_a, u_p, u_r]
    # vel_int = time integral of [v_x, v_y, v_z]
    att = np.random.uniform(-att_max, att_max, size=3)
    # att *= 0
    pos = np.zeros(3)
    att_rate = np.random.uniform(-att_rate_max, att_rate_max, size=3)
    att_rate[0] *= 0                # No yaw rate
    # att_rate *= 0
    vel = np.random.uniform(-vel_max, vel_max, size=3)
    u_a = np.random.uniform(u_a_mean - 0, u_a_mean + 0)
    u_p = np.random.uniform(-40, 40)
    u_r = np.random.uniform(-5, 5)
    u = np.array([u_a, u_p, 0, u_r])
    # u = np.array([u_a_mean, 0, 0, 0])
    x_wings = np.zeros(8)

    x = np.concatenate([x_wings, att, pos, att_rate, vel])

    bee_model.run(data_dir=folder, x=x, u=u, verbose=False, seed=np.random.randint(n_trials * 100))