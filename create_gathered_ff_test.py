 # Generate the data for the SNN directly from the controller without simulating the RoboBee

import numpy as np
import matplotlib.pyplot as plt
import pytry
import pandas as pd
import seaborn as sns
from nengo_bee import NengoBee
import sys
import scipy
sys.path.append(r'..\PyBee3D\PyBee3D')
from controllers.pif.pif_compensator import PIFCompensator
import robobee
from itertools import product

DATA_SET = 'ff_1_8_test'

vel_range = np.array([0.5, 1.0])
xi_dot_range = np.array([0, 90, 180])
gamma_range = np.arange(-90, 105, 15)
beta_range = [0]

y_star_vals = list(product(vel_range, xi_dot_range, gamma_range, beta_range))
y_star_vals.insert(0, (0, 0, 0, 0))

n_y_star = len(y_star_vals)

u_star_vals = np.zeros((n_y_star, 4))
x_star_vals = np.zeros((n_y_star, 20))
ss_gains = np.zeros((n_y_star, 4*8))

pif = PIFCompensator(gains_file='PIF_Gains_Body_New_Yaw.mat')
bee = robobee.RoboBee(random_wing_bias=False,
                      a_theta=-0.2, b_theta=0.04,
                      new_yaw_control=True)

y_star_vals = np.array(y_star_vals)

for i in range(n_y_star):
    K = pif.get_gains(y_star_vals[i, :])
    K_scaled = np.diag(pif.scale_u_dot) @ K @ np.diag(pif.scale_chi)
    K_1 = K_scaled[:,0:8]
    K_2 = K_scaled[:,8:12]
    K_ss = -np.linalg.inv(K_2) @ K_1

    ss_gains[i,:] = K_ss.flatten()
    u_star_vals[i, :] = pif.get_u_star(y_star_vals[i, :])
    x_star_vals[i,:] = pif.get_x_star(y_star_vals[i, :])

mean_gains = np.mean(ss_gains, axis=0)
mean_u_star = np.mean(u_star_vals, axis=0)
mean_x_star = np.mean(x_star_vals, axis=0)
mean_y_star = np.mean(y_star_vals, axis=0)

std_gains = np.std(ss_gains, axis=0)
std_u_star = np.std(u_star_vals, axis=0)
std_x_star = np.std(x_star_vals, axis=0)
std_y_star = np.std(y_star_vals, axis=0)

norm_gains = (ss_gains - mean_gains) / std_gains
norm_u_star = (u_star_vals - mean_u_star) / std_u_star
norm_x_star = (x_star_vals - mean_x_star) / std_x_star
norm_y_star = (y_star_vals - mean_y_star) / std_y_star

np.savez('gather-{0}'.format(DATA_SET),
         mean_gains=mean_gains,
         mean_u_star=mean_u_star,
         mean_x_star=mean_x_star,
         mean_y_star=mean_y_star,
         std_gains=std_gains,
         std_u_star=std_u_star,
         std_x_star=std_x_star,
         std_y_star=std_y_star,
         ss_gains=ss_gains,
         u_star_vals=u_star_vals,
         x_star_vals=x_star_vals,
         all_y_star=y_star_vals,
         norm_gains=norm_gains,
         norm_u_star=norm_u_star,
         norm_x_star=norm_x_star,
         norm_y_star=norm_y_star)
#
#
#
#
#
#
#
#
#
#
#
#
#
# data = pytry.read('gather_{0}'.format(DATA_SET))
#
# bee = NengoBee()
#
# all_x = np.vstack([d['x'][:] for d in data])
# all_x_body = bee.bee.world_state_to_body(all_x)
# all_u = np.vstack([d['u'][:] for d in data])
# all_u_ff = np.vstack([d['pif_ff'][:] for d in data])
# all_y_star = np.vstack([d['y_star'][:] for d in data])
#
# all_delta_u = all_u - all_u_ff
#
# mean_x = np.mean(all_x, axis=0)
# mean_x_body = np.mean(all_x_body, axis=0)
# mean_delta_u = np.mean(all_delta_u, axis=0)
# mean_y_star = np.mean(all_y_star, axis=0)
#
# std_x = np.std(all_x, axis=0)
# std_x_body = np.std(all_x_body, axis=0)
# std_delta_u = np.std(all_delta_u, axis=0)
# std_y_star = np.std(all_y_star, axis=0)
#
# print(mean_x_body)
# print(mean_delta_u)
# print(mean_y_star)
#
# norm_x = (all_x - mean_x) / std_x
# norm_x_body = (all_x_body - mean_x_body) / std_x_body
# norm_delta_u = (all_delta_u - mean_delta_u) / std_delta_u
# norm_y_star = (all_y_star - mean_y_star) / std_y_star
#
# print(std_x)
# print(std_x_body)
# print(std_delta_u)
# print(std_y_star)
#
# np.savez('gather-{0}'.format(DATA_SET),
#          mean_x=mean_x,
#          mean_x_body=mean_x_body,
#          mean_delta_u=mean_delta_u,
#          mean_y_star=mean_y_star,
#          std_x=std_x,
#          std_x_body=std_x_body,
#          std_delta_u=std_delta_u,
#          std_y_star=std_y_star,
#          all_delta_u=all_delta_u,
#          all_x=all_x,
#          all_x_body=all_x_body,
#          all_y_star=all_y_star,
#          norm_x=norm_x,
#          norm_x_body=norm_x_body,
#          norm_delta_u=norm_delta_u,
#          norm_y_star=norm_y_star)
#
# # Plot some stuff
# df = pd.DataFrame(data)
#
# u_names = ['$u_a$', '$u_p$', '$u_y$', '$u_r$']
# n_trials = df.shape[0]
#
# # Convert y_star to something we can use for a groupby operation:
# rounded_y_star = [np.rint(x*np.array([10,1,1,1])).astype(int) for x in df['y_star']]
# df['y_star'] = [tuple(map(tuple,x)) for x in rounded_y_star]
# y_star_groups = df.groupby(['y_star'])
#
# for y_star, group in y_star_groups:
#     # fig, axes = plt.subplots(3,1)
#     # n_trials = group.shape[0]
#     #
#     # for row in group['x_unfilt']:
#     #     axes[0].plot(row[:, bee.bee.idx_body_att[0]])
#     #     axes[1].plot(row[:, bee.bee.idx_body_att[1]])
#     #     axes[2].plot(row[:, bee.bee.idx_body_att[2]])
#     # axes[0].set_ylabel('$\phi$')
#     # axes[1].set_ylabel('$\\theta$')
#     # axes[2].set_ylabel('$\psi$')
#     # axes[0].set_title('$y^*$ = {0}'.format(y_star[0]))
#
#     fig, axes = plt.subplots(3,1)
#     n_trials = group.shape[0]
#
#     for row in group['x_unfilt']:
#         axes[0].plot(row[:, bee.bee.idx_body_vel[0]])
#         axes[1].plot(row[:, bee.bee.idx_body_vel[1]])
#         axes[2].plot(row[:, bee.bee.idx_body_vel[2]])
#     axes[0].set_ylabel('$v_x$')
#     axes[1].set_ylabel('$v_y$')
#     axes[2].set_ylabel('$v_z$')
#     axes[0].set_title('$y^*$ = {0}'.format(y_star[0]))
#
#     fig, axes = plt.subplots(4,1)
#     n_trials = group.shape[0]
#     for row in group['u']:
#         axes[0].plot(row[:, 0])
#         axes[1].plot(row[:, 1])
#         axes[2].plot(row[:, 2])
#         axes[3].plot(row[:, 3])
#     axes[0].set_ylabel('$u_a$')
#     axes[1].set_ylabel('$u_p$')
#     axes[2].set_ylabel('$u_y$')
#     axes[3].set_ylabel('$u_r$')
#     axes[0].set_title('$y^*$ = {0}'.format(y_star[0]))
#
#
# for i in range(4):
#     plt.figure()
#     if std_delta_u[i] > 0:
#         sns.distplot(all_delta_u[:,i], kde=0)
#     plt.title('u[%d] %s ($\mu$=%1.4f, $\sigma$=%1.4f)' % (i, u_names[i], mean_delta_u[i], std_delta_u[i]))
#
# plt.show()