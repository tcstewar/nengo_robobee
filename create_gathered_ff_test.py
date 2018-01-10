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

DATA_SET = 'ff_1_8_test'

t_f = 0.5
n_pts = 500

y_star = np.array([[0, 0, 0, 0],
                   [0.5, 0, 0, 0],
                   [0.5, 0, 90, 0],
                   [0.5, 0, -90, 0],
                   [0.5, 0, 45, 0],
                   [0.5, 0, -45, 0]])

n_y_star = len(y_star)

wing_range = np.zeros(8)
angle_range = np.ones(3)*0.8
pos_range = np.zeros(3)
angle_rate_range = np.array([4, 4, 15])
vel_range = np.ones(3)*1.0

x_range = np.hstack((wing_range, angle_range, pos_range, angle_rate_range, vel_range))

y_star_vals = np.repeat(y_star, n_pts, axis=0)
x_vals = np.random.uniform(-x_range, x_range, (n_pts*n_y_star, len(x_range)))
u_vals = np.zeros((n_pts*n_y_star, 4))
u_ff_vals = np.zeros(np.shape(u_vals))

pif = PIFCompensator(gains_file='PIF_Gains_Body_New_Yaw.mat')
bee = robobee.RoboBee(random_wing_bias=False,
                      a_theta=-0.2, b_theta=0.04,
                      new_yaw_control=True)

def control_ode_fun(t, eta, y_star, x, u, bee):
    eta[4:] = np.zeros(3)
    eta = pif.get_control_dynamics(eta, t, y_star, x, eta[:4], bee)
    return eta

integrator_control = scipy.integrate.ode(control_ode_fun).set_integrator('dopri5')
eta_0 = np.zeros(7)
eta_0[:4] = pif.get_u_star(np.array([0, 0, 0, 0]))


for i in range(n_pts*n_y_star):
    print('{0}/{1}\r'.format(i, n_pts*n_y_star), end='')
    x = x_vals[i,:]
    integrator_control.set_initial_value(eta_0, 0)
    u = np.zeros(4)
    integrator_control.set_f_params(y_star_vals[i,:], x, u, bee)
    eta = integrator_control.integrate(t_f)
    u = eta[:4]
    u_vals[i,:] = u
    u_ff_vals[i,:] = pif.get_u_star(y_star_vals[i,:])

x_vals_body = bee.world_state_to_body(x_vals)

delta_u_vals = u_vals - u_ff_vals

mean_x = np.mean(x_vals, axis=0)
mean_x_body = np.mean(x_vals_body, axis=0)
mean_delta_u = np.mean(delta_u_vals, axis=0)
mean_y_star = np.mean(y_star_vals, axis=0)

std_x = np.std(x_vals, axis=0)
std_x_body = np.std(x_vals_body, axis=0)
std_delta_u = np.std(delta_u_vals, axis=0)
std_y_star = np.std(y_star_vals, axis=0)

norm_x = (x_vals - mean_x) / std_x
norm_x_body = (x_vals_body - mean_x_body) / std_x_body
norm_delta_u = (delta_u_vals - mean_delta_u) / std_delta_u
norm_y_star = (y_star_vals - mean_y_star) / std_y_star

np.savez('gather-{0}'.format(DATA_SET),
         mean_x=mean_x,
         mean_x_body=mean_x_body,
         mean_delta_u=mean_delta_u,
         mean_y_star=mean_y_star,
         std_x=std_x,
         std_x_body=std_x_body,
         std_delta_u=std_delta_u,
         std_y_star=std_y_star,
         all_delta_u=delta_u_vals,
         all_x=x_vals,
         all_x_body=x_vals_body,
         all_y_star=y_star_vals,
         norm_x=norm_x,
         norm_x_body=norm_x_body,
         norm_delta_u=norm_delta_u,
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