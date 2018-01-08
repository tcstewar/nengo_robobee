# Run a test to see how well the SNN can approximate the feedforward control input from the PIF
import pytry
import numpy as np
import neuron_bee
import nengo_bee
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import io
from itertools import product
from pif_control import PIFControl
from nengo_bee import NengoBee

snn_trial = neuron_bee.GatherDataTrial()

t_max = 0.1
bee = NengoBee()
pif = PIFControl(bee.bee)

vel_pts = [0, 0.5, 1.0]
turn_rate_pts = [0]
angle_pts = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
sideslip_pts = [0]

# get the feedforward control input at different points
y_star_pts = list(product(vel_pts, turn_rate_pts, angle_pts, sideslip_pts))
y_star_pts = np.array(y_star_pts)
u_star_pts = np.zeros(y_star_pts.shape)

u_star_target = [pif.controller.get_u_star(y_star) for y_star in y_star_pts]
u_star_target = np.array(u_star_target)

for i in range(len(y_star_pts)):
    VEL_TARGET = y_star_pts[i,0]
    TURN_RATE = y_star_pts[i,1]
    CLIMB_ANGLE = y_star_pts[i,2]
    SIDESLIP_ANGLE = y_star_pts[i,3]

    data = snn_trial.run(velocity=VEL_TARGET,
                         angle=CLIMB_ANGLE,
                         ctrl_filename='gather-gain_scheduled_12_29.npz',
                         init_with_y_star=True,
                         wing_bias=False,
                         use_pif=True,
                         adapt=False,
                         T=t_max,
                         dt=1E-4,
                         verbose=False)
    u = data['ens']
    u_final = np.mean(u[-200:, :], axis=0)
    u_star_pts[i,:] = u_final

io.savemat('test_telluride_stuff.mat', {'y_star': y_star_pts,
                        'u_star': u_star_pts,
                        'u_star_target': u_star_target})

# sns.set()
# fig, axes = plt.subplots(2,2)
# axes = axes.flatten()
# axis_names = ['$u_a$', '$u_p$', '$u_y$', '$u_r$']
# for i in range(4):
#     axes[i].plot(u[:,i])
#     axes[i].set_ylabel(axis_names[i])
#
# plt.show()

#
# sns.set()
# plt.figure()
# plt.plot(t_log, x_body[:, bee.idx_body_att])
# plt.ylabel('Angle (rad)')
# plt.xlabel('t (s)')
# plt.title('Euler Angles')
# plt.legend(['$\phi$', '$\\theta$', '$\psi$'])
#
# plt.figure()
# plt.plot(t_log, x_body[:, bee.idx_body_pos])
# plt.ylabel('Position (m)')
# plt.xlabel('t (s)')
# plt.title('Position')
# plt.legend(['$x$', '$y$', '$z$'])
#
# plt.figure()
# plt.plot(t_log, x_body[:, bee.idx_body_att_rate])
# plt.plot(t_log, x_star_log[:, bee.idx_body_att_rate], 'k--')
# plt.ylabel('Angular Rate (rad/s)')
# plt.xlabel('t (s)')
# plt.title('Angular Rate')
# plt.legend(['$d\phi/dt$', '$d\\theta/dt$', '$d\psi/dt$'])
#
# plt.figure()
# plt.plot(t_log, x_body[:, bee.idx_body_vel])
# plt.plot(t_log, x_star_log[:, bee.idx_body_vel], 'k--')
# plt.ylabel('Velocity (m/s)')
# plt.xlabel('t (s)')
# plt.title('Velocity')
# plt.legend(['$v_x$', '$v_y$', '$v_z$'])
#
# plt.figure()
# plt.plot(t_log, u)
# # plt.plot(u_pif[:, [0,1,3]], '--')
# plt.ylabel('Control Input')
# plt.title('Control Inputs')
# plt.legend(['$u_a$', '$u_p$', '$u_y$', '$u_r$'])
#
# f, axarr = plt.subplots(3, sharex=True)
# axarr[0].plot(t_log, adapt_x)
# axarr[0].set_title('Adapt Terms')
# axarr[0].set_ylabel('Adapt x')
# axarr[1].plot(t_log, adapt_y)
# axarr[1].set_ylabel('Adapt y')
# axarr[2].plot(t_log, adapt_z)
# axarr[2].set_ylabel('Adapt z')
# axarr[2].set_xlabel('t (s)')
#
# plt.figure()
# plt.plot(t_log, ens)
# plt.plot(t_log, u_pif, '--')
# plt.ylabel('Control Input')
# plt.title('$u_0$')
# plt.legend(['$u_a$', '$u_p$', '$u_y$', '$u_r$', '$u_a^{PIF}$', '$u_p^{PIF}$', '$u_y^{PIF}$', '$u_r^{PIF}$'])
#
# # f, axarr = plt.subplots(4, sharex=True)
# # axarr[0].plot(t_log, ens[:,0] - u_pif[:,0])
# # axarr[0].set_title('$u_0$ Error')
# # axarr[0].set_ylabel('$\Delta u_a$')
# # axarr[1].plot(t_log, ens[:,1] - u_pif[:,1])
# # axarr[1].set_ylabel('$\Delta u_p$')
# # axarr[2].plot(t_log, ens[:,2] - u_pif[:,2])
# # axarr[2].set_ylabel('$\Delta u_y$')
# # axarr[3].plot(t_log, ens[:,3] - u_pif[:,3])
# # axarr[3].set_ylabel('$\Delta u_r$')
# # axarr[3].set_xlabel('t (s)')
#
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.plot(x_body[:, bee.idx_body_pos[0]],
# #         x_body[:, bee.idx_body_pos[1]],
# #         x_body[:, bee.idx_body_pos[2]])
#
# plt.show()