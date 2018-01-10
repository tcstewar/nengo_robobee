import pytry
import numpy as np
from neuron_bee_ff import GatherDataTrial
import nengo_bee
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import io

bee_trial = GatherDataTrial()

t_max = 0.5

USE_SNN = True
ADAPT = False
VEL_TARGET = 0.5
CLIMB_ANGLE = 0
TURN_RATE = 0

SAVE_FILE_NAME = 'saved_data/{0}_debug_trial.mat'.format(('snn' if USE_SNN else 'pif'))
# SAVE_FILE_NAME = 'saved_data/{0}_longitudinal_wind_05.mat'.format(('snn' if USE_SNN else 'pif'))
# SAVE_FILE_NAME = 'saved_data/{0}_turn_{1}_vel_04_wind_02.mat'.format(('snn' if USE_SNN else 'pif'), TURN_RATE)

print('Running: t_max={0:3.2f}, use_snn={1}, v={2:3.2f}, gamma={3:3.2f}, xi={4:3.2f}'.format(t_max, USE_SNN, VEL_TARGET, CLIMB_ANGLE, TURN_RATE))

data = bee_trial.run(use_pif=(not USE_SNN),
                     adapt=ADAPT,
                     ctrl_filename='gather-ff_1_8_test.npz',
                     velocity=VEL_TARGET,
                     angle=CLIMB_ANGLE,
                     turn_rate=TURN_RATE,
                     pose_var=0.4,
                     dpose_var=6,
                     vel_var=0.4,
                     # pose_var=0,
                     # dpose_var=0,
                     use_learning_display=False,
                     T=t_max,
                     n_neurons=500,
                     n_adapt_neurons=100,
                     seed=10,
                     wing_bias=False,
                     v_wind=0,
                     phi_0=0,
                     actuator_failure=False,
                     adapt_Kp=1.0,
                     adapt_Kd=1.0, # adapt_Kd=0.6
                     fancy_flight=False,
                     init_with_y_star=True)

bee = nengo_bee.NengoBee().bee

# print(data.keys())

x_world = data['x']
x_unfilt = data['x_unfilt']
u = data['u']
u_pif = data['pif_u']
# u_dot = data['pif_u_dot']
# ens = data['ens']
adapt_x = data['adapt_x']
adapt_y = data['adapt_y']
adapt_z = data['adapt_z']
x_star_log = data['x_star']
y_star_log = data['y_star']

x_body = bee.world_state_to_body(x_world)

t_log = np.linspace(0, t_max, len(u))

io.savemat(SAVE_FILE_NAME,
           {'x_filt': x_world,
            'u_log': u,
            't': t_log,
            # 'ens': ens,
            'u_pif': u_pif,
            'adapt_x': adapt_x,
            'adapt_y': adapt_y,
            'adapt_z': adapt_z,
            'x_log': x_unfilt,
            'x_star_log': x_star_log[:,-12:],
            'y_star_log': y_star_log})
sns.set()
plt.figure()
plt.plot(t_log, x_body[:, bee.idx_body_att])
plt.ylabel('Angle (rad)')
plt.xlabel('t (s)')
plt.title('Euler Angles')
plt.legend(['$\phi$', '$\\theta$', '$\psi$'])

plt.figure()
plt.plot(t_log, x_body[:, bee.idx_body_pos])
plt.ylabel('Position (m)')
plt.xlabel('t (s)')
plt.title('Position')
plt.legend(['$x$', '$y$', '$z$'])

plt.figure()
plt.plot(t_log, x_body[:, bee.idx_body_att_rate])
plt.plot(t_log, x_star_log[:, bee.idx_body_att_rate], 'k--')
plt.ylabel('Angular Rate (rad/s)')
plt.xlabel('t (s)')
plt.title('Angular Rate')
plt.legend(['$d\phi/dt$', '$d\\theta/dt$', '$d\psi/dt$'])

plt.figure()
plt.plot(t_log, x_body[:, bee.idx_body_vel])
plt.plot(t_log, x_star_log[:, bee.idx_body_vel], 'k--')
plt.ylabel('Velocity (m/s)')
plt.xlabel('t (s)')
plt.title('Velocity')
plt.legend(['$v_x$', '$v_y$', '$v_z$'])

# plt.figure()
# plt.plot(t_log, u)
# # plt.plot(u_pif[:, [0,1,3]], '--')
# plt.ylabel('Control Input')
# plt.title('Control Inputs')
# plt.legend(['$u_a$', '$u_p$', '$u_y$', '$u_r$'])

f, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(t_log, adapt_x)
axarr[0].set_title('Adapt Terms')
axarr[0].set_ylabel('Adapt x')
axarr[1].plot(t_log, adapt_y)
axarr[1].set_ylabel('Adapt y')
axarr[2].plot(t_log, adapt_z)
axarr[2].set_ylabel('Adapt z')
axarr[2].set_xlabel('t (s)')

plt.figure()
plt.plot(t_log, u)
plt.plot(t_log, u_pif, '--')
plt.ylabel('Control Input')
plt.title('$u_0$')
plt.legend(['$u_a$', '$u_p$', '$u_y$', '$u_r$', '$u_a^{PIF}$', '$u_p^{PIF}$', '$u_y^{PIF}$', '$u_r^{PIF}$'])

# f, axarr = plt.subplots(4, sharex=True)
# axarr[0].plot(t_log, ens[:,0] - u_pif[:,0])
# axarr[0].set_title('$u_0$ Error')
# axarr[0].set_ylabel('$\Delta u_a$')
# axarr[1].plot(t_log, ens[:,1] - u_pif[:,1])
# axarr[1].set_ylabel('$\Delta u_p$')
# axarr[2].plot(t_log, ens[:,2] - u_pif[:,2])
# axarr[2].set_ylabel('$\Delta u_y$')
# axarr[3].plot(t_log, ens[:,3] - u_pif[:,3])
# axarr[3].set_ylabel('$\Delta u_r$')
# axarr[3].set_xlabel('t (s)')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x_body[:, bee.idx_body_pos[0]],
#         x_body[:, bee.idx_body_pos[1]],
#         x_body[:, bee.idx_body_pos[2]])

plt.show()