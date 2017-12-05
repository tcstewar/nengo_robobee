import pytry
import numpy as np
import neuron_bee
import nengo_bee
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import io

bee_trial = neuron_bee.GatherDataTrial()

t_max = 6.0

data = bee_trial.run(use_pif=False,
                     adapt=True,
                     ctrl_filename='gather-gain_scheduled_12_1.npz',
                     velocity=0.4,
                     pose_var=0.5,
                     dpose_var=20,
                     # pose_var=0,
                     # dpose_var=0,
                     use_learning_display=False,
                     T=t_max,
                     n_neurons=500,
                     seed=10,
                     wing_bias=False,
                     v_wind=0,
                     phi_0=0,
                     actuator_failure=False,
                     adapt_Kp=0.6,
                     adapt_Kd=18)

bee = nengo_bee.NengoBee().bee

# print(data.keys())

x_world = data['x']
x_unfilt = data['x_unfilt']
u = data['u']
u_pif = data['pif_u']
# u_dot = data['pif_u_dot']
ens = data['ens']
adapt_x = data['adapt_x']
adapt_y = data['adapt_y']
adapt_z = data['adapt_z']

x_body = bee.world_state_to_body(x_world)

t_log = np.linspace(0, t_max, len(u))

io.savemat('saved_data/snn_debug_trial.mat',
           {'x_log': x_world,
            'u_log': u,
            't': t_log,
            'ens': ens,
            'u_pif': u_pif,
            'adapt_x': adapt_x,
            'adapt_y': adapt_y,
            'adapt_z': adapt_z,
            'x_unfilt': x_unfilt})
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
plt.ylabel('Angular Rate (rad/s)')
plt.xlabel('t (s)')
plt.title('Angular Rate')
plt.legend(['$d\phi/dt$', '$d\\theta/dt$', '$d\psi/dt$'])

plt.figure()
plt.plot(t_log, x_body[:, bee.idx_body_vel])
plt.ylabel('Velocity (m/s)')
plt.xlabel('t (s)')
plt.title('Velocity')
plt.legend(['$v_x$', '$v_y$', '$v_z$'])

plt.figure()
plt.plot(t_log, u[:, [0, 1, 3]])
# plt.plot(u_pif[:, [0,1,3]], '--')
plt.ylabel('Control Input')
plt.title('Control Inputs')
plt.legend(['$u_a$', '$u_p$', '$u_r$'])

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
plt.plot(t_log, ens[:, [0, 1, 3]])
plt.plot(t_log, u_pif[:, [0, 1, 3]], '--')
plt.ylabel('Control Input')
plt.title('$u_0$')
plt.legend(['$u_a$', '$u_p$', '$u_r$', '$u_a^{PIF}$', '$u_p^{PIF}$', '$u_r^{PIF}$'])

f, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(t_log, ens[:,0] - u_pif[:,0])
axarr[0].set_title('$u_0$ Error')
axarr[0].set_ylabel('$\Delta u_a$')
axarr[1].plot(t_log, ens[:,1] - u_pif[:,1])
axarr[1].set_ylabel('$\Delta u_p$')
axarr[2].plot(t_log, ens[:,3] - u_pif[:,3])
axarr[2].set_ylabel('$\Delta u_r$')
axarr[2].set_xlabel('t (s)')

plt.show()