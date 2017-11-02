import pytry
import numpy as np
import neuron_bee
import nengo_bee
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import io

bee_trial = neuron_bee.GatherDataTrial()

t_max = 1.0

data = bee_trial.run(use_pif=False,
                     adapt=True,
                     pose_var=0.5,
                     dpose_var=20,
                     use_learning_display=False,
                     T=t_max,
                     n_neurons=500,
                     seed=1)

bee = nengo_bee.NengoBee().bee

print(data.keys())

x_world = data['x']
u = data['u']
u_pif = data['pif_u']
# u_dot = data['pif_u_dot']
ens = data['ens']
# learn_y = data['learny']

x_body = bee.world_state_to_body(x_world)

t_log = np.linspace(0, t_max, len(u))

io.savemat('debug_trial.mat', {'x_log': x_world,
                        'u_log': u,
                        't': t_log})
sns.set()
plt.figure()
plt.plot(x_body[:, bee.idx_body_att])
plt.ylabel('Angle (rad)')
plt.title('Euler Angles')
plt.legend(['$\phi$', '$\\theta$', '$\psi$'])

plt.figure()
plt.plot(x_body[:, bee.idx_body_pos])
plt.ylabel('Position (m)')
plt.title('Position')
plt.legend(['$x$', '$y$', '$z$'])

plt.figure()
plt.plot(x_body[:, bee.idx_body_att_rate])
plt.ylabel('Angular Rate (rad/s)')
plt.title('Angular Rate')
plt.legend(['$d\phi/dt$', '$d\\theta/dt$', '$d\psi/dt$'])

plt.figure()
plt.plot(x_body[:, bee.idx_body_vel])
plt.ylabel('Velocity (m/s)')
plt.title('Velocity')
plt.legend(['$\dot{x}$', '$\dot{y}$', '$\dot{z}$'])

plt.figure()
plt.plot(u[:, [0, 1, 3]])
plt.plot(u_pif[:, [0,1,3]], '--')
plt.ylabel('Control Input')
plt.title('Control Inputs')
plt.legend(['$u_a$', '$u_p$', '$u_r$', '$u_a^{pif}$', '$u_p^{pif}$', '$u_r^{pif}$'])
#
# plt.figure()
# plt.plot(u_dot[:, [0, 1, 3]])
# plt.ylabel('$\dot{u}$')
# plt.title('Control Input Rate')
# plt.legend(['$\dot{u}_a$', '$\dot{u}_p$', '$\dot{u}_r$'])
#
# plt.figure()
# plt.plot(ens[:, [0, 1, 3]])
# plt.ylabel('$\hat{\dot{u}}$')
# plt.title('Approx Control Rate')
# plt.legend(['$\dot{u}_a$', '$\dot{u}_p$', '$\dot{u}_r$'])
#
# plt.figure()
# plt.plot(learn_y)
# plt.ylabel('Learned $u_r$')
# plt.title('Control Input')

plt.show()