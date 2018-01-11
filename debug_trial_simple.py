import pytry
import numpy as np
from neuron_bee_simple import GatherDataTrial
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import io
import sys
sys.path.append(r'..\PyBee3D\PyBee3D')
import robobee

bee_trial = GatherDataTrial()

t_max = 3.0

SAVE_FILE_NAME = 'saved_data/simple_debug_trial.mat'

USE_SNN = True
VEL_TARGET = 0.3
CLIMB_ANGLE = 0
TURN_RATE = 0

print('Using {0}: t_max={1:3.2f}, v={2:3.2f}, gamma={3:3.2f}, xi={4:3.2f}'.format(
    ('SNN' if USE_SNN else 'PIF'), t_max, VEL_TARGET, CLIMB_ANGLE, TURN_RATE))

data = bee_trial.run(T=t_max,
                     seed=10,
                     use_pif=(not USE_SNN),
                     vel_target=VEL_TARGET,
                     climb_angle=CLIMB_ANGLE,
                     turn_rate=TURN_RATE)

bee = robobee.RoboBee(random_wing_bias=False)

x_world = data['x']
x_body = data['x_body']
u = data['u']
u_pif = data['u_pif']
y_star_log = data['y_star']
x_star_log = data['x_star']
u_star_log = data['u_star']

t_log = np.linspace(0, t_max, len(u))

io.savemat(SAVE_FILE_NAME,
           {'x_log': x_world,
            'x_body': x_body,
            'u_log': u,
            'u_pif': u_pif,
            't': t_log,
            'y_star_log': y_star_log,
            'x_star_log': x_star_log,
            'u_star_log': u_star_log})

sns.set()
prop_cycle = mpl.rcParams['axes.prop_cycle']

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

mpl.rcParams['axes.prop_cycle'] = prop_cycle[0:3]
plt.figure()
plt.plot(t_log, x_body[:, bee.idx_body_vel])
plt.plot(t_log, x_star_log[:, bee.idx_body_vel], ':')
plt.ylabel('Velocity (m/s)')
plt.xlabel('t (s)')
plt.title('Velocity')
plt.legend(['$v_x$', '$v_y$', '$v_z$', '$v_x^*$', '$v_y^*$', '$v_z^*$'])


mpl.rcParams['axes.prop_cycle'] = prop_cycle[0:4]
plt.figure()
plt.plot(t_log, u)
plt.plot(t_log, u_star_log, ':')
plt.ylabel('Control Input')
plt.xlabel('$t$ (s)')
plt.title('Control Inputs')
plt.legend(['$u_a$', '$u_p$', '$u_y$', '$u_r$', '$u_a^*$', '$u_p^*$', '$u_y^*$', '$u_r^*$'])
mpl.rcParams['axes.prop_cycle'] = prop_cycle

plt.show()