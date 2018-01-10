 # TODO: Write this file. Should analyze the gathered data, create the gather-gain_scheduled file,
 # and plot time series plots for all of the trials

import numpy as np
import matplotlib.pyplot as plt
import pytry
import pandas as pd
import seaborn as sns
from nengo_bee import NengoBee

DATA_SET = 'ff_1_8'

data = pytry.read('gather_{0}'.format(DATA_SET))

bee = NengoBee()

all_x = np.vstack([d['x'][:] for d in data])
all_x_body = bee.bee.world_state_to_body(all_x)
all_u = np.vstack([d['u'][:] for d in data])
all_u_ff = np.vstack([d['pif_ff'][:] for d in data])
all_y_star = np.vstack([d['y_star'][:] for d in data])

all_delta_u = all_u - all_u_ff

mean_x = np.mean(all_x, axis=0)
mean_x_body = np.mean(all_x_body, axis=0)
mean_delta_u = np.mean(all_delta_u, axis=0)
mean_y_star = np.mean(all_y_star, axis=0)

std_x = np.std(all_x, axis=0)
std_x_body = np.std(all_x_body, axis=0)
std_delta_u = np.std(all_delta_u, axis=0)
std_y_star = np.std(all_y_star, axis=0)

print(mean_x_body)
print(mean_delta_u)
print(mean_y_star)

norm_x = (all_x - mean_x) / std_x
norm_x_body = (all_x_body - mean_x_body) / std_x_body
norm_delta_u = (all_delta_u - mean_delta_u) / std_delta_u
norm_y_star = (all_y_star - mean_y_star) / std_y_star

print(std_x)
print(std_x_body)
print(std_delta_u)
print(std_y_star)

np.savez('gather-{0}'.format(DATA_SET),
         mean_x=mean_x,
         mean_x_body=mean_x_body,
         mean_delta_u=mean_delta_u,
         mean_y_star=mean_y_star,
         std_x=std_x,
         std_x_body=std_x_body,
         std_delta_u=std_delta_u,
         std_y_star=std_y_star,
         all_delta_u=all_delta_u,
         all_x=all_x,
         all_x_body=all_x_body,
         all_y_star=all_y_star,
         norm_x=norm_x,
         norm_x_body=norm_x_body,
         norm_delta_u=norm_delta_u,
         norm_y_star=norm_y_star)

# Plot some stuff
df = pd.DataFrame(data)

u_names = ['$u_a$', '$u_p$', '$u_y$', '$u_r$']
n_trials = df.shape[0]

# Convert y_star to something we can use for a groupby operation:
rounded_y_star = [np.rint(x*np.array([10,1,1,1])).astype(int) for x in df['y_star']]
df['y_star'] = [tuple(map(tuple,x)) for x in rounded_y_star]
y_star_groups = df.groupby(['y_star'])

for y_star, group in y_star_groups:
    # fig, axes = plt.subplots(3,1)
    # n_trials = group.shape[0]
    #
    # for row in group['x_unfilt']:
    #     axes[0].plot(row[:, bee.bee.idx_body_att[0]])
    #     axes[1].plot(row[:, bee.bee.idx_body_att[1]])
    #     axes[2].plot(row[:, bee.bee.idx_body_att[2]])
    # axes[0].set_ylabel('$\phi$')
    # axes[1].set_ylabel('$\\theta$')
    # axes[2].set_ylabel('$\psi$')
    # axes[0].set_title('$y^*$ = {0}'.format(y_star[0]))

    fig, axes = plt.subplots(3,1)
    n_trials = group.shape[0]

    for row in group['x_unfilt']:
        axes[0].plot(row[:, bee.bee.idx_body_vel[0]])
        axes[1].plot(row[:, bee.bee.idx_body_vel[1]])
        axes[2].plot(row[:, bee.bee.idx_body_vel[2]])
    axes[0].set_ylabel('$v_x$')
    axes[1].set_ylabel('$v_y$')
    axes[2].set_ylabel('$v_z$')
    axes[0].set_title('$y^*$ = {0}'.format(y_star[0]))

    fig, axes = plt.subplots(4,1)
    n_trials = group.shape[0]
    for row in group['u']:
        axes[0].plot(row[:, 0])
        axes[1].plot(row[:, 1])
        axes[2].plot(row[:, 2])
        axes[3].plot(row[:, 3])
    axes[0].set_ylabel('$u_a$')
    axes[1].set_ylabel('$u_p$')
    axes[2].set_ylabel('$u_y$')
    axes[3].set_ylabel('$u_r$')
    axes[0].set_title('$y^*$ = {0}'.format(y_star[0]))


for i in range(4):
    plt.figure()
    if std_delta_u[i] > 0:
        sns.distplot(all_delta_u[:,i], kde=0)
    plt.title('u[%d] %s ($\mu$=%1.4f, $\sigma$=%1.4f)' % (i, u_names[i], mean_delta_u[i], std_delta_u[i]))

plt.show()