import sys
sys.path.append('.')
sys.path.append('../PyBee3D_Basic')
import pytry
import numpy as np
import robobee
import scipy
import nengo
from nengo_bee import NengoBee
import seaborn

class GatherDataTrial(pytry.NengoTrial):
    def params(self):
        self.param('velocity', velocity=0.0)
        self.param('angle', angle=0.0)
        self.param('initial pose variability', pose_var=0.0)
        self.param('initial rotation rate variability', dpose_var=0.0)
        self.param('controller filename', ctrl_filename='gather7-hover.npz')
        self.param('time', T=1.0)
        self.param('number of neurons', n_neurons=500)
        self.param('regularization', reg=0.1)
        self.param('radius scaling', radius_scaling=1.0)
        self.param('intercept minimum', low_intercept=-1.0)
        self.param('learning_rate', learning_rate=1e-4)
        self.param('learning adaptation rate', adapt_learn_rate=1e-4)
        self.param('use learning display', use_learning_display=True)
        self.param('apply wing bias', wing_bias=True)
        self.param('adapt Kp scale', adapt_Kp=0.0)
        self.param('adapt Kd scale', adapt_Kd=100.0)
        self.param('adapt Ki scale', adapt_Ki=0.0)

    def model(self, p):

        ctrl = np.load(p.ctrl_filename)

        model = nengo.Network()
        with model:
            pose_offset = np.random.uniform(-p.pose_var, p.pose_var, size=3)
            dpose_offset = np.random.uniform(-p.dpose_var, p.dpose_var, size=3)

            bee = NengoBee(pose_offset=pose_offset, dpose_offset=dpose_offset,
                           random_wing_bias=p.wing_bias)
            
            #keep_x = [9, 15, 17]
            keep_x = [9, 10, 15, 16, 17]
            keep_u = []#1, 3]

            # Use the body frame velocities
            x_vals = ctrl['norm_x_body'][:,keep_x]
            u_vals = ctrl['norm_u'][:,keep_u]

            target = ctrl['all_u']

            pts = np.hstack([x_vals, u_vals])


            D = pts.shape[1]
            ens = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=D,
                                 intercepts=nengo.dists.Uniform(p.low_intercept, 1.0),
                                 neuron_type=nengo.LIF(), radius=np.sqrt(D)*p.radius_scaling)

            u_unfilt = nengo.Node(None, size_in=4)
            u = nengo.Node(None, size_in=4)

            source_body_vel = bee.xyz_rate_body
            source_att = bee.attitude

            adapt_velx = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.LIF())

            vel_learnx_u = nengo.Node(None, size_in=1)
            vel_learnx = nengo.Connection(adapt_velx, vel_learnx_u, learning_rule_type=nengo.PES(p.adapt_learn_rate, pre_tau=0.01),
                                        function=lambda x:0,
                                        synapse=.01)
            nengo.Connection(bee.attitude[2], adapt_velx, synapse=None)

            nengo.Connection(vel_learnx_u, u[1], synapse=None)

            adapt_velz = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.LIF())

            vel_learnz_u = nengo.Node(None, size_in=1)
            vel_learnz = nengo.Connection(adapt_velz, vel_learnz_u, learning_rule_type=nengo.PES(p.adapt_learn_rate, pre_tau=0.01),
                                        function=lambda x:0,
                                        synapse=.01)

            nengo.Connection(vel_learnz_u, u[0], synapse=None)

            if p.adapt_Kp > 0:
                nengo.Connection(source_body_vel[0], vel_learnx.learning_rule, transform=-1*p.adapt_Kp, synapse=None)
            if p.adapt_Kd > 0:
                nengo.Connection(source_body_vel[0], vel_learnx.learning_rule, transform=-1*p.adapt_Kd, synapse=None)
                nengo.Connection(source_body_vel[0], vel_learnx.learning_rule, transform=1*p.adapt_Kd, synapse=0.005)

            #currently using same K's for both dims
            if p.adapt_Kp > 0:
                nengo.Connection(source_body_vel[2], vel_learnz.learning_rule, transform=-1*p.adapt_Kp, synapse=None)
            if p.adapt_Kd > 0:
                nengo.Connection(source_body_vel[2], vel_learnz.learning_rule, transform=-1*p.adapt_Kd, synapse=None)
                nengo.Connection(source_body_vel[2], vel_learnz.learning_rule, transform=1*p.adapt_Kd, synapse=0.005)

            nengo.Connection(bee.plant[keep_x], ens[:len(keep_x)], synapse=None, transform=1.0/ctrl['std_x_body'][keep_x])
            if len(keep_u) > 0:
                nengo.Connection(bee.u[keep_u], ens[len(keep_x):], synapse=None, transform=1.0/ctrl['std_u'][keep_u])

            nengo.Connection(nengo.Node(ctrl['mean_x_body']/ctrl['std_x_body'])[keep_x], ens[:len(keep_x)], transform=-1, synapse=None)
            if len(keep_u) > 0:
                nengo.Connection(nengo.Node(ctrl['mean_u']/ctrl['std_u'])[keep_u], ens[len(keep_x):], transform=-1, synapse=None)

            self.conn = nengo.Connection(ens, u_unfilt, eval_points=pts, scale_eval_points=False, function=target,
                             synapse=0.01,
                             solver=nengo.solvers.LstsqL2(reg=p.reg))

            nengo.Connection(u_unfilt[0], u[0], synapse=None)
            nengo.Connection(u_unfilt[1:], u[1:], synapse=None)

            nengo.Connection(u, bee.u, synapse=0)
            
            self.probe_u = nengo.Probe(u, synapse=None)


            plot_thrust = nengo.Node(None, size_in=2)
            nengo.Connection(u[0], plot_thrust[1], synapse=None)

            plot_pitch = nengo.Node(None, size_in=2)
            nengo.Connection(u[1], plot_pitch[1], synapse=None)

            plot_roll = nengo.Node(None, size_in=2)
            nengo.Connection(u[3], plot_roll[1], synapse=None)

            if p.use_learning_display:
                import nengo_learning_display

                S = 30
                D = ens.dimensions
                learn_plots = []
                for i in range(D):
                    domain = np.zeros((S, D))
                    domain[:,i] = np.linspace(-2, 2, S)

                    learn_plots.append(nengo_learning_display.Plot1D(self.conn, domain,
                           range=(-1,1)))

                    if i < len(keep_x):
                        learn_plots[i].label = 'x[%d]' % keep_x[i]
                    else:
                        learn_plots[i].label = 'u[%d]' % keep_u[i-len(keep_x)]

                domain = np.zeros((S, S, D))
                grid = np.meshgrid(np.linspace(-1, 1, S), np.linspace(-1,1,S))
                grid = np.array(grid).T
                domain[:,:,[1,4]] = grid

                vel_plot = nengo_learning_display.Plot1D(vel_learnx, np.linspace(-1,1,30), range=(-1,1))
                vel_plot.label = 'adapting for velocity'
                learn_plots.append(vel_plot)

                def on_step(sim):
                    for p in learn_plots:
                        p.update(sim)
            

        if p.gui:
            self.locals = locals()
        return model

    def evaluate(self, p, sim, plt):
        with sim:
            sim.run(p.T)

        if plt:
            plt.subplot(4, 2, 1)
            plt.plot(sim.trange(), sim.data[self.probe_x][:,11:14])
            plt.ylabel('position (m)')
            plt.legend(['x', 'y', 'z'], loc='best')

            plt.subplot(4, 2, 3)
            plt.plot(sim.trange(), sim.data[self.probe_x][:,17:20])
            plt.ylabel('velocity (m)')
            plt.legend(['x', 'y', 'z'], loc='best')

            plt.subplot(4, 2, 5)
            plt.plot(sim.trange(), sim.data[self.probe_x][:,8:11])
            plt.ylabel('attitude (radians)')
            plt.legend(['yaw $\phi$', 'roll $\\theta$', 'pitch $\psi$'], loc='best')

            plt.subplot(4, 2, 7)
            plt.plot(sim.trange(), sim.data[self.probe_x][:,14:17])
            plt.ylabel('attitude rate (radians)')
            plt.legend(['yaw $\phi$', 'roll $\\theta$', 'pitch $\psi$'], loc='best')

            plt.subplot(4, 2, 2)
            plt.plot(sim.trange(), sim.data[self.probe_u])
            plt.ylabel('u')
            plt.legend(['stroke ampl.', 'pitch torque', 'yaw torque', 'roll'], loc='best')

            plt.subplot(4, 2, 4)
            plt.plot(sim.trange(), sim.data[self.probe_pif_u])
            plt.ylabel('PIF u')
            plt.legend(['stroke ampl.', 'pitch torque', 'yaw torque', 'roll'], loc='best')

            plt.subplot(4, 2, 6)
            plt.plot(sim.trange(), sim.data[self.probe_u] - sim.data[self.probe_pif_u])
            plt.ylabel('u error')
            plt.legend(['stroke ampl.', 'pitch torque', 'yaw torque', 'roll'], loc='best')

        return dict(
            solver_error = sim.data[self.conn].solver_info['rmses'],
            x=sim.data[self.probe_x],
            u=sim.data[self.probe_u],
            pif_u=sim.data[self.probe_pif_u])

        






    