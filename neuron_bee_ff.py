import sys
sys.path.append('.')
sys.path.append(r'..\PyBee3D\PyBee3D')
import pytry
import numpy as np
import robobee
import scipy
from scipy import integrate
import nengo
# from controllers.pif.pif_compensator import PIFCompensator
from pif_control import PIFControl
from nengo_bee import NengoBee
import seaborn
from itertools import product

class GatherDataTrial(pytry.NengoTrial):
    def params(self):
        self.param('velocity', velocity=0.0)
        self.param('angle', angle=0.0)
        self.param('turn rate', turn_rate=0.0)
        self.param('initial pose variability', pose_var=0.0)
        self.param('initial rotation rate variability', dpose_var=0.0)
        self.param('initial velocity variability', vel_var=0.0)
        self.param('controller filename', ctrl_filename='gather-gain_scheduled_12_11.npz')
        self.param('time', T=1.0)
        self.param('number of neurons', n_neurons=500)
        self.param('Adaptive Neurons', n_adapt_neurons=100)
        self.param('regularization', reg=0.1)
        self.param('use PIF control', use_pif=False)
        self.param('radius scaling', radius_scaling=1.0)
        self.param('oracle learning', oracle=False)
        self.param('intercept minimum', low_intercept=-1.0)
        self.param('learning_rate', learning_rate=1e-4)
        self.param('Ki', Ki=0.0)
        self.param('use world frame for adaptation', world_adapt=False)
        self.param('use learning adaptation', adapt=True)
        self.param('learning adaptation rate', adapt_learn_rate=1e-4)
        self.param('use learning display', use_learning_display=False)
        self.param('apply wing bias', wing_bias=True)
        self.param('adapt Kp scale', adapt_Kp=1.0)
        self.param('adapt Kd scale', adapt_Kd=30.0)
        self.param('adapt Ki scale', adapt_Ki=0.0)
        self.param('wind speed', v_wind=0.0)
        self.param('Initial heading', phi_0=0.0)
        self.param('Actuator Failure', actuator_failure=False)
        self.param('Controller Time Step', control_dt=0.001)
        self.param('Fancy Flight', fancy_flight=False)
        self.param('Initial y_star Conditions', init_with_y_star=False)

    def model(self, p):

        ctrl = np.load(p.ctrl_filename)

        model = nengo.Network()
        with model:
            pose_offset = np.random.uniform(-p.pose_var, p.pose_var, size=3)
            dpose_offset = np.random.uniform(-p.dpose_var, p.dpose_var, size=3)
            vel_offset = np.random.uniform(-p.vel_var, p.vel_var, size=3)

            y_star_init = np.zeros(4)
            if p.init_with_y_star:
                y_star_init[0] = p.velocity
                y_star_init[1] = p.turn_rate
                y_star_init[2] = p.angle

            bee = NengoBee(pose_offset=pose_offset,
                           dpose_offset=dpose_offset,
                           vel_offset=vel_offset,
                           random_wing_bias=p.wing_bias,
                           v_wind=p.v_wind,
                           phi_0=p.phi_0,
                           actuator_failure=p.actuator_failure,
                           sample_dt=p.control_dt,
                           y_star_init=y_star_init)
            
            control = PIFControl(bee.bee)

            # Set up command input node
            if p.fancy_flight:
                def get_velocity_set_point(t):
                    if t < 1.0:
                        return 0
                    elif t < 2.0:
                        return 0.2
                    elif t < 3.0:
                        return 0.2
                    elif t < 4.0:
                        return 0.2
                    else:
                        return 0
                    # if t < 2.0:
                    #     return 0
                    # else:
                    #     return 0.5
                def get_angle_set_point(t):
                    if t < 1.0:
                        return 0
                    elif t < 2.0:
                        return 90
                    elif t < 3.0:
                        return 0
                    elif t < 4.0:
                        return -90
                    else:
                        return 0
                    # if t < 2.0:
                    #     return 0
                    # else:
                    #     return 90*np.sin(2*np.pi/6*(t - 2.0))
                v = nengo.Node(get_velocity_set_point)
                a = nengo.Node(get_angle_set_point)
            else:
                v = nengo.Node(p.velocity)
                a = nengo.Node(p.angle)

            xi_dot = nengo.Node(p.turn_rate)
            nengo.Connection(xi_dot, control.y_star[1], synapse=None)
            nengo.Connection(v, control.y_star[0], synapse=None)
            nengo.Connection(a, control.y_star[2], synapse=None)

            keep_x_names = ['theta', 'psi', 'phi_dot', 'theta_dot', 'psi_dot', 'v_x', 'v_y', 'v_z']
            # keep_x_names = ['theta_dot', 'psi_dot', 'v_x', 'v_y', 'v_z']
            keep_x = [robobee.state_names.index(name) for name in keep_x_names]
            keep_u = []
            keep_y_star = [0, 2]

            # Use the body frame velocities
            y_star_vals = ctrl['norm_y_star'][:,keep_y_star]
            x_vals = ctrl['norm_x_body'][:,keep_x]
            u_vals = ctrl['norm_delta_u'][:,keep_u]

            target = ctrl['all_delta_u']

            pts = np.hstack([x_vals, u_vals, y_star_vals])

            D = pts.shape[1]
            ens = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=D,
                                 intercepts=nengo.dists.Uniform(p.low_intercept, 1.0),
                                 neuron_type=nengo.LIF(), radius=np.sqrt(D)*p.radius_scaling)

            vel_pts = [0, 0.5, 1.0]
            turn_rate_pts = [0]
            angle_pts = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
            sideslip_pts = [0]

            # get the feedforward control input at different points
            y_star_pts = list(product(vel_pts, turn_rate_pts, angle_pts, sideslip_pts))
            u_star_pts = [control.controller.get_u_star(y_star) for y_star in y_star_pts]

            y_star_pts = np.array(y_star_pts)
            u_star_pts = np.array(u_star_pts)

            mean_y_star = np.mean(y_star_pts, axis=0)
            std_y_star = np.std(y_star_pts, axis=0)
            norm_y_star = (y_star_pts - mean_y_star) / std_y_star

            y_star_vals_ff = norm_y_star[:, keep_y_star]

            D = y_star_vals_ff.shape[1]
            ens_ff = nengo.Ensemble(n_neurons=500, dimensions=D,
                                    intercepts=nengo.dists.Uniform(-1.0, 1.0),
                                    neuron_type=nengo.LIF(), radius=np.sqrt(D) * p.radius_scaling)

            nengo.Connection(control.y_star[keep_y_star], ens_ff, synapse=None, transform=1.0 / std_y_star[keep_y_star])
            nengo.Connection(nengo.Node((mean_y_star / std_y_star)[keep_y_star]), ens_ff, transform=-1, synapse=None)

            u_ff = nengo.Node(None, size_in=4)

            self.conn = nengo.Connection(ens_ff, u_ff,
                                         eval_points=y_star_vals_ff,
                                         scale_eval_points=False,
                                         function=u_star_pts,
                                         synapse=0.04,
                                         learning_rule_type=None,
                                         solver=nengo.solvers.LstsqL2(reg=p.reg))

            delta_u_unfilt = nengo.Node(None, size_in=4)
            u = nengo.Node(None, size_in=4)

            if p.Ki > 0:
                source_body_vel = bee.xyz_rate_body
                dz_error = nengo.Ensemble(n_neurons=p.n_adapt_neurons, dimensions=1, neuron_type=nengo.LIF())
                nengo.Connection(source_body_vel[2], dz_error, synapse=None)
                nengo.Connection(dz_error, dz_error, synapse=0.1)
                nengo.Connection(dz_error, u[0], transform=p.Ki, synapse=.01)
                
                dx_error = nengo.Ensemble(n_neurons=p.n_adapt_neurons, dimensions=1, neuron_type=nengo.LIF())
                nengo.Connection(source_body_vel[0], dx_error, synapse=None)
                nengo.Connection(dx_error, dx_error, synapse=0.1)
                nengo.Connection(dx_error, u[1], transform=.01*p.Ki, synapse=.01)

            if p.adapt:
                # source_body_vel = bee.xyz_rate_body
                source_body_vel = bee.body_vel_sampled
                source_att = bee.attitude

                #t_inhibit = 1.0

                #def inhibit(t, x):
                    #if t < t_inhibit:
                    #    return np.zeros(np.shape(x))
                    #else:
                    #    return x

                #learn_vel_switch = nengo.Node(inhibit, size_in=3, size_out=3)

                adapt_velx = nengo.Ensemble(n_neurons=p.n_adapt_neurons, dimensions=1, neuron_type=nengo.LIF())

                vel_learnx_u = nengo.Node(None, size_in=1)
                vel_learnx = nengo.Connection(adapt_velx, vel_learnx_u, learning_rule_type=nengo.PES(p.adapt_learn_rate, pre_tau=0.01),
                                            function=lambda x:0,
                                            synapse=.01)
                nengo.Connection(bee.attitude[2], adapt_velx, synapse=None)

                nengo.Connection(vel_learnx_u, u[1], synapse=None)

                adapt_velz = nengo.Ensemble(n_neurons=p.n_adapt_neurons, dimensions=1, neuron_type=nengo.LIF())

                vel_learnz_u = nengo.Node(None, size_in=1)
                vel_learnz = nengo.Connection(adapt_velz, vel_learnz_u, learning_rule_type=nengo.PES(p.adapt_learn_rate, pre_tau=0.01),
                                            function=lambda x:0,
                                            synapse=.01)
                #nengo.Connection(bee.attitude[2], adapt_velz, synapse=None)

                nengo.Connection(vel_learnz_u, u[0], synapse=None)


                adapt_vely = nengo.Ensemble(n_neurons=p.n_adapt_neurons, dimensions=1, neuron_type=nengo.LIF())

                vel_learny_u = nengo.Node(None, size_in=1)
                vel_learny = nengo.Connection(adapt_vely, vel_learny_u, learning_rule_type=nengo.PES(p.adapt_learn_rate, pre_tau=0.01),
                                            function=lambda x:0,
                                            synapse=.01)
                nengo.Connection(vel_learny_u, u[3], synapse=None)
                # vel_learny_u_probed = nengo.Node(None, size_in=1)
                # nengo.Connection(vel_learny_u, vel_learny_u_probed, synapse=None)
                # nengo.Connection(bee.attitude[1], adapt_vely, synapse=None)

                self.probe_adaptx_u = nengo.Probe(vel_learnx_u, synapse=None)
                self.probe_adapty_u = nengo.Probe(vel_learny_u, synapse=None)
                self.probe_adaptz_u = nengo.Probe(vel_learnz_u, synapse=None)

                # TODO: Replace source_body_vel below with source_body_vel - target_body_vel
                vel_error = nengo.Node(None, size_in=3)

                nengo.Connection(control.target_body_vel, vel_error, transform=-1)
                nengo.Connection(source_body_vel, vel_error)

                # Gains for 12-14 (longitudinal flight)
                # adapt_Kp=0.6, adapt_Kd=0.3,
                # kp: -60, 4, -40
                # kd: -1600, 0.2, -8000

                kp_x = -40
                kp_y =2
                kp_z = -100

                kd_x = -140
                kd_y = 20
                kd_z = -200

                if p.adapt_Kp > 0:
                    nengo.Connection(vel_error[0], vel_learnx.learning_rule, transform=kp_x*p.adapt_Kp, synapse=None)
                if p.adapt_Kd > 0:
                    nengo.Connection(vel_error[0], vel_learnx.learning_rule, transform=kd_x*p.adapt_Kd, synapse=None)
                    nengo.Connection(vel_error[0], vel_learnx.learning_rule, transform=-kd_x*p.adapt_Kd, synapse=0.005)
                if p.adapt_Ki > 0:
                    nengo.Connection(bee.xyz[0], vel_learnx.learning_rule, transform=0*p.adapt_Ki, synapse=None)
                #currently using same K's for all dims
                if p.adapt_Kp > 0:
                    nengo.Connection(vel_error[1], vel_learny.learning_rule, transform=kp_y*p.adapt_Kp, synapse=None)
                if p.adapt_Kd > 0:
                    nengo.Connection(vel_error[1], vel_learny.learning_rule, transform=kd_y*p.adapt_Kd, synapse=None)
                    nengo.Connection(vel_error[1], vel_learny.learning_rule, transform=-kd_y*p.adapt_Kd, synapse=0.005)
                if p.adapt_Ki > 0:
                    nengo.Connection(bee.xyz[1], vel_learny.learning_rule, transform=0*p.adapt_Ki, synapse=None)

                if p.adapt_Kp > 0:
                    nengo.Connection(vel_error[2], vel_learnz.learning_rule, transform=kp_z*p.adapt_Kp, synapse=None)
                if p.adapt_Kd > 0:
                    nengo.Connection(vel_error[2], vel_learnz.learning_rule, transform=kd_z*p.adapt_Kd, synapse=None)
                    nengo.Connection(vel_error[2], vel_learnz.learning_rule, transform=-kd_z*p.adapt_Kd, synapse=0.005)
                if p.adapt_Ki > 0:
                    nengo.Connection(bee.xyz[2], vel_learnz.learning_rule, transform=1*p.adapt_Ki, synapse=None)

                #inhibit_adapt = nengo.Node(0)
                #nengo.Connection(inhibit_adapt, adapt_vel.neurons, transform=-10*np.ones((adapt_vel.n_neurons, 1)))
            else:
                self.probe_adaptx_u = nengo.Probe(nengo.Node(None, size_in=1), synapse=None)
                self.probe_adapty_u = nengo.Probe(nengo.Node(None, size_in=1), synapse=None)
                self.probe_adaptz_u = nengo.Probe(nengo.Node(None, size_in=1), synapse=None)

            nengo.Connection(bee.x_body_sampled[keep_x], ens[:len(keep_x)], synapse=None, transform=1.0/ctrl['std_x_body'][keep_x])
            if len(keep_u) > 0:
                nengo.Connection(bee.u[keep_u], ens[len(keep_x):], synapse=None, transform=1.0/ctrl['std_u'][keep_u])
            nengo.Connection(control.y_star[keep_y_star], ens[-len(keep_y_star):], synapse=None, transform=1.0/ctrl['std_y_star'][keep_y_star])

            nengo.Connection(nengo.Node((ctrl['mean_x_body']/ctrl['std_x_body'])[keep_x]), ens[:len(keep_x)], transform=-1, synapse=None)
            if len(keep_u) > 0:
                nengo.Connection(nengo.Node((ctrl['mean_u']/ctrl['std_u'])[keep_u]), ens[len(keep_x):], transform=-1, synapse=None)
            nengo.Connection(nengo.Node((ctrl['mean_y_star'] / ctrl['std_y_star'])[keep_y_star]), ens[-len(keep_y_star):], transform=-1, synapse=None)

            if p.oracle:
                learning_rule=nengo.PES(learning_rate=p.learning_rate)
            else:
                learning_rule=None
            self.conn = nengo.Connection(ens, delta_u_unfilt,
                                         eval_points=pts,
                                         scale_eval_points=False,
                                         function=target,
                                         synapse=0.01,
                                         learning_rule_type=learning_rule,
                                         solver=nengo.solvers.LstsqL2(reg=p.reg))

            if p.use_pif:
                nengo.Connection(control.control, u, synapse=None)
            else:
                nengo.Connection(delta_u_unfilt[0], u[0], synapse=None)
                nengo.Connection(delta_u_unfilt[1:], u[1:], synapse=None)
                nengo.Connection(u_ff, u, synapse=None)
            nengo.Connection(u, bee.u, synapse=None)

            self.probe_u_ff = nengo.Probe(u_ff, synapse=None)
            self.probe_pif_ff = nengo.Probe(control.u_star, synapse=None)
            
            nengo.Connection(bee.plant_unfilt, control.x, synapse=0)
            nengo.Connection(control.control, control.u, synapse=0)

            self.probe_pif_u = nengo.Probe(control.control, synapse=None)
            self.probe_x = nengo.Probe(control.x, synapse=None)
            self.probe_x_unfilt = nengo.Probe(bee.plant_unfilt, synapse=None)
            self.probe_u = nengo.Probe(bee.u, synapse=None)
            self.probe_y_star = nengo.Probe(control.y_star, synapse=None)
            self.probe_ens = nengo.Probe(delta_u_unfilt, synapse=None)
            self.probe_x_star = nengo.Probe(control.x_star, synapse=None)

            plot_thrust = nengo.Node(None, size_in=2)
            nengo.Connection(control.control[0], plot_thrust[0], synapse=None)
            nengo.Connection(u[0], plot_thrust[1], synapse=None)

            plot_pitch = nengo.Node(None, size_in=2)
            nengo.Connection(control.control[1], plot_pitch[0], synapse=None)
            nengo.Connection(u[1], plot_pitch[1], synapse=None)

            plot_roll = nengo.Node(None, size_in=2)
            nengo.Connection(control.control[3], plot_roll[0], synapse=None)
            nengo.Connection(u[3], plot_roll[1], synapse=None)

            if p.oracle:
                def error_func(t, x):
                    if x[-1] > 0.5:
                        return x[:-1] * 0
                    else:
                        return x[:-1]
                error = nengo.Node(error_func, size_in=5)
                nengo.Connection(control.control, error[:4], synapse=None, transform=-1)
                nengo.Connection(u, error[:4], synapse=None)
                nengo.Connection(error, self.conn.learning_rule, synapse=0)

                stop_learning = nengo.Node(0)
                nengo.Connection(stop_learning, error[-1], synapse=None)

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
                vel_plot.label = 'adapting for x velocity'
                learn_plots.append(vel_plot)

                vel_plot = nengo_learning_display.Plot1D(vel_learny, np.linspace(-1,1,30), range=(-.1,.1))
                vel_plot.label = 'adapting for y velocity'
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

        l = len(sim.data[self.probe_x])
        if sim.dt <= 1E-4:
            idx = np.arange(0, l, 20)
        else:
            idx = np.arange(0, l, 2)

        return dict(
            solver_error = sim.data[self.conn].solver_info['rmses'],
            x=sim.data[self.probe_x][idx, :],
            u=sim.data[self.probe_u][idx, :],
            y_star=sim.data[self.probe_y_star][idx, :],
            pif_u=sim.data[self.probe_pif_u][idx, :],
            # pif_u_dot=sim.data[self.probe_pif_u_dot],
            ens=sim.data[self.probe_ens][idx, :],
            adapt_x=sim.data[self.probe_adaptx_u][idx, :],
            adapt_y=sim.data[self.probe_adapty_u][idx, :],
            adapt_z=sim.data[self.probe_adaptz_u][idx, :],
            x_unfilt=sim.data[self.probe_x_unfilt][idx, :],
            x_star=sim.data[self.probe_x_star][idx, :],
            u_ff=sim.data[self.probe_u_ff][idx, :],
            pif_ff=sim.data[self.probe_pif_ff][idx, :])
            # learny=sim.data[self.probe_learny_u])









