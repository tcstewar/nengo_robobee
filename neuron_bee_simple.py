import sys
sys.path.append('.')
sys.path.append(r'..\PyBee3D\PyBee3D')
import pytry
import numpy as np
import robobee
import scipy
from scipy import integrate
import nengo
from controllers.pif.pif_compensator import PIFCompensator
from pif_control import PIFControl
from nengo_bee import NengoBee
import seaborn
from itertools import product
from scipy.integrate import ode

class GatherDataTrial(pytry.NengoTrial):
    def params(self):
        self.param('time', T=1.0)
        self.param('velocity', vel_target=0.0)
        self.param('angle', climb_angle=0.0)
        self.param('turn rate', turn_rate=0.0)
        self.param('controller file name', ctrl_filename='gather-ff_1_8_test.npz')
        self.param('Number of Neurons', n_neurons=500)
        self.param('radius scaling', radius_scaling=1.0)
        self.param('intercept minimum', low_intercept=-1.0)
        self.param('regularization', reg=0.1)
        self.param('use PIF control', use_pif=False)

    def model(self, p):
        data = np.load(p.ctrl_filename)

        model = nengo.Network()
        with model:
            y_star_0 = np.zeros(4)

            self.bee = robobee.RoboBee(random_wing_bias=False,
                                       actuator_failure=False,
                                       a_theta=-0.2, b_theta=0.04,
                                       new_yaw_control=True)

            self.pif = PIFCompensator(gains_file='PIF_Gains_Body_New_Yaw.mat')

            def control_ode_fun(t, eta, y_star, x, bee):
                u = eta[0:4]
                return self.pif.get_control_dynamics(eta, t, y_star, x, u, bee)

            eta_0 = np.zeros(7)
            x_0 = self.pif.get_x_star(y_star_0)*0
            u_0 = self.pif.get_u_star(y_star_0)*0
            eta_0[0:4] = u_0

            x_0[self.bee.idx_body_att] += np.array([0.3, -0.2, 0.5])

            self.integrator_control = ode(control_ode_fun).set_integrator('dopri5')
            self.integrator_dynamics = ode(self.bee.get_dynamics).set_integrator('dopri5')
            self.integrator_dynamics.set_f_params(u_0, np.zeros(3))
            self.integrator_dynamics.set_initial_value(x_0, 0)
            self.integrator_control.set_initial_value(eta_0, 0)

            keep_x_names = ['theta', 'psi', 'phi_dot', 'theta_dot', 'psi_dot', 'v_x', 'v_y', 'v_z']
            keep_x = [robobee.state_names.index(name) for name in keep_x_names]
            keep_y_star = [0, 1, 2]

            # --------------- Load training data ---------------
            y_star_norm = data['norm_y_star'][:,keep_y_star]

            # --------------- Set up SNN Layers ---------------
            D = y_star_norm.shape[1]
            ens_gains = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=D,
                                       intercepts=nengo.dists.Uniform(p.low_intercept, 1.0),
                                       neuron_type=nengo.LIF(), radius=np.sqrt(D)*p.radius_scaling)

            ens_u_star = nengo.Ensemble(n_neurons=500, dimensions=D,
                                        intercepts=nengo.dists.Uniform(-1.0, 1.0),
                                        neuron_type=nengo.LIF(), radius=np.sqrt(D) * p.radius_scaling)

            # --------------- Node Functions ---------------
            def get_pif_control(t, v):
                if t > 0:
                    y_star = v[0:4]
                    x = v[4:24]
                    self.integrator_control.set_f_params(y_star, x, self.bee)
                    u = self.integrator_control.integrate(t)[0:4]
                else:
                    u = self.pif.get_u_star([0, 0, 0, 0])
                return u

            def get_state(t, u):
                if t > 0:
                    self.integrator_dynamics.set_f_params(u, np.zeros(3))
                    x = self.integrator_dynamics.integrate(t)
                else:
                    x = self.pif.get_x_star([0, 0, 0, 0])
                return x

            def get_x_star(t, v):
                return self.pif.get_x_star(v)

            def get_u_star(t, v):
                return self.pif.get_u_star(v)

            def world_state_to_body(t, x):
                body_x = self.bee.world_state_to_body(x)
                return body_x

            def multiply_gain(t, v):
                K = np.reshape(v[0:4*len(keep_x)], (4, len(keep_x)))
                x = v[-len(keep_x):]
                return K @ x

            # --------------- Set up Nodes ---------------
            u_pif = nengo.Node(get_pif_control, size_in=24)
            u = nengo.Node(None, size_in=4)
            plant = nengo.Node(get_state, size_in=4)
            x_body = nengo.Node(world_state_to_body, size_in=20)
            y_star = nengo.Node(None, size_in=4)
            u_star = nengo.Node(None, size_in=4)

            set_point = nengo.Node(None, size_in=24)
            u_tilde = nengo.Node(multiply_gain, size_in=(4 * len(keep_x) + len(keep_x)), size_out=4)
            x_tilde = nengo.Node(None, size_in=len(keep_x))
            if p.use_pif:
                x_star = nengo.Node(get_x_star, size_in=4)
            else:
                x_star = nengo.Node(None, size_in=20)

            v = nengo.Node(p.vel_target)
            a = nengo.Node(p.climb_angle)
            xi_dot = nengo.Node(p.turn_rate)

            # --------------- Set up Connections ---------------
            nengo.Connection(v, y_star[0], synapse=None)
            nengo.Connection(xi_dot, y_star[1], synapse=None)
            nengo.Connection(a, y_star[2], synapse=None)

            nengo.Connection(y_star, u_pif[0:4], synapse=None)
            nengo.Connection(plant, u_pif[4:], synapse=None)
            nengo.Connection(plant, x_body, synapse=None)
            if p.use_pif:
                nengo.Connection(u_pif, u, synapse=None)
                nengo.Connection(y_star, x_star)
                nengo.Connection(y_star, u_star)
            else:
                set_point_vals = np.hstack((data['u_star_vals'], data['x_star_vals']))

                nengo.Connection(y_star[keep_y_star], ens_u_star,
                                 synapse=None, transform=1.0 / data['std_y_star'][keep_y_star])
                nengo.Connection(nengo.Node((data['mean_y_star'] / data['std_y_star'])[keep_y_star]),
                                 ens_u_star, transform=-1, synapse=None)
                nengo.Connection(ens_u_star, set_point,
                                 eval_points=y_star_norm,
                                 scale_eval_points=False,
                                 function=set_point_vals,
                                 synapse=0.02,
                                 learning_rule_type=None,
                                 solver=nengo.solvers.LstsqL2(reg=p.reg))
                nengo.Connection(set_point[0:4], u_star)
                nengo.Connection(set_point[4:], x_star)

                nengo.Connection(y_star[keep_y_star], ens_gains,
                                 synapse=None, transform=1.0 / data['std_y_star'][keep_y_star])
                nengo.Connection(nengo.Node((data['mean_y_star'] / data['std_y_star'])[keep_y_star]),
                                 ens_gains, transform=-1, synapse=None)
                nengo.Connection(ens_gains, u_tilde[:-len(keep_x)],
                                 eval_points=y_star_norm,
                                 scale_eval_points=False,
                                 function=data['ss_gains'],
                                 synapse=0.01,
                                 learning_rule_type=None,
                                 solver=nengo.solvers.LstsqL2(reg=p.reg))

                nengo.Connection(x_star[keep_x], x_tilde, synapse=None, transform=-1)
                nengo.Connection(x_body[keep_x], x_tilde, synapse=None)
                nengo.Connection(x_tilde, u_tilde[-len(keep_x):], synapse=0.02)
                nengo.Connection(u_tilde, u, synapse=0.01)
                nengo.Connection(u_star, u, synapse=0.02)

            nengo.Connection(u, plant, synapse=0)

            # --------------- Set up Probes ---------------
            self.probe_u = nengo.Probe(u, synapse=None)
            self.probe_x = nengo.Probe(plant, synapse=None)
            self.probe_x_body = nengo.Probe(x_body, synapse=None)
            self.probe_y_star = nengo.Probe(y_star, synapse=None)
            self.probe_u_star = nengo.Probe(u_star, synapse=None)
            self.probe_x_star = nengo.Probe(x_star, synapse=None)
            self.probe_u_pif = nengo.Probe(u_pif, synapse=None)

        if p.gui:
            self.locals = locals()
        return model

    def evaluate(self, p, sim, plt):
        with sim:
            sim.run(p.T)

        return dict(
            x=sim.data[self.probe_x],
            x_body=sim.data[self.probe_x_body],
            u=sim.data[self.probe_u],
            u_pif=sim.data[self.probe_u_pif],
            y_star=sim.data[self.probe_y_star],
            x_star=sim.data[self.probe_x_star],
            u_star=sim.data[self.probe_u_star])









