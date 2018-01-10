# Test file - currently being used to test the PIF compensator in the nengo framework

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
                    if t < 2.0:
                        return 0
                    elif t < 3.0:
                        return 0.2
                    elif t < 3.5:
                        return 0
                    elif t < 4.5:
                        return 0.2
                    elif t < 5.0:
                        return 0
                    elif t < 6.0:
                        return 0.2
                    else:
                        return 0
                    # if t < 2.0:
                    #     return 0
                    # else:
                    #     return 0.5
                def get_angle_set_point(t):
                    if t < 2.0:
                        return 0
                    elif t < 3.0:
                        return 90
                    elif t < 3.5:
                        return 0
                    elif t < 4.5:
                        return 0
                    elif t < 5.0:
                        return 0
                    elif t < 6.0:
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

            # delta_u_unfilt = nengo.Node(None, size_in=4)
            u = nengo.Node(None, size_in=4)

            self.probe_adaptx_u = nengo.Probe(nengo.Node(None, size_in=1), synapse=None)
            self.probe_adapty_u = nengo.Probe(nengo.Node(None, size_in=1), synapse=None)
            self.probe_adaptz_u = nengo.Probe(nengo.Node(None, size_in=1), synapse=None)

            nengo.Connection(control.control, u, synapse=None)
            nengo.Connection(u, bee.u, synapse=0)

            self.probe_pif_ff = nengo.Probe(control.u_star, synapse=None)
            
            nengo.Connection(bee.plant_unfilt, control.x, synapse=None)
            nengo.Connection(bee.u, control.u, synapse=None)

            self.probe_pif_u = nengo.Probe(control.control, synapse=None)
            self.probe_x = nengo.Probe(control.x, synapse=None)
            self.probe_x_unfilt = nengo.Probe(bee.plant_unfilt, synapse=None)
            self.probe_u = nengo.Probe(bee.u, synapse=None)
            self.probe_y_star = nengo.Probe(control.y_star, synapse=None)
            # self.probe_ens = nengo.Probe(delta_u_unfilt, synapse=None)
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
            # solver_error = sim.data[self.conn].solver_info['rmses'],
            x=sim.data[self.probe_x][idx, :],
            u=sim.data[self.probe_u][idx, :],
            y_star=sim.data[self.probe_y_star][idx, :],
            pif_u=sim.data[self.probe_pif_u][idx, :],
            # pif_u_dot=sim.data[self.probe_pif_u_dot],
            # ens=sim.data[self.probe_ens][idx, :],
            adapt_x=sim.data[self.probe_adaptx_u][idx, :],
            adapt_y=sim.data[self.probe_adapty_u][idx, :],
            adapt_z=sim.data[self.probe_adaptz_u][idx, :],
            x_unfilt=sim.data[self.probe_x_unfilt][idx, :],
            x_star=sim.data[self.probe_x_star][idx, :],
            # u_ff=sim.data[self.probe_u_ff][idx, :],
            pif_ff=sim.data[self.probe_pif_ff][idx, :])
            # learny=sim.data[self.probe_learny_u])









