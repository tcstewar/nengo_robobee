import sys
sys.path.append('.')
sys.path.append(r'C:\Users\taylor\OneDrive\Cornell\LISC\Code\RoboBee\PyBee3D\PyBee3D')
import pytry
import numpy as np
import nengo
from pif_control import PIFControl
from controllers.pif.pif_compensator import PIFCompensator
from nengo_bee import NengoBee

class GatherDataTrial(pytry.PlotTrial):
    def params(self):
        self.param('x', x=np.zeros(12))
        self.param('u', u=np.zeros(4))

    def evaluate(self, p, plt):

        controller = PIFCompensator()

        # Get gains for hovering
        y_star = [0, 0, 0, 0]

        # we are ignoring the integral term in the approximation
        eta = np.zeros(7)       # eta contains the control input u followed by the output y (dimension 4 and 3)

        bee = NengoBee(random_wing_bias=False)
        t = np.random.uniform(0, bee.bee.T)

        u_dot = controller.get_control_dynamics(eta, t, y_star, p.x, p.u, bee.bee)[:4]

        return dict(u_dot=u_dot)
        #
        # model = nengo.Network()
        # with model:
        #     pose_offset = np.random.uniform(-p.pose_var, p.pose_var, size=3)
        #     dpose_offset = np.random.uniform(-p.dpose_var, p.dpose_var, size=3)
        #
        #     bee = NengoBee(pose_offset=pose_offset, dpose_offset=dpose_offset)
        #
        #     control = PIFControl(bee.bee)
        #
        #     nengo.Connection(bee.plant, control.x, synapse=None)
        #     nengo.Connection(control.control, bee.u, synapse=0)
        #     nengo.Connection(bee.u, control.u, synapse=None)
        #
        #     v = nengo.Node(p.velocity)
        #     nengo.Connection(v, control.y_star[0], synapse=None)
        #     a = nengo.Node(p.angle)
        #     nengo.Connection(a, control.y_star[2], synapse=None)
        #
        #     probe_x = nengo.Probe(control.x, synapse=None)
        #     probe_u = nengo.Probe(bee.u, synapse=None)
        #
        # sim = nengo.Simulator(model, dt=p.dt)
        # with sim:
        #     sim.run(p.T)
        #
        # if plt:
        #     plt.subplot(5, 1, 1)
        #     plt.plot(sim.trange(), sim.data[probe_x][:,11:14])
        #     plt.ylabel('position (m)')
        #     plt.legend(['x', 'y', 'z'], loc='best')
        #
        #     plt.subplot(5, 1, 2)
        #     plt.plot(sim.trange(), sim.data[probe_x][:,17:20])
        #     plt.ylabel('velocity (m)')
        #     plt.legend(['x', 'y', 'z'], loc='best')
        #
        #     plt.subplot(5, 1, 3)
        #     plt.plot(sim.trange(), sim.data[probe_x][:,8:11])
        #     plt.ylabel('attitude (radians)')
        #     plt.legend(['yaw $\phi$', 'roll $\\theta$', 'pitch $\psi$'], loc='best')
        #
        #     plt.subplot(5, 1, 4)
        #     plt.plot(sim.trange(), sim.data[probe_x][:,14:17])
        #     plt.ylabel('attitude rate (radians)')
        #     plt.legend(['yaw $\phi$', 'roll $\\theta$', 'pitch $\psi$'], loc='best')
        #
        #     plt.subplot(5, 1, 5)
        #     plt.plot(sim.trange(), sim.data[probe_u])
        #     plt.ylabel('u')
        #     plt.legend(['stroke ampl.', 'pitch torque', 'yaw torque', 'roll'], loc='best')
        #
        # return dict(
        #     x_world=sim.data[probe_x],
        #     u=sim.data[probe_u])

        






    
