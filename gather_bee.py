import sys
sys.path.append('.')
sys.path.append(r'C:\Users\taylo\OneDrive\Cornell\LISC\Code\RoboBee\PyBee3D_Basic')
import pytry
import numpy as np
import nengo
from pif_control import PIFControl
from nengo_bee import NengoBee

class GatherDataTrial(pytry.PlotTrial):
    def params(self):
        self.param('velocity', velocity=0.0)
        self.param('angle', angle=0.0)
        self.param('initial pose variability', pose_var=0.0)
        self.param('initial rotation rate variability', dpose_var=0.0)
        self.param('time', T=1.0)
        self.param('dt', dt=0.001)

    def evaluate(self, p, plt):

        model = nengo.Network()
        with model:
            pose_offset = np.random.uniform(-p.pose_var, p.pose_var, size=3)
            dpose_offset = np.random.uniform(-p.dpose_var, p.dpose_var, size=3)

            bee = NengoBee(pose_offset=pose_offset, dpose_offset=dpose_offset)
            
            control = PIFControl(bee.bee)
            
            nengo.Connection(bee.plant, control.x, synapse=None)
            nengo.Connection(control.control, bee.u, synapse=0)
            nengo.Connection(bee.u, control.u, synapse=None)

            v = nengo.Node(p.velocity)
            nengo.Connection(v, control.y_star[0], synapse=None)
            a = nengo.Node(p.angle)
            nengo.Connection(a, control.y_star[2], synapse=None)

            probe_x = nengo.Probe(control.x, synapse=None)
            probe_u = nengo.Probe(bee.u, synapse=None)
            
        sim = nengo.Simulator(model, dt=p.dt)
        with sim:
            sim.run(p.T)

        if plt:
            plt.subplot(5, 1, 1)
            plt.plot(sim.trange(), sim.data[probe_x][:,11:14])
            plt.ylabel('position (m)')
            plt.legend(['x', 'y', 'z'], loc='best')

            plt.subplot(5, 1, 2)
            plt.plot(sim.trange(), sim.data[probe_x][:,17:20])
            plt.ylabel('velocity (m)')
            plt.legend(['x', 'y', 'z'], loc='best')

            plt.subplot(5, 1, 3)
            plt.plot(sim.trange(), sim.data[probe_x][:,8:11])
            plt.ylabel('attitude (radians)')
            plt.legend(['yaw $\phi$', 'roll $\\theta$', 'pitch $\psi$'], loc='best')

            plt.subplot(5, 1, 4)
            plt.plot(sim.trange(), sim.data[probe_x][:,14:17])
            plt.ylabel('attitude rate (radians)')
            plt.legend(['yaw $\phi$', 'roll $\\theta$', 'pitch $\psi$'], loc='best')

            plt.subplot(5, 1, 5)
            plt.plot(sim.trange(), sim.data[probe_u])
            plt.ylabel('u')
            plt.legend(['stroke ampl.', 'pitch torque', 'yaw torque', 'roll'], loc='best')

        return dict(
            x_world=sim.data[probe_x],
            u=sim.data[probe_u])

        






    
