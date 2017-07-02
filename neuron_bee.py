import sys
sys.path.append('.')
import pytry
import numpy as np
import robobee
import scipy
import nengo
from controllers.pif.pif_compensator import PIFCompensator
import seaborn

class PIFControl(nengo.Network):
    def __init__(self, bee, label=None):
        self.bee = bee
        super(PIFControl, self).__init__(label=label)
        self.controller = PIFCompensator()

        def control_ode_fun(t, eta, y_star, x, u, bee):
            return self.controller.get_control_dynamics(eta, t, y_star, x, u, bee)
        self.integrator_control = scipy.integrate.ode(control_ode_fun).set_integrator('dopri5')
        eta = np.zeros(7)
        self.integrator_control.set_initial_value(eta, 0)
        
        with self:
            self.control = nengo.Node(self.update, size_in=28)
            
            self.y_star = nengo.Node(None, size_in=4)
            self.x = nengo.Node(None, size_in=20)
            self.u = nengo.Node(None, size_in=4)
            nengo.Connection(self.y_star, self.control[:4], synapse=None)
            nengo.Connection(self.x, self.control[4:24], synapse=None)
            nengo.Connection(self.u, self.control[24:], synapse=None)
            
    def update(self, t, v):
        y_star, x, u = v[:4], v[4:24], v[24:28]
        self.integrator_control.set_f_params(y_star, x, u, self.bee)
        eta = self.integrator_control.integrate(t)
        u = eta[:4]
        return u
        

class NengoBee(nengo.Network):
    def __init__(self, label=None, pose_offset=(0,0,0), dpose_offset=(0,0,0)):
        super(NengoBee, self).__init__(label=label)

        self.bee = robobee.RoboBee()

        traj_data = scipy.io.loadmat('Hover_Data.mat')
        x = traj_data['x'][0]

        x[8:11] += pose_offset
        x[14:17] += dpose_offset

        self.integrator_dynamics = scipy.integrate.ode(self.bee.get_dynamics).set_integrator('dopri5')
        self.integrator_dynamics.set_initial_value(x, 0)

        self.u_0 = traj_data['u'][0]

        with self:
            
            self.plant = nengo.Node(self.update, size_in=len(self.u_0))
            
            self.u = nengo.Node(None, size_in=self.plant.size_in)
            nengo.Connection(self.u, self.plant, synapse=None)

            self.xyz = nengo.Node(None, size_in=3)
            nengo.Connection(self.plant[11:14], self.xyz, synapse=None)

            self.attitude = nengo.Node(None, size_in=3)
            nengo.Connection(self.plant[8:11], self.attitude, synapse=None)


    def update(self, t, u):
        self.integrator_dynamics.set_f_params(u)
        x = self.integrator_dynamics.integrate(t)
        return x

class GatherDataTrial(pytry.PlotTrial):
    def params(self):
        self.param('velocity', velocity=0.0)
        self.param('angle', angle=0.0)
        self.param('initial pose variability', pose_var=0.0)
        self.param('initial rotation rate variability', dpose_var=0.0)
        self.param('controller filename', ctrl_filename='gather1-hover.npz')
        self.param('time', T=1.0)
        self.param('dt', dt=0.001)
        self.param('number of neurons', n_neurons=500)
        self.param('regularization', reg=0.1)

    def evaluate(self, p, plt):

        ctrl = np.load(p.ctrl_filename)

        model = nengo.Network()
        with model:


            pose_offset = np.random.uniform(-p.pose_var, p.pose_var, size=3)
            dpose_offset = np.random.uniform(-p.dpose_var, p.dpose_var, size=3)

            bee = NengoBee(pose_offset=pose_offset, dpose_offset=dpose_offset)
            
            control = PIFControl(bee.bee)

            ens = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=23,
                                 neuron_type=nengo.LIFRate(), radius=np.sqrt(23))

            

            u = nengo.Node(None, size_in=4)
            nengo.Connection(u, u, synapse=0)
            nengo.Connection(u, bee.u, synapse=0)



            nengo.Connection(bee.plant, ens[:20], synapse=None, transform=1.0/ctrl['std_x'])
            nengo.Connection(u[[0,1,3]], ens[20:], synapse=None, transform=1.0/ctrl['std_x'][[0,1,3]])

            nengo.Connection(nengo.Node(ctrl['mean_x']), ens[:20], transform=-1, synapse=None)
            nengo.Connection(nengo.Node(ctrl['mean_u'][[0,1,3]]), ens[20:], transform=-1, synapse=None)

            conn = nengo.Connection(ens, u[[0,1,3]], eval_points=ctrl['pts'], scale_eval_points=False, function=ctrl['fn'],
                             transform=ctrl['mean_du'][[0,1,3]]*0.001,
                             solver=nengo.solvers.LstsqL2(reg=p.reg))




            
            nengo.Connection(bee.plant, control.x, synapse=None)
            #nengo.Connection(control.control, bee.u, synapse=0)
            nengo.Connection(bee.u, control.u, synapse=None)

            v = nengo.Node(p.velocity)
            nengo.Connection(v, control.y_star[0], synapse=None)
            a = nengo.Node(p.angle)
            nengo.Connection(a, control.y_star[2], synapse=None)

            probe_pif_u = nengo.Probe(control.control, synapse=None)
            probe_x = nengo.Probe(control.x, synapse=None)
            probe_u = nengo.Probe(bee.u, synapse=None)
            
        sim = nengo.Simulator(model, dt=p.dt)
        with sim:
            sim.run(p.T)

        if plt:
            plt.subplot(4, 2, 1)
            plt.plot(sim.trange(), sim.data[probe_x][:,11:14])
            plt.ylabel('position (m)')
            plt.legend(['x', 'y', 'z'], loc='best')

            plt.subplot(4, 2, 3)
            plt.plot(sim.trange(), sim.data[probe_x][:,17:20])
            plt.ylabel('velocity (m)')
            plt.legend(['x', 'y', 'z'], loc='best')

            plt.subplot(4, 2, 5)
            plt.plot(sim.trange(), sim.data[probe_x][:,8:11])
            plt.ylabel('attitude (radians)')
            plt.legend(['yaw $\phi$', 'roll $\\theta$', 'pitch $\psi$'], loc='best')

            plt.subplot(4, 2, 7)
            plt.plot(sim.trange(), sim.data[probe_x][:,14:17])
            plt.ylabel('attitude rate (radians)')
            plt.legend(['yaw $\phi$', 'roll $\\theta$', 'pitch $\psi$'], loc='best')

            plt.subplot(4, 2, 2)
            plt.plot(sim.trange(), sim.data[probe_u])
            plt.ylabel('u')
            plt.legend(['stroke ampl.', 'pitch torque', 'yaw torque', 'roll'], loc='best')

            plt.subplot(4, 2, 4)
            plt.plot(sim.trange(), sim.data[probe_pif_u])
            plt.ylabel('PIF u')
            plt.legend(['stroke ampl.', 'pitch torque', 'yaw torque', 'roll'], loc='best')

            plt.subplot(4, 2, 6)
            plt.plot(sim.trange(), sim.data[probe_u] - sim.data[probe_pif_u])
            plt.ylabel('u error')
            plt.legend(['stroke ampl.', 'pitch torque', 'yaw torque', 'roll'], loc='best')

        return dict(
            solver_error = sim.data[conn].solver_info['rmses'],
            x=sim.data[probe_x],
            u=sim.data[probe_u],
            pif_u=sim.data[probe_pif_u])

        






    
