import numpy as np
import robobee
import scipy
import nengo
from controllers.pif.pif_compensator import PIFCompensator

class PIFControl(nengo.Network):
    def __init__(self, bee, label=None):
        self.bee = bee
        super(PIFControl, self).__init__(label=label)
        self.controller = PIFCompensator()
        #self.controller.dt = 1e-4

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
    def __init__(self, label=None):
        super(NengoBee, self).__init__(label=label)

        self.bee = robobee.RoboBee()

        traj_data = scipy.io.loadmat('Hover_Data.mat')
        x = traj_data['x'][0]

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


model = nengo.Network()
with model:

    bee = NengoBee()
    
    control = PIFControl(bee.bee)
    
    nengo.Connection(bee.plant, control.x, synapse=None)
    nengo.Connection(control.control, bee.u, synapse=0)
    nengo.Connection(bee.u, control.u, synapse=None)
    
