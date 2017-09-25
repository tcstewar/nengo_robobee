import sys
# sys.path.append('.')
sys.path.append(r'C:\Users\taylo\OneDrive\Cornell\LISC\Code\RoboBee\PyBee3D\PyBee3D')
import numpy as np
import scipy
import nengo
from controllers.pif.pif_compensator import PIFCompensator


class PIFControl(nengo.Network):
    def __init__(self, bee, label=None):
        self.bee = bee
        super(PIFControl, self).__init__(label=label)
        self.controller = PIFCompensator()

        def control_ode_fun(t, eta, y_star, x, u, bee):
            return self.controller.get_control_dynamics(eta, t, y_star, x, u, bee)

        self.integrator_control = scipy.integrate.ode(control_ode_fun).set_integrator('dopri5')
        eta = np.zeros(7)
        eta[:4] = self.controller.get_u_star([0, 0, 0, 0])
        self.integrator_control.set_initial_value(eta, 0)

        with self:
            self.control = nengo.Node(self.update, size_in=28)

            self.y_star = nengo.Node(None, size_in=4)
            self.x = nengo.Node(None, size_in=20)
            self.u = nengo.Node(None, size_in=4)
            def get_u_dot(t, v): return self.u_dot
            self.u_dot_node = nengo.Node(get_u_dot, size_in=4)
            nengo.Connection(self.y_star, self.control[:4], synapse=None)
            nengo.Connection(self.x, self.control[4:24], synapse=None)
            nengo.Connection(self.u, self.control[24:], synapse=None)
            # nengo.Connection(self.u_dot_node, self.u_dot, synapse=None)

    def update(self, t, v):
        y_star, x, u = v[:4], v[4:24], v[24:28]
        self.integrator_control.set_f_params(y_star, x, u, self.bee)
        eta = self.integrator_control.integrate(t)
        self.u_dot = self.controller.get_control_dynamics(eta, t, y_star, x, u, self.bee)[:4]
        u = eta[:4]
        return u
    #
    # def update(self, t, v):
    #     y_star, x, u = v[:4], v[4:24], v[24:28]
    #     self.integrator_control.set_f_params(y_star, x, u, self.bee)
    #     eta = self.integrator_control.integrate(t)
    #     # u = eta[:4]
    #     xi = eta[4:]
    #
    #     eta = np.concatenate((u, xi))
    #
    #     eta_dot = self.controller.get_control_dynamics(eta, t, y_star, x, u, self.bee)
    #     u_dot = eta_dot[:4]
    #
    #     return u_dot