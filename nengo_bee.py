import sys
# sys.path.append('.')
sys.path.append(r'C:\Users\taylo\OneDrive\Cornell\LISC\Code\RoboBee\PyBee3D_Basic')
import robobee
import scipy
import nengo

class NengoBee(nengo.Network):
    def __init__(self, label=None, pose_offset=(0, 0, 0), dpose_offset=(0, 0, 0)):
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

            self.xyz_rate = nengo.Node(None, size_in=3)
            nengo.Connection(self.plant[17:20], self.xyz_rate, synapse=None)

            self.attitude_rate = nengo.Node(None, size_in=3)
            nengo.Connection(self.plant[14:17], self.attitude_rate, synapse=None)

            def rotate(t, x):
                body_x = self.bee.world_state_to_body(x)
                return body_x[17:20]

            self.xyz_rate_body = nengo.Node(rotate, size_in=20)
            nengo.Connection(self.plant, self.xyz_rate_body, synapse=None)

    def update(self, t, u):
        self.integrator_dynamics.set_f_params(u)
        x = self.integrator_dynamics.integrate(t)
        return x