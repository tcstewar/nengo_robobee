import sys
import scipy
import nengo
import numpy as np
from os import path
# sys.path.append('.')
sys.path.append(r'..\PyBee3D\PyBee3D')
import robobee
from pif_control import PIFControl

class NengoBee(nengo.Network):
    def __init__(self, label=None,
                 pose_offset=(0, 0, 0),
                 dpose_offset=(0, 0, 0),
                 vel_offset=(0, 0, 0),
                 random_wing_bias=False,
                 v_wind=0.0,
                 phi_0=0.0,
                 actuator_failure=False,
                 sample_dt=1E-3,
                 y_star_init=np.array([0, 0, 0, 0])):
        super(NengoBee, self).__init__(label=label)

        self.v_wind=v_wind
        self.sample_dt = sample_dt

        # self.data_buffer = []

        self.bee = robobee.RoboBee(random_wing_bias=random_wing_bias,
                                   actuator_failure=actuator_failure,
                                   a_theta=-0.2, b_theta=0.04,
                                   new_yaw_control=True)
        self.pif_controller = PIFControl(self.bee)

        # if random_wing_bias == True:
        #     self.bee.ROLL_BIAS = 1.0
        #     self.bee.PITCH_BIAS = 10.0

        x, self.u_0 = self.get_initial_set_point(y_star_init)

        x[8:11] += pose_offset
        x[14:17] += dpose_offset
        x[17:20] += vel_offset
        x[8] = phi_0

        self.integrator_dynamics = scipy.integrate.ode(self.bee.get_dynamics).set_integrator('dopri5')
        self.integrator_dynamics.set_initial_value(x, 0)
        self.integrator_dynamics.set_f_params(self.u_0, np.array([0, 0, 0]))

        with self:
            self.plant_unfilt = nengo.Node(self.update, size_in=len(self.u_0))

            self.plant = nengo.Node(None, size_in=20)
            nengo.Connection(self.plant_unfilt, self.plant, synapse=None)

            self.u = nengo.Node(None, size_in=self.plant_unfilt.size_in)
            nengo.Connection(self.u, self.plant_unfilt, synapse=None)

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

            def rotate_full(t, x):
                body_x = self.bee.world_state_to_body(x)
                return body_x

            self.sampled_body_state = np.zeros(20)
            self.sampled_body_vel = np.zeros(3)
            self.sampled_t = [0, 0]

            def sample_body_state(t, x):
                # return x
                if self.sampled_t[0] <= t:
                    self.sampled_body_state = x
                    self.sampled_t[0] += max(self.sample_dt, t - self.sampled_t[0])
                return self.sampled_body_state

            def sample_body_vel(t, x):
                # return x
                if self.sampled_t[1] <= t:
                    self.sampled_body_vel = x
                    self.sampled_t[1] += max(self.sample_dt, t - self.sampled_t[1])
                return self.sampled_body_vel
            #
            # def delayed_state(t, x):
            #     tau = 0
            #     while len(self.data_buffer) > 0 and tau < t - self.delay:
            #         tau = self.data_buffer

            self.xyz_rate_body = nengo.Node(rotate, size_in=20)
            nengo.Connection(self.plant, self.xyz_rate_body, synapse=None)

            self.x_body = nengo.Node(rotate_full, size_in=20)
            nengo.Connection(self.plant, self.x_body, synapse=None)

            self.x_body_sampled = nengo.Node(sample_body_state, size_in=20)
            nengo.Connection(self.x_body, self.x_body_sampled, synapse=None)

            self.body_vel_sampled = nengo.Node(sample_body_vel, size_in=3)
            nengo.Connection(self.xyz_rate_body, self.body_vel_sampled, synapse=None)

    def update(self, t, u):
        if t > 0:
            self.integrator_dynamics.set_f_params(u, np.array([self.v_wind, 0, 0]))
            x = self.integrator_dynamics.integrate(t)
        else:
            x = np.zeros(20)
        # self.data_buffer.append({'data': x, 't': t})
        # self.update_delayed_state(

        return x

    def get_initial_set_point(self, y_star):
        folder_name = r'..\PyBee3D\PyBee3D\Saved Data\Maneuvers\Flight_Envelope\New_Yaw'
        file_name = 'vel{0:3.1f}_turn{1:3.1f}_climb{2:3.1f}_slip{3:3.1f}'.format(y_star[0], y_star[1], y_star[2], y_star[3])
        file_path = path.join(folder_name, file_name)
        data = scipy.io.loadmat(file_path)
        x_star = data['x_0'].flatten()
        u_star = data['u_0'].flatten()
        # u_star = self.pif_controller.controller.get_u_star(y_star)
        return x_star, u_star
