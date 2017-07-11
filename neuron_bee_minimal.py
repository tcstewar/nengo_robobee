import sys
sys.path.append('.')
sys.path.append('../PyBee3D_Basic')
import pytry
import numpy as np
import robobee
import scipy
import nengo
from nengo_bee import NengoBee
import seaborn

class GatherDataTrial(pytry.NengoTrial):
    def params(self):
        self.param('velocity', velocity=0.0)
        self.param('angle', angle=0.0)
        self.param('initial pose variability', pose_var=0.0)
        self.param('initial rotation rate variability', dpose_var=0.0)
        self.param('controller filename', ctrl_filename='gather7-hover.npz')
        self.param('time', T=1.0)
        self.param('number of neurons', n_neurons=500)
        self.param('regularization', reg=0.1)
        self.param('radius scaling', radius_scaling=1.0)
        self.param('intercept minimum', low_intercept=-1.0)
        self.param('learning_rate', learning_rate=1e-4)
        self.param('learning adaptation rate', adapt_learn_rate=1e-4)
        self.param('use learning display', use_learning_display=True)
        self.param('apply wing bias', wing_bias=True)
        self.param('adapt Kp scale', adapt_Kp=0.0)
        self.param('adapt Kd scale', adapt_Kd=100.0)
        self.param('adapt Ki scale', adapt_Ki=0.0)

    def model(self, p):

        ctrl = np.load(p.ctrl_filename)

        model = nengo.Network()
        with model:
            pose_offset = np.random.uniform(-p.pose_var, p.pose_var, size=3)
            dpose_offset = np.random.uniform(-p.dpose_var, p.dpose_var, size=3)

            bee = NengoBee(pose_offset=pose_offset, dpose_offset=dpose_offset,
                           random_wing_bias=p.wing_bias)
            
            #keep_x = [9, 15, 17]
            keep_x = [9, 10, 15, 16, 17]
            keep_u = []#1, 3]

            # Use the body frame velocities
            x_vals = ctrl['norm_x_body'][:,keep_x]
            u_vals = ctrl['norm_u'][:,keep_u]

            target = ctrl['all_u']

            pts = np.hstack([x_vals, u_vals])


            D = pts.shape[1]
            ens = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=D,
                                 intercepts=nengo.dists.Uniform(p.low_intercept, 1.0),
                                 neuron_type=nengo.LIF(), radius=np.sqrt(D)*p.radius_scaling)

            u_unfilt = nengo.Node(None, size_in=4)
            u = nengo.Node(None, size_in=4)

            source_body_vel = bee.xyz_rate_body
            source_att = bee.attitude

            adapt_velx = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.LIF())

            vel_learnx_u = nengo.Node(None, size_in=1)
            vel_learnx = nengo.Connection(adapt_velx, vel_learnx_u, learning_rule_type=nengo.PES(p.adapt_learn_rate, pre_tau=0.01),
                                        function=lambda x:0,
                                        synapse=.01)
            nengo.Connection(bee.attitude[2], adapt_velx, synapse=None)

            nengo.Connection(vel_learnx_u, u[1], synapse=None)

            adapt_velz = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.LIF())

            vel_learnz_u = nengo.Node(None, size_in=1)
            vel_learnz = nengo.Connection(adapt_velz, vel_learnz_u, learning_rule_type=nengo.PES(p.adapt_learn_rate, pre_tau=0.01),
                                        function=lambda x:0,
                                        synapse=.01)
            nengo.Connection(vel_learnz_u, u[0], synapse=None)

            adapt_vely = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.LIF())

            vel_learny_u = nengo.Node(None, size_in=1)
            vel_learny = nengo.Connection(adapt_vely, vel_learny_u, learning_rule_type=nengo.PES(p.adapt_learn_rate, pre_tau=0.01),
                                        function=lambda x:0,
                                        synapse=.01)
            nengo.Connection(vel_learny_u, u[3], synapse=None)
            nengo.Connection(bee.attitude[1], adapt_vely, synapse=None)

            if p.adapt_Kp > 0:
                nengo.Connection(source_body_vel[0], vel_learnx.learning_rule, transform=-1*p.adapt_Kp, synapse=None)
            if p.adapt_Kd > 0:
                nengo.Connection(source_body_vel[0], vel_learnx.learning_rule, transform=-1*p.adapt_Kd, synapse=None)
                nengo.Connection(source_body_vel[0], vel_learnx.learning_rule, transform=1*p.adapt_Kd, synapse=0.005)

            #currently using same K's for both dims
            if p.adapt_Kp > 0:
                nengo.Connection(source_body_vel[2], vel_learnz.learning_rule, transform=-1*p.adapt_Kp, synapse=None)
            if p.adapt_Kd > 0:
                nengo.Connection(source_body_vel[2], vel_learnz.learning_rule, transform=-1*p.adapt_Kd, synapse=None)
                nengo.Connection(source_body_vel[2], vel_learnz.learning_rule, transform=1*p.adapt_Kd, synapse=0.005)
            if p.adapt_Kp > 0:
                nengo.Connection(source_body_vel[1], vel_learny.learning_rule, transform=.1*p.adapt_Kp, synapse=None)
            if p.adapt_Kd > 0:
                nengo.Connection(source_body_vel[1], vel_learny.learning_rule, transform=.03*p.adapt_Kd, synapse=None)
                nengo.Connection(source_body_vel[1], vel_learny.learning_rule, transform=-.03*p.adapt_Kd, synapse=0.005)

            nengo.Connection(bee.plant[keep_x], ens[:len(keep_x)], synapse=None, transform=1.0/ctrl['std_x_body'][keep_x])
            if len(keep_u) > 0:
                nengo.Connection(bee.u[keep_u], ens[len(keep_x):], synapse=None, transform=1.0/ctrl['std_u'][keep_u])

            nengo.Connection(nengo.Node(ctrl['mean_x_body']/ctrl['std_x_body'])[keep_x], ens[:len(keep_x)], transform=-1, synapse=None)
            if len(keep_u) > 0:
                nengo.Connection(nengo.Node(ctrl['mean_u']/ctrl['std_u'])[keep_u], ens[len(keep_x):], transform=-1, synapse=None)

            self.conn = nengo.Connection(ens, u_unfilt, eval_points=pts, scale_eval_points=False, function=target,
                             synapse=0.01,
                             solver=nengo.solvers.LstsqL2(reg=p.reg))

            nengo.Connection(u_unfilt[0], u[0], synapse=None)
            nengo.Connection(u_unfilt[1:], u[1:], synapse=None)

            nengo.Connection(u, bee.u, synapse=0)
            
            self.probe_x = nengo.Probe(bee.plant, synapse=0.02)

        if p.gui:
            self.locals = locals()
        return model

    def evaluate(self, p, sim, plt):
        with sim:
            sim.run(p.T)

        t = sim.trange()
        xyz = sim.data[self.probe_x][:,11:14]
        xyz_rate = sim.data[self.probe_x][:,17:20]
        att_rate = sim.data[self.probe_x][:,14:17]

        max_xyz = np.max(np.abs(xyz),axis=0)
        xyz_rate_norm = np.linalg.norm(xyz_rate, axis=1)
        thresholds = [.05,0.07,.1]
        thresh_times = []
        for thresh in thresholds:
            if np.all(xyz_rate_norm<thresh):
                thresh_times.append(0)
            else:
                ind = np.argmax(xyz_rate_norm[::-1]>thresh)
                thresh_times.append(t[::-1][ind])


        att_rate_norm = np.linalg.norm(att_rate, axis=1)
        att_thresholds = [2.5, 5, 10]
        att_thresh_times = []
        for thresh in att_thresholds:
            if np.all(att_rate_norm<thresh):
                att_thresh_times.append(0)
            else:
                ind = np.argmax(att_rate_norm[::-1]>thresh)
                att_thresh_times.append(t[::-1][ind])

        if plt:
            plt.subplot(3, 1, 1)
            plt.plot(t, xyz)
            [plt.axhline(_x) for _x in max_xyz]            
            [plt.axhline(-_x) for _x in max_xyz]            
            plt.ylabel('position (m)')
            plt.legend(['x', 'y', 'z'], loc='best')

            plt.subplot(3, 1, 2)
            plt.plot(t, xyz_rate)
            plt.plot(t, xyz_rate_norm)
            [plt.axvline(_x) for _x in thresh_times]            
            [plt.axhline(_x) for _x in thresholds]            
            plt.ylabel('velocity (m)')
            plt.legend(['x', 'y', 'z'], loc='best')

            plt.subplot(3, 1, 3)
            plt.plot(t, att_rate)
            plt.plot(t, att_rate_norm)
            [plt.axvline(_x) for _x in att_thresh_times]            
            [plt.axhline(_x) for _x in att_thresholds]            
            plt.ylabel('attitude rate (radians)')
            plt.legend(['yaw $\phi$', 'roll $\\theta$', 'pitch $\psi$'], loc='best')

        return dict(
            max_xyz=max_xyz,
            thresh_times=thresh_times,
            att_thresh_times=att_thresh_times,
            thresholds=thresholds,
            att_thresholds=att_thresholds,
           )

        






    
