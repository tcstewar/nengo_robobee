# I'm going to try using an SNN to approximate just the feedforward part of the PIF Compensator, hopefully it can do this
# better than it approximates the gains with the current training strategy...

import sys
sys.path.append('.')
sys.path.append(r'..\PyBee3D\PyBee3D')
import pytry
import numpy as np
import robobee
import scipy
from scipy import integrate
import nengo
from pif_control import PIFControl
from nengo_bee import NengoBee
import seaborn as sns
from itertools import product

class GatherDataTrial(pytry.NengoTrial):
    def params(self):
        self.param('Simulation time', T=1.0)
        self.param('Target velocity', velocity=0.0)
        self.param('Target climb angle', climb_angle=0.0)

    def model(self, p):
        bee = NengoBee()
        pif = PIFControl(bee.bee)

        vel_pts = [0, 0.5, 1.0]
        turn_rate_pts = [0]
        angle_pts = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
        sideslip_pts = [0]

        # get the feedforward control input at different points
        y_star_pts = list(product(vel_pts, turn_rate_pts, angle_pts, sideslip_pts))
        u_star_pts = [pif.controller.get_u_star(y_star) for y_star in y_star_pts]

        y_star_pts = np.array(y_star_pts)
        u_star_pts = np.array(u_star_pts)

        mean_y_star = np.mean(y_star_pts, axis=0)
        std_y_star = np.std(y_star_pts, axis=0)
        mean_u_star = np.mean(u_star_pts, axis=0)
        std_u_star = np.std(u_star_pts, axis=0)
        norm_y_star = (y_star_pts - mean_y_star) / std_y_star
        norm_u_star = (u_star_pts - mean_u_star) / std_u_star

        model = nengo.Network()
        with model:
            keep_y_star = [0, 2]

            y_star_vals = norm_y_star[:, keep_y_star]
            target_u = norm_u_star

            y_star = nengo.Node(np.array([p.velocity, 0, p.climb_angle, 0]))

            D = y_star_vals.shape[1]
            radius_scaling = 1.0
            reg = 0.1
            feedforward_ens = nengo.Ensemble(n_neurons=500, dimensions=D,
                                             intercepts=nengo.dists.Uniform(-1.0, 1.0),
                                             neuron_type=nengo.LIF(), radius=np.sqrt(D) * radius_scaling)

            nengo.Connection(y_star[keep_y_star], feedforward_ens, synapse=None, transform=1.0 /std_y_star[keep_y_star])
            nengo.Connection(nengo.Node((mean_y_star / std_y_star)[keep_y_star]), feedforward_ens, transform=-1, synapse=None)

            u_unfilt = nengo.Node(None, size_in=4)

            self.conn = nengo.Connection(feedforward_ens, u_unfilt,
                                         eval_points=y_star_vals,
                                         scale_eval_points=False,
                                         function=u_star_pts,
                                         synapse=0.02,
                                         learning_rule_type=None,
                                         solver=nengo.solvers.LstsqL2(reg=reg))

            self.probe_u = nengo.Probe(u_unfilt, synapse=None)
        return model

    def evaluate(self, p, sim, plt):
        # Now evaluate how good the thing is and plot the results
        with sim:
            sim.run(p.T)

        return dict(
            solver_error=sim.data[self.conn].solver_info['rmses'],
            u=sim.data[self.probe_u])