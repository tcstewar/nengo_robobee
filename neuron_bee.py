import sys
sys.path.append('.')
sys.path.append('../PyBee3D_Basic')
import pytry
import numpy as np
import robobee
import scipy
import nengo
# from controllers.pif.pif_compensator import PIFCompensator
from pif_control import PIFControl
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
        self.param('use PIF control', use_pif=False)
        self.param('radius scaling', radius_scaling=1.0)
        self.param('oracle learning', oracle=False)
        self.param('intercept minimum', low_intercept=-1.0)
        self.param('learning_rate', learning_rate=1e-4)
        self.param('Ki', Ki=0.0)
        self.param('use world frame for adaptation', world_adapt=True)
        self.param('use learning adaptation', adapt=True)
        self.param('learning adaptation rate', adapt_learn_rate=1e-4)
        self.param('use learning display', use_learning_display=True)
        self.param('apply wing bias', wing_bias=True)
        self.param('adapt Kp scale', adapt_Kp=0.0)
        self.param('adapt Kd scale', adapt_Kd=100.0)
        self.param('adapt Ki scale', adapt_Ki=0.0)
        self.param('parametric disturbance learning', pdl=False)
        self.param('pdl derivative tau', pdl_tau=0.01)
        self.param('pdl learning rate', pdl_learning_rate=1e-5)

        

    def model(self, p):

        ctrl = np.load(p.ctrl_filename)

        model = nengo.Network()
        with model:
            pose_offset = np.random.uniform(-p.pose_var, p.pose_var, size=3)
            dpose_offset = np.random.uniform(-p.dpose_var, p.dpose_var, size=3)

            bee = NengoBee(pose_offset=pose_offset, dpose_offset=dpose_offset,
                           random_wing_bias=p.wing_bias)
            
            control = PIFControl(bee.bee)

            #keep_x = [8, 9, 10, 14, 15, 16, 17, 18, 19]
            #keep_x = [9, 15, 17]
            keep_x = [9, 10, 15, 16, 17]
            keep_u = []#1, 3]

            # x_vals = ctrl['norm_x'][:,keep_x]
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

            if p.Ki > 0:
                if p.world_adapt:
                    source_body_vel = bee.xyz_rate
                else:
                    source_body_vel = bee.xyz_rate_body
                #dz_error = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.LIF())
                #nengo.Connection(source_body_vel[2], dz_error, synapse=None)
                #nengo.Connection(dz_error, dz_error, synapse=0.1)
                #nengo.Connection(dz_error, u[0], transform=p.Ki, synapse=.01)
                
                '''roll_error = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.LIF())
                nengo.Connection(bee.plant[bee.bee.idx_body_att[-2]], roll_error, synapse=None)
                nengo.Connection(roll_error, roll_error, synapse=0.1)
                nengo.Connection(roll_error, u[3], transform=.05*p.Ki, synapse=.01)'''

                '''dpitch_error = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.LIF())
                nengo.Connection(bee.plant[bee.bee.idx_body_att_rate[-1]], dpitch_error, synapse=None)
                nengo.Connection(dpitch_error, dpitch_error, synapse=0.1)
                nengo.Connection(dpitch_error, u[2], transform=.3*p.Ki, synapse=.01)'''
                
                dx_error = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.LIF())
                nengo.Connection(source_body_vel[0], dx_error, synapse=None)
                nengo.Connection(dx_error, dx_error, synapse=0.1)
                nengo.Connection(dx_error, bee.u[1], transform=.01*p.Ki, synapse=.01)

                '''yaw_error = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.LIF())
                nengo.Connection(bee.plant[bee.bee.idx_body_att[-2]], yaw_error, synapse=None)
                nengo.Connection(yaw_error, yaw_error, synapse=0.1)
                nengo.Connection(yaw_error, u[3], transform=.05*p.Ki, synapse=.01)'''

            if p.adapt:
                if p.world_adapt:
                    source_body_vel = bee.xyz_rate
                else:
                    source_body_vel = bee.xyz_rate_body
                source_att = bee.attitude

                t_inhibit = 1.0

                def inhibit(t, x):
                    if t < t_inhibit:
                        return np.zeros(np.shape(x))
                    else:
                        return x
                    # return 2.0 if t < 1.0 else 0.0

                learn_vel_switch = nengo.Node(inhibit, size_in=3, size_out=3)
                # learn_att_switch = nengo.Node(inhibit, size_in=2)
                adapt_vel = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.LIF())
                # adapt_att = nengo.Ensemble(n_neurons=100, dimensions=2, neuron_type=nengo.LIF())
                # nengo.Connection(learn_vel_switch, adapt_vel.neurons, transform=[[-1]]*adapt_vel.n_neurons)
                
                #learn the adaptation for dz to thrust
                # dz_learn = nengo.Connection(adapt_vel, u[0],
                #     learning_rule_type=nengo.PES(p.adapt_learn_rate),
                #     function=lambda x:0,
                #     synapse=.01)
                # nengo.Connection(source_body_vel[2], dz_learn.learning_rule, transform=-1)
                #learn the adapatation for dx to pitch
                vel_learn_u = nengo.Node(None, size_in=1)
                vel_learn = nengo.Connection(adapt_vel, vel_learn_u, learning_rule_type=nengo.PES(p.adapt_learn_rate, pre_tau=0.01),
                                            function=lambda x:0,
                                            synapse=.01)
                nengo.Connection(bee.attitude[2], adapt_vel, synapse=None)

                nengo.Connection(vel_learn_u, u[1], synapse=None)
                #nengo.Connection(vel_learn_u, vel_learn.learning_rule, transform=0.5, synapse=0)
                # att_learn = nengo.Connection(adapt_att, u[[1, 3]], learning_rule_type=nengo.PES(.5*p.adapt_learn_rate),
                #                             function=lambda x:[0,0],
                #                             transform=[-1]*2,
                #                             synapse=.01)
                # nengo.Connection(source_body_vel, learn_vel_switch)
                # nengo.Connection(learn_vel_switch, adapt_vel.neurons, transform=[[-1, -1, -1]]*adapt_vel.n_neurons)
                if p.adapt_Kp > 0:
                    nengo.Connection(source_body_vel[0], vel_learn.learning_rule, transform=-1*p.adapt_Kp, synapse=None)
                if p.adapt_Kd > 0:
                    nengo.Connection(source_body_vel[0], vel_learn.learning_rule, transform=-1*p.adapt_Kd, synapse=None)
                    nengo.Connection(source_body_vel[0], vel_learn.learning_rule, transform=1*p.adapt_Kd, synapse=0.005)
                if p.adapt_Ki > 0:
                    nengo.Connection(bee.xyz[0], vel_learn.learning_rule, transform=1*p.adapt_Ki, synapse=None)
                #learn adaptation for dy to roll
                '''dy_learn = nengo.Connection(adapt_vel, u[3], 
                    learning_rule_type=nengo.PES(.1*p.adapt_learn_rate), 
                    function=lambda x:0, 
                    synapse=.01)
                nengo.Connection(source_body_vel[1], dy_learn.learning_rule, transform=1)'''
                # nengo.Connection(bee.plant[bee.bee.idx_body_att[1]], adapt_vel, synapse=None)

                inhibit_adapt = nengo.Node(0)
                nengo.Connection(inhibit_adapt, adapt_vel.neurons, transform=-10*np.ones((adapt_vel.n_neurons, 1)))

            if p.pdl:
                pdl_state = nengo.Ensemble(n_neurons=100, dimensions=1)
                pdl_u = nengo.Node((lambda t, x: np.clip(x, -0.2, 0.2)), size_in=1)
                pdl_conn = nengo.Connection(pdl_state, pdl_u,
                                            function=lambda x: 0,
                                            synapse=0.01,
                                            learning_rule_type=nengo.PES(p.pdl_learning_rate),
                                            )

                tau = 0.01



                ddpsi = nengo.Node(None, size_in=1)
                f_dpsi = nengo.Node(None, size_in=1)
                nengo.Connection(bee.attitude_rate[2], f_dpsi, synapse=tau)

                nengo.Connection(f_dpsi, ddpsi, synapse=None, transform=1.0/p.pdl_tau)
                nengo.Connection(f_dpsi, ddpsi, synapse=p.pdl_tau, transform=-1.0/p.pdl_tau)

                error = nengo.Node(None, size_in=1)
                nengo.Connection(pdl_u, error, synapse=tau)
                nengo.Connection(ddpsi, error, transform=-1.0/0.1282, synapse=tau)
                nengo.Connection(bee.attitude[2], error, synapse=tau, transform=-46.2811)
                nengo.Connection(bee.attitude_rate[2], error, synapse=tau, transform=-19.6916)
                nengo.Connection(bee.xyz_rate_body[0], error, synapse=tau, transform=915.22)

                nengo.Connection(error, pdl_conn.learning_rule, synapse=None, transform=-1)

                '''
                ideal_ddpsi = nengo.Node(None, size_in=1)
                nengo.Connection(bee.attitude[2], ideal_ddpsi, synapse=tau, transform=-46.2811)
                nengo.Connection(bee.attitude_rate[2], ideal_ddpsi, synapse=tau, transform=-19.6916)
                nengo.Connection(bee.xyz_rate_body[0], ideal_ddpsi, synapse=tau, transform=915.22)
                if p.use_pif:
                    nengo.Connection(control.control[1], ideal_ddpsi, synapse=tau, transform=0.1282)
                else:
                    nengo.Connection(u[1], ideal_ddpsi, synapse=tau, transform=0.1282)

                both_ddpsi = nengo.Node(None, size_in=2)
                nengo.Connection(ideal_ddpsi, both_ddpsi[0], synapse=None)
                nengo.Connection(ddpsi, both_ddpsi[1], synapse=None)

                nengo.Connection(ideal_ddpsi, pdl_conn.learning_rule, synapse=None, transform=-1)
                nengo.Connection(ddpsi, pdl_conn.learning_rule, synapse=None, transform=1)
                '''
                #nengo.Connection(pdl_u, bee.u[1], synapse=None)


            nengo.Connection(bee.plant[keep_x], ens[:len(keep_x)], synapse=None, transform=1.0/ctrl['std_x_body'][keep_x])
            if len(keep_u) > 0:
                nengo.Connection(bee.u[keep_u], ens[len(keep_x):], synapse=None, transform=1.0/ctrl['std_u'][keep_u])

            nengo.Connection(nengo.Node(ctrl['mean_x_body']/ctrl['std_x_body'])[keep_x], ens[:len(keep_x)], transform=-1, synapse=None)
            if len(keep_u) > 0:
                nengo.Connection(nengo.Node(ctrl['mean_u']/ctrl['std_u'])[keep_u], ens[len(keep_x):], transform=-1, synapse=None)

            if p.oracle:
                learning_rule=nengo.PES(learning_rate=p.learning_rate)
            else:
                learning_rule=None
            self.conn = nengo.Connection(ens, u_unfilt, eval_points=pts, scale_eval_points=False, function=target,
                             synapse=0.01,
                             learning_rule_type=learning_rule,
                             solver=nengo.solvers.LstsqL2(reg=p.reg))

            nengo.Connection(u_unfilt[0], u[0], synapse=None)
            nengo.Connection(u_unfilt[1:], u[1:], synapse=None)


            if p.use_pif:
                nengo.Connection(control.control, bee.u, synapse=0)
            else:
                nengo.Connection(u, bee.u, synapse=0)
            
            nengo.Connection(bee.plant, control.x, synapse=None)
            nengo.Connection(bee.u, control.u, synapse=None)

            v = nengo.Node(p.velocity)
            nengo.Connection(v, control.y_star[0], synapse=None)
            a = nengo.Node(p.angle)
            nengo.Connection(a, control.y_star[2], synapse=None)

            self.probe_pif_u = nengo.Probe(control.control, synapse=None)
            self.probe_x = nengo.Probe(control.x, synapse=None)
            self.probe_u = nengo.Probe(u, synapse=None)


            plot_thrust = nengo.Node(None, size_in=2)
            nengo.Connection(control.control[0], plot_thrust[0], synapse=None)
            nengo.Connection(u[0], plot_thrust[1], synapse=None)

            plot_pitch = nengo.Node(None, size_in=2)
            nengo.Connection(control.control[1], plot_pitch[0], synapse=None)
            nengo.Connection(u[1], plot_pitch[1], synapse=None)

            plot_roll = nengo.Node(None, size_in=2)
            nengo.Connection(control.control[3], plot_roll[0], synapse=None)
            nengo.Connection(u[3], plot_roll[1], synapse=None)

            if p.oracle:
                def error_func(t, x):
                    if x[-1] > 0.5:
                        return x[:-1] * 0
                    else:
                        return x[:-1]
                error = nengo.Node(error_func, size_in=5)
                nengo.Connection(control.control, error[:4], synapse=None, transform=-1)
                nengo.Connection(u, error[:4], synapse=None)
                nengo.Connection(error, self.conn.learning_rule, synapse=0)

                stop_learning = nengo.Node(0)
                nengo.Connection(stop_learning, error[-1], synapse=None)

            if p.use_learning_display:
                import nengo_learning_display

                learn_plots = []

                if p.adapt:

                    vel_plot = nengo_learning_display.Plot1D(vel_learn, np.linspace(-1,1,30), range=(-1,1))
                    vel_plot.label = 'adapting for velocity'
                    learn_plots.append(vel_plot)

                def on_step(sim):
                    for p in learn_plots:
                        p.update(sim)
            

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

        return dict(
            solver_error = sim.data[self.conn].solver_info['rmses'],
            x=sim.data[self.probe_x],
            u=sim.data[self.probe_u],
            pif_u=sim.data[self.probe_pif_u])

        






    
