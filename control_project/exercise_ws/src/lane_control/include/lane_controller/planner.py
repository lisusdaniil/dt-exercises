import numpy as np
import dubins
import scipy.optimize as sciopt

class LanePlanner:
    """
    

    Args:
        

    """
    def __init__(self, parameters):
        self.params = parameters

    def update_parameters(self, parameters):
        """Updates parameters of LaneController object.

            Args:
                parameters (:obj:`dict`): dictionary containing the new parameters for LaneController object.
        """
        self.params = parameters

    def get_new_path(self, x_curr, x_targ):
        """

        Args:
            
        Returns:
            
        """
        # Get maximum radius of curvature
        dt = self.params['~dt_path']
        max_dist_fact = self.params['~max_dist_fact']
        min_r = self.params['~min_r']
        max_r = self.params['~max_r']
        bnds = ((min_r, max_r),)    # Bounds on curvature

        dist_min = np.sqrt((x_curr[0]-x_targ[0])**2 + (x_curr[1]-x_targ[1])**2)
        dist_max = max_dist_fact*dist_min

        con = {'type': 'ineq', 'fun': self.cons_fun, 'args': (x_curr, x_targ, dt, dist_max)}
        res = sciopt.minimize(self.opt_rad, min_r, args=(x_curr, x_targ, dt), bounds=bnds, constraints=con)
        max_r = res.x

        path, u, dist = self.gen_path(x_curr, x_targ, max_r, dt)
        return path, u, dist

    def gen_path(self, q0, q1, r, dt):
        path = dubins.shortest_path(q0, q1, r)
        qs, _ = path.sample_many(dt)
        qs = np.array(qs)
        qs = np.vstack((qs, q1))

        # Figure out velocity stuff
        omega = np.zeros(qs.shape[0]-1)
        dist = 0.0
        for ii in range(qs.shape[0]-1):
            x_k = qs[ii,0]
            x_k1 = qs[ii+1,0]
            y_k = qs[ii,1]
            y_k1 = qs[ii+1,1]
            t_k = qs[ii,2]
            t_k1 = qs[ii+1,2]

            del_x = x_k1 - x_k
            del_y = y_k1 - y_k
            del_t = t_k1 - t_k

            dist = dist + np.sqrt(del_x**2 + del_y**2)

            omega[ii] = del_t/dt
        v = np.ones(omega.shape)
        u = np.vstack((v, omega))

        return np.transpose(qs), u, dist

    def cons_fun(self, r, q0, q1, dt, dist_max):
        neg_dist = self.opt_rad(r, q0, q1, dt)
        return neg_dist + dist_max

    def opt_rad(self, r, q0, q1, dt):
        path = dubins.shortest_path(q0, q1, r)
        qs, _ = path.sample_many(dt)
        qs = np.array(qs)
        qs = np.vstack((qs, q1))

        # Figure out velocity stuff
        dist = 0.0
        for ii in range(qs.shape[0]-1):
            x_k = qs[ii,0]
            x_k1 = qs[ii+1,0]
            y_k = qs[ii,1]
            y_k1 = qs[ii+1,1]

            del_x = x_k1 - x_k
            del_y = y_k1 - y_k

            dist = dist + np.sqrt(del_x**2 + del_y**2)
        return -dist