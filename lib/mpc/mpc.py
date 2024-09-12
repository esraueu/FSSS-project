import numpy as np


class mpc:
    def __init__(self, x_expr, u_expr, f_expr, x_steady_state, u_steady_state, name='model'):
        # Input bounds
        umin = None
        umax = None

        # State bounds
        xmin = None
        xmax = None

        # Output bounds
        ymin = None
        ymax = None

        # Input bounds on the target problem
        umin_ss = None
        umax_ss = None

        # State bounds on the target problem
        xmin_ss = None
        xmax_ss = None

        # Output bounds on the target problem
        ymin_ss = None
        ymax_ss = None

        # Input bounds on the dynamic problem
        umin_dyn = None
        umax_dyn = None

        # State bounds on the dynamic problem
        xmin_dyn = None
        xmax_dyn = None

        # Output bounds on the dynamic problem
        ymin_dyn = None
        ymax_dyn = None

        # Disturbance bounds
        dmin = None
        dmax = None

        # DeltaInput bounds
        Dumin = None
        Dumax = None

        # State noise bounds
        wmin = None
        wmax = None

        # Ouput noise bounds
        vmin = None
        vmax = None
