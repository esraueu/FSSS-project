
import numpy as np


class model_in:

    def __init__(self):

        linearity = 'linear'
        A = None
        B = None
        C = None
        D = None

        f = None

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
