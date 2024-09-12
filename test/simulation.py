import numpy as np
import casadi as ca
from active_suspension_model import SuspensionParameters, MpcSuspensionParameters, setup_suspension_model
from model import Model
from utils import integrate_RK4, plot_cstr
from ocp_solver_ipopt import OCPsolver
import matplotlib.pyplot as plt



class simulation:
    def __init__(self, dt, num_steps, x0, u0):

        self.model_param = SuspensionParameters()
        self.mpc_param = MpcSuspensionParameters()

        self.mpc_solver = None
        self.correct_model = None
        self.offset_model = None

        self.dt = dt
        self.num_steps = num_steps

        self.x0 = x0
        self.u0 = u0

        self.uk = None
        self.x_curr = None    # current state
        self.x_curr_n = None  # current state with noise

        self.mpc_init()


    def mpc_init(self):
        # Setup model
        A,B,x,u = setup_suspension_model(self.dt, self.num_steps, self.model_param)
        
        # State Space dynamics
        f_expl = ca.mtimes(A, x) + ca.mtimes(B, u) + self.x0
        
        # Discretize dynamics using RK4
        f_discrete = integrate_RK4(x, u, f_expl, self.dt, 10)
        self.correct_model = Model(x, u, f_discrete, self.x0, self.u0, name='suspension')

        # MPC problem setup
        # Stage cost as a symbolic expression
        delta_x = self.correct_model.x_expr - self.model_param.xs
        delta_u = self.correct_model.u_expr - self.model_param.us
        stage_cost = 0.5 * ca.mtimes([delta_x.T, self.mpc_param.Q, delta_x]) + 0.5 * ca.mtimes([delta_u.T, self.mpc_param.R, delta_u])
        
        # Terminal cost as a symbolic expression
        terminal_cost = 0.5 * ca.mtimes([delta_x.T, self.mpc_param.Q, delta_x])

        # Initialize OCP solver
        self.mpc_solver = OCPsolver(self.correct_model, stage_cost, terminal_cost, self.mpc_param.N, self.mpc_param.umax, self.mpc_param.umin)

        return
            

    def open_loop_mpc(self):
        if self.mpc_solver == None:
            self.mpc_solver()

        # Solve OCP for open-loop control
        u_init = np.zeros((1, self.mpc_param.N))  # Initial guess for control inputs
        u_traj, cost = self.mpc_solver.solve(self.x0, u_traj_init=u_init)

        # Simulate resulting state trajectory
        x_traj = self.correct_model.simulate_traj(self.x0, u_traj)
        x_traj = np.transpose(x_traj)
        # Ensure state trajectory has N_horizon + 1 steps (states at each time step)
        assert x_traj.shape == (self.mpc_param.N+1, 4), "State trajectory dimension mismatch."
        # Plot the control and state results
        plot_results(self.dt, x_traj, u_traj, self.mpc_param.N)   
        return
    

    def closed_loop_mpc(self,nsim):
        # self.uk
        self.x_curr = self.x0
        for i in range(nsim):
            u_init = np.zeros((1, self.mpc_param.N))  # Initial guess for control inputs
            u_traj, cost = self.mpc_solver.solve(self.x_curr, u_traj_init=u_init)
            u0 = u_traj[0][0]
            self.x_curr = self.correct_model.simulate(self.x_curr, u0)
            print(self.x_curr)
        return




# Function to plot control inputs and state trajectories
def plot_results(dt, x_traj, u_traj, N_horizon):
    # Time vector for the trajectory
    time_vector = np.arange(0, (N_horizon+1) * dt, dt)

    # Create subplots for state and control plots
    fig, axs = plt.subplots(5, 1, figsize=(10, 12))

    # Plot state trajectories (zs, z_s, zus, z_us)
    axs[0].plot(time_vector, x_traj[:, 0], label='Suspension deflection (zs - zus)')
    axs[0].set_ylabel('zs - zus (m)')
    axs[0].legend()

    axs[1].plot(time_vector, x_traj[:, 1], label='Body velocity (z_s)')
    axs[1].set_ylabel('z_s (m/s)')
    axs[1].legend()

    axs[2].plot(time_vector, x_traj[:, 2], label='Tire deflection (zus - zr)')
    axs[2].set_ylabel('zus - zr (m)')
    axs[2].legend()

    axs[3].plot(time_vector, x_traj[:, 3], label='Tire velocity (z_us)')
    axs[3].set_ylabel('z_us (m/s)')
    axs[3].legend()

    # Plot control input (Fz_rc) (one step fewer than states)
    axs[4].plot(time_vector[:-1], u_traj.T, label='Control Input (Fz_rc)', color='r')
    axs[4].set_ylabel('Control Input (N)')
    axs[4].set_xlabel('Time (s)')
    axs[4].legend()
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    x0 = np.array([0.01, 0.0, 0.01, 0.0])
    u0 = np.array([0.0])
    sim = simulation(0.01, 1000, x0, u0)
    sim.closed_loop_mpc(10)
    # sim.open_loop_mpc()
