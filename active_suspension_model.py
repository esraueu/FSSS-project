import numpy as np
import casadi as ca
from dataclasses import dataclass
from utils import integrate_RK4, plot_cstr
from ocp_solver_ipopt import OCPsolver
from model import Model
import matplotlib.pyplot as plt

@dataclass
class SuspensionParameters:
    Ms: float = 2.45  # Sprung Mass (kg)
    Mus: float = 1.0  # Unsprung Mass (kg)
    Ks: float = 900.0  # Suspension Stiffness (N/m)
    Kus: float = 1250.0  # Tire Stiffness (N/m)
    Bs: float = 7.5  # Suspension Damping Coefficient (Nsec/m)
    Bus: float = 5.0  # Tire Damping Coefficient (Nsec/m)
    xs: np.ndarray = np.zeros(4)  # State reference
    us: np.ndarray = np.zeros(2)  # Input reference


@dataclass
class MpcSuspensionParameters:
    Q: np.ndarray
    R: np.ndarray
    Tf: float = 1.0  # Prediction horizon length (seconds)
    N: int = 20  # Number of control intervals
    dt: float = 0.05  # Sampling time
    num_rk4_steps: int = 10

    def __init__(self, xs, us):
        self.Q = np.eye(len(xs))  # State cost weight
        self.R = np.eye(len(us))  # Input cost weight

def setup_suspension_model(dt: float, num_steps: int, params: SuspensionParameters):

    # States: [zs - zus, z_s, zus - zr, z_us]
    x = ca.SX.sym("x", 4)

    # Inputs: [Fz_rc]
    u = ca.SX.sym("u", 1)

    # Dynamics matrices
    A = np.array([
        [0, 1, 0, -1],
        [-params.Ks / params.Ms, -params.Bs / params.Ms, 0, params.Bs / params.Ms],
        [0, 0, 0, 1],
        [params.Ks / params.Mus, params.Bs / params.Mus, -params.Kus / params.Mus, -(params.Bs + params.Bus) / params.Mus]
    ])

    B = np.array([
        [0],
        [1 / params.Ms],
        [0],
        [-1 / params.Mus]
    ])

    # State Space dynamics
    f_expl = ca.mtimes(A, x) + ca.mtimes(B, u)

    # Discretize dynamics using RK4
    f_discrete = integrate_RK4(x, u, f_expl, dt, num_steps)

    model = Model(x, u, f_discrete, params.xs, params.us, name='suspension')

    return model


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

def main_mpc_open_loop():

    # Define system parameters
    params = SuspensionParameters()

    # Time discretization and horizon
    dt = 0.05
    num_steps = 10
    N_horizon = 20

    # Setup model
    model = setup_suspension_model(dt, num_steps, params)

    # Define initial state (perturbed from xs)
    x0 = np.array([0.01, 0.0, 0.01, 0.0])

    # Define cost matrices
    Q = np.diag([100.0, 1.0, 100.0, 1.0])  # State cost (for suspension travel, body velocity, tire deflection, tire velocity)
    R = np.diag([1.0])  # Input cost (control force)

    # MPC problem setup
    # Stage cost as a symbolic expression
    delta_x = model.x_expr - params.xs
    delta_u = model.u_expr - params.us
    stage_cost = 0.5 * ca.mtimes([delta_x.T, Q, delta_x]) + 0.5 * ca.mtimes([delta_u.T, R, delta_u])
    
    # Terminal cost as a symbolic expression
    terminal_cost = 0.5 * ca.mtimes([delta_x.T, Q, delta_x])

    umin = np.array([-500])  # Min control force
    umax = np.array([500])   # Maximum control force

    # Initialize OCP solver
    ocp_solver = OCPsolver(model, stage_cost, terminal_cost, N_horizon, umax, umin)

    # Solve OCP for open-loop control
    u_init = np.zeros((1, N_horizon))  # Initial guess for control inputs
    u_traj, cost = ocp_solver.solve(x0, u_traj_init=u_init)

    # Simulate resulting state trajectory
    x_traj = model.simulate_traj(x0, u_traj)

    # Ensure state trajectory has N_horizon + 1 steps (states at each time step)
    assert x_traj.shape == (N_horizon+1, 4), "State trajectory dimension mismatch."
    # Plot the control and state results
    plot_results(dt, x_traj, u_traj, N_horizon)

if __name__ == "__main__":
    main_mpc_open_loop()
