import numpy as np
import casadi as ca
from IPython import embed
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from helpers import plot_ellipses


def get_bicyce_model_dynamics_function(x_model, u_model, dt):
    """Generates and returns a Casadi Function which computes the discrete-time dynamics of the bicycle model

    Args:
        x_model (ca.MX): 4x1 Casadi symbolic of the state
        u_model (ca.MX): 2x1 Casadi symbolic of the control input
        dt (float): time step of the dynamics

    Returns:
        ca.Function: Function which computes discrete time dynamics of the bicycle model
    """
    # Define front and rear length of car
    L_f = 1.0
    L_r = 1.0
    # Define sideslip
    beta = ca.atan(L_r * ca.atan(u_model[1]) / (L_r + L_f))
    # Define derivative of the state
    xdot = ca.vertcat(
        x_model[3] * ca.cos(x_model[2] + beta),
        x_model[3] * ca.sin(x_model[2] + beta),
        x_model[3] * ca.sin(beta) / L_r,
        u_model[0],
    )
    # Calculate the next state using forward-euler
    xkp1 = x_model + xdot * dt
    # Create Casadi function of the dynamics
    bicyce_model_dynamics = ca.Function("f_dt", [x_model, u_model], [xkp1])
    # Return
    return bicyce_model_dynamics


def get_stage_cost_function(x_model, u_model, params):
    """Generates and returns a Casadi Function which computes the stage cost of a state-control pair

    Args:
        x_model (ca.MX): 4x1 Casadi symbolic of the state
        u_model (ca.MX): 2x1 Casadi symbolic of the state
        params (ca.MX): the parameter vector from nmpc_controller

    Returns:
        ca.Function: Function which computes the stage cost of a state-control pair
    """
    # Extract parameters: params = [xi, yi, psi_i, vi, xg, yg, delta_last]'
    pos_des = params[4:6]
    # Compute desired heading
    psi_des = ca.arctan2(pos_des[1], pos_des[0])
    # Compute deviation from straight-line path
    d = x_model[:2] - (pos_des.T @ x_model[:2]) / (ca.norm_2(pos_des) ** 2) * pos_des
    # Define weights
    W = {"psi": 10.0, "d": 10.0, "v": 1.0, "a": 1.0, "delta": 1.0}
    # Compute stage cost
    J_stage = 0
    J_stage += W["psi"] * (psi_des - x_model[2]) ** 2
    J_stage += W["d"] * d.T @ d
    J_stage += W["a"] * (u_model[0]) ** 2
    J_stage += W["delta"] * (u_model[1]) ** 2
    # Define stage cost function
    J_stage_func = ca.Function("J_stage", [x_model, u_model, params], [J_stage])
    # Return
    return J_stage_func


def get_terminal_cost_function(x_model, params):
    """Generates and returns a Casadi Function which computes the terminal cost of a state

    Args:
        x_model (ca.MX): 4x1 Casadi symbolic of the state
        params (ca.MX): the parameter vector from nmpc_controller

    Returns:
        ca.Function: Function which computes the terminal cost of a state
    """
    # Extract parameters: params = [xi, yi, psi_i, vi, xg, yg, delta_last]'
    pos_des = params[4:6]
    # Define weights
    W = {"x": 1000.0, "y": 1000.0}
    # Compute terminal cost
    J_term = 0
    J_term += W["x"] * (pos_des[0] - x_model[0]) ** 2
    J_term += W["y"] * (pos_des[1] - x_model[1]) ** 2
    # Define terminal cost function
    J_term_func = ca.Function("J_term", [x_model, params], [J_term])
    # Return
    return J_term_func


def get_dynamic_model_constraints(dynamics_func, x, u, N):
    """Generates list of dynamic model constraints and lists of bounds for those constraints

    Args:
        dynamics_func (ca.Function): Dynamic model function
        x (ca.MX): 4x(N+1) Casadi symbolic of all states
        u (ca.MX): 2xN Casadi symbolic of all controls
        N (int): Planning horizon in steps

    Returns:
        tuple: (dynamic model constraints, lower bounds, upper bounds)
    """
    # Prepare a list for constraints
    cons_dynamics = []
    # Loop through planning horizon
    for k in range(N):
        # Compute the next state
        xkp1 = dynamics_func(x[:, k], u[:, k])
        # Enforce the next state is correct
        for i in range(x.shape[0]):
            cons_dynamics.append(x[i, k + 1] - xkp1[i])
    # Define bounds to be zero
    ub_dynamics = np.zeros((x.shape[0] * N, 1))
    lb_dynamics = np.zeros((x.shape[0] * N, 1))
    # Return
    return cons_dynamics, lb_dynamics, ub_dynamics


def get_state_constraints(ellipse_coeffs, x, u, params, N, dt):
    """Generates list of state constraints and lists of bounds for those constraints

    Args:
        ellipse_coeffs (ndarray): Mx6 array of coefficients for ellipses defining obstacles
        x (ca.MX): 4x(N+1) Casadi symbolic of all states
        u (ca.MX): 2xN Casadi symbolic of all controls
        params (ca.MX): parameter vector from nmpc_controller
        N (int): Planning horizon in steps
        dt (float): time step

    Returns:
        tuple: (state constraints, lower bounds, upper bounds)
    """
    # Unpack parameters
    delta_last = params[6]
    # Prepare a list for constraints
    cons_state = []
    for k in range(N):
        # Collision avoidance - no barrier function for now
        for A, B, C, D, E, F in ellipse_coeffs:
            # -(Ax^2 + By^2 + Cxy + Dx + Ey + F) <= 0
            hk = A * (x[0, k]) ** 2
            hk += B * (x[1, k]) ** 2
            hk += C * (x[0, k] * x[1, k])
            hk += D * (x[0, k])
            hk += E * (x[1, k])
            hk += F

            hkp1 = A * (x[0, k + 1]) ** 2
            hkp1 += B * (x[1, k + 1]) ** 2
            hkp1 += C * (x[0, k + 1] * x[1, k])
            hkp1 += D * (x[0, k + 1])
            hkp1 += E * (x[1, k + 1])
            hkp1 += F

            gamma = 1
            cons_state.append((1 - gamma) * hk - hkp1)
        # Maximum lateral acceleration
        vy = (x[2, k + 1] - x[2, k]) / dt
        ay = x[3, k] * vy
        gmu = 0.5 * 0.6 * 9.81
        cons_state.append(-ay - gmu)
        cons_state.append(ay - gmu)
        # Steering rate
        d_delta = (
            (u[1, k] - u[1, k - 1]) / dt if (k > 0) else (u[1, k] - delta_last) / dt
        )
        cons_state.append(-d_delta - 0.6)
        cons_state.append(d_delta - 0.6)
    # Define boundaries: (-inf, 0]
    ub_state_cons = np.zeros((len(cons_state), 1))
    lb_state_cons = np.zeros((len(cons_state), 1)) - 1e9
    # Return
    return cons_state, lb_state_cons, ub_state_cons


def nmpc_controller(ellipse_coeffs):
    """Create a nonlinear model predictive controller assuming bicycle model dynamics, and the
    only obstacles are the ellipses defined by the coefficients in the rows of ellipse_coeffs

    Args:
        goal_state (ndarray): 2x1 representing target x and y coordinates
        ellipse_coeff (ndarray): Mx6 array, where M is the number of obstacles, and each row is a set of coefficients
    """
    # Declare simulation constants
    T = 3  # Planning horizon in seconds
    N = 30  # Planning horizon in steps
    dt = T / N  # Time step: 0.1

    # system dimensions
    nx = 4
    nu = 2

    # additional parameters: initial state, target position, desired velocity, previous steering angle
    x_init = ca.MX.sym("x_init", (nx, 1))
    pos_goal = ca.MX.sym("pos_goal", (2, 1))
    delta_last = ca.MX.sym("delta_last")
    params = ca.vertcat(x_init, pos_goal, delta_last)

    # Define Casadi symbolics for states and controls
    x_model = ca.MX.sym("xm", (nx, 1))  # [x; y; psi; v]
    u_model = ca.MX.sym("um", (nu, 1))  # [a; delta]

    # Get discrete time dynamics function
    dynamics_func = get_bicyce_model_dynamics_function(x_model, u_model, dt)

    # Get stage cost function
    J_stage_func = get_stage_cost_function(x_model, u_model, params)

    # Get terminal cost function
    J_term_func = get_terminal_cost_function(x_model, params)

    # Define state and control bounds
    state_ub = np.array([1e8, 1e8, 10, 1e8])
    state_lb = np.array([-1e8, -1e8, -10, 0])
    ctrl_ub = np.array([4, 0.6])
    ctrl_lb = np.array([-10, -0.6])

    # Define upper bound and lower bound arrays for state and control
    ub_x = np.matlib.repmat(state_ub, N + 1, 1)
    lb_x = np.matlib.repmat(state_lb, N + 1, 1)
    ub_u = np.matlib.repmat(ctrl_ub, N, 1)
    lb_u = np.matlib.repmat(ctrl_lb, N, 1)

    # Define upper and lower bound arrays for massed decision variables
    ub_var = np.concatenate(
        (ub_u.reshape((nu * N, 1)), ub_x.reshape((nx * (N + 1), 1)))
    )
    lb_var = np.concatenate(
        (lb_u.reshape((nu * N, 1)), lb_x.reshape((nx * (N + 1), 1)))
    )

    # Declare model variables
    x = ca.MX.sym("x", (nx, N + 1))
    u = ca.MX.sym("u", (nu, N))

    # Build constraints: dynamics, state constraints, initial state constraint
    cons_dynamics, lb_dynamics_cons, ub_dynamics_cons = get_dynamic_model_constraints(
        dynamics_func, x, u, N
    )
    cons_state, lb_state_cons, ub_state_cons = get_state_constraints(
        ellipse_coeffs, x, u, params, N, dt
    )
    cons_init = [x[:, 0] - x_init]
    ub_init_cons = np.zeros((nx, 1))
    lb_init_cons = np.zeros((nx, 1))

    # Collect constraints and bounds
    cons_NLP = cons_dynamics + cons_state + cons_init
    cons_NLP = ca.vertcat(*cons_NLP)
    lb_cons = np.concatenate((lb_dynamics_cons, lb_state_cons, lb_init_cons))
    ub_cons = np.concatenate((ub_dynamics_cons, ub_state_cons, ub_init_cons))

    # Define decision variable: w = [ubar; xbar]
    vars_NLP = ca.vertcat(u.reshape((nu * N, 1)), x.reshape((nx * (N + 1), 1)))

    # Compute cost of trajectory
    J = J_term_func(x[:, -1], params)
    for k in range(N):
        J += J_stage_func(x[:, k], u[:, k], params)

    # Create and return all parameters for NLP solver
    prob = {"x": vars_NLP, "p": params, "f": J, "g": cons_NLP}

    return (
        prob,
        N,
        vars_NLP.shape[0],
        cons_NLP.shape[0],
        params.shape[0],
        lb_var,
        ub_var,
        lb_cons,
        ub_cons,
    )


def simulate(ellipse_coefs, params):
    """
    params: [x, y, yaw, v, x goal, y goal, delta_last]
    """
    # Define time step, simulation length, dimensions of state and control
    dt = 0.1
    N_sim = int(np.ceil(17 / dt))
    nx = 4
    nu = 2
    # Continuous dynamics model
    x_model = ca.MX.sym("xm", (nx, 1))
    u_model = ca.MX.sym("um", (nu, 1))
    # Get bicycle model dynamics
    dynamics_func = get_bicyce_model_dynamics_function(x_model, u_model, dt)
    # Construct obstacle-aware NMPC policy
    prob, N_mpc, n_x, n_g, n_p, lb_var, ub_var, lb_cons, ub_cons = nmpc_controller(
        ellipse_coefs
    )
    # Set options and construct NLP solver
    opts = {"ipopt.print_level": 1, "print_time": 0}
    solver = ca.nlpsol("solver", "ipopt", prob, opts)
    # Extract initial state and previous steering angle
    x0 = params[:4]
    delta_last = params[6]
    # logger of states
    xlog = np.zeros((N_sim + 1, nx))
    ulog = np.zeros((N_sim, nu))
    tlog = np.linspace(0, N_sim * dt, N_sim + 1)
    # Initial guess for warm start
    x0_nlp = np.random.randn(n_x, 1) * 0.01  # np.zeros((n_x, 1))
    lamx0_nlp = np.random.randn(n_x, 1) * 0.01  # np.zeros((n_x, 1))
    lamg0_nlp = np.random.randn(n_g, 1) * 0.01  # np.zeros((n_g, 1))
    # Set initial position
    xlog[0, :] = x0
    # main loop of simulation
    for k in range(N_sim):
        # Grab current state
        xk = xlog[k, :]
        # Construct up-to-date parameter vector for MPC
        par_nlp = np.concatenate((xk, params[4:6], np.array([delta_last])))
        # Solve NLP trajectory optimization
        sol = solver(
            x0=x0_nlp,
            lam_x0=lamx0_nlp,
            lam_g0=lamg0_nlp,
            lbx=lb_var,
            ubx=ub_var,
            lbg=lb_cons,
            ubg=ub_cons,
            p=par_nlp,
        )
        # Store solution x, lambda, and mu for next iteration warm-start
        x0_nlp = sol["x"].full()
        lamx0_nlp = sol["lam_x"].full()
        lamg0_nlp = sol["lam_g"].full()
        # Extract control input for current step
        ulog[k, :] = np.squeeze(sol["x"].full()[0:nu])
        # Saturate control inputs as necessary
        ulog[k, 0] = np.clip(ulog[k, 0], -10, 4)
        ulog[k, 1] = np.clip(ulog[k, 1], -0.6, 0.6)
        # Store previous steering angle for steering rate calculation
        delta_last = ulog[k, 1]
        # Compute next state
        xkp1 = dynamics_func(xlog[k, :], ulog[k, :])
        # Store next state
        xlog[k + 1, :] = np.squeeze(xkp1.full())

    return xlog, ulog, tlog
