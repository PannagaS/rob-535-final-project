import numpy as np
import casadi as ca
from IPython import embed
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon


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
    # Extract parameters: params = [xi, yi, psi_i, vi, xg, yg, vdes, delta_last]'
    pos_des = params[4:6]
    v_des = params[6]
    # Compute desired heading
    psi_des = ca.arctan2(pos_des[1], pos_des[0])
    # Compute deviation from straight-line path
    d = x_model[:2] - (pos_des.T @ x_model[:2]) / (ca.norm_2(pos_des) ** 2) * pos_des
    # Define weights
    W = {"psi": 1.0, "d": 1.0, "v": 1.0, "a": 1.0, "delta": 1.0}
    # Compute stage cost
    J_stage = 0
    J_stage += W["psi"] * (psi_des - x_model[2]) ** 2
    J_stage += W["d"] * d.T @ d
    J_stage += W["v"] * (v_des - x_model[3]) ** 2
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
    # Extract parameters: params = [xi, yi, psi_i, vi, xg, yg, vdes, delta_last]'
    pos_des = params[4:6]
    # Define weights
    W = {"x": 1.0, "y": 1.0}
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
    delta_last = params[7]
    # Prepare a list for constraints
    cons_state = []
    for k in range(N):
        # Collision avoidance - no barrier function for now
        for A, B, C, D, E, F in ellipse_coeffs:
            # -(Ax^2 + By^2 + Cxy + Dx + Ey + F) <= 0
            hk = -A * (x[0, k]) ** 2
            hk -= B * (x[1, k]) ** 2
            hk -= C * (x[0, k] * x[1, k])
            hk -= D * (x[0, k])
            hk -= E * (x[1, k])
            hk -= F
            cons_state.append(hk)
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
    v_des = ca.MX.sym("v_des")
    delta_last = ca.MX.sym("delta_last")
    params = ca.vertcat(x_init, pos_goal, v_des, delta_last)

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
    state_ub = np.array([1e8, 1e8, np.pi / 2, 1e8])
    state_lb = np.array([-1e8, -1e8, -np.pi / 2, 0])
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


def simulate(ellipse_coefs, parameters):
    """
    parameters: [x, y, yaw, v, x goal, y goal, v des, delta_last]
    """

    parameters = np.array(parameters)

    # We define the default evaluation rate and other constants here
    dt = 0.1
    N_sim = int(np.ceil(17 / dt))
    nx = 4
    nu = 2

    # define some parameters
    x_init = ca.MX.sym("x_init", (nx, 1))
    pos_goal = ca.MX.sym("pos_goal", (2, 1))  # leader car's velocity
    v_des = ca.MX.sym("v_des")  # desired speed of ego car
    delta_last = ca.MX.sym("delta_last")  # steering angle at last step
    par = ca.vertcat(x_init, pos_goal, v_des, delta_last)  # concatenate them

    # Continuous dynamics model
    x_model = ca.MX.sym("xm", (nx, 1))
    u_model = ca.MX.sym("um", (nu, 1))

    L_f = 1.0
    L_r = 1.0

    beta = ca.atan(L_r / (L_r + L_f) * ca.atan(u_model[1]))

    xdot = ca.vertcat(
        x_model[3] * ca.cos(x_model[2] + beta),
        x_model[3] * ca.sin(x_model[2] + beta),
        x_model[3] / L_r * ca.sin(beta),
        u_model[0],
    )

    # Discretized dynamics model
    Fun_dynmaics_dt = ca.Function(
        "f_dt", [x_model, u_model, par], [xdot * dt + x_model]
    )

    # student controller are constructed here:
    prob, N_mpc, n_x, n_g, n_p, lb_var, ub_var, lb_cons, ub_cons = nmpc_controller(
        ellipse_coefs
    )
    # students are expected to provide
    # NLP problem, the problem size (n_x, n_g, n_p), horizon and bounds

    opts = {"ipopt.print_level": 0, "print_time": 0}  # , 'ipopt.sb': 'yes'}
    solver = ca.nlpsol("solver", "ipopt", prob, opts)

    # Extract initial state and previous steering angle
    x0 = parameters[:4]
    d_last = parameters[7]

    # logger of states
    xt = np.zeros((N_sim + 1, nx))
    ut = np.zeros((N_sim, nu))

    # Initial guess for warm start
    x0_nlp = np.random.randn(n_x, 1) * 0.01  # np.zeros((n_x, 1))
    lamx0_nlp = np.random.randn(n_x, 1) * 0.01  # np.zeros((n_x, 1))
    lamg0_nlp = np.random.randn(n_g, 1) * 0.01  # np.zeros((n_g, 1))

    xt[0, :] = x0

    # main loop of simulation
    for k in range(N_sim):
        xk = xt[k, :]

        # the leader car's velocity and desired velocity will not change in the planning horizon
        par_nlp = np.concatenate((xk, parameters[4:7], np.array([d_last])))

        # Solve
        # embed()
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

        x0_nlp = sol["x"].full()
        lamx0_nlp = sol["lam_x"].full()
        lamg0_nlp = sol["lam_g"].full()

        ut[k, :] = np.squeeze(sol["x"].full()[0:nu])
        d_last = ut[k, 1]

        ut[k, 0] = np.clip(ut[k, 0], -10, 4)
        ut[k, 1] = np.clip(ut[k, 1], -0.6, 0.6)

        xkp1 = Fun_dynmaics_dt(xt[k, :], ut[k, :], par_nlp)

        xt[k + 1, :] = np.squeeze(xkp1.full())

    return xt, ut


def plot_results(xt, ut):
    ### Maximum lateral acceleration ###
    h = 0.1
    gmu = 0.5 * 0.6 * 9.81
    dx = (xt[1:, :] - xt[0:-1, :]) / h
    ay = dx[:, 2] * xt[0:-1, 3]  # dx[:, 3]
    ay = np.abs(ay) - gmu

    ### Plot trajectory ###
    x_w = xt[:, 0]
    y_w = xt[:, 1]
    yaw = xt[:, 2]
    x0 = np.expand_dims(x_w, 1)
    y0 = np.expand_dims(y_w, 1)
    yaw = np.expand_dims(yaw, 1)
    dx, dy = np.cos(yaw), np.sin(yaw)
    arrows = np.concatenate((x0, y0, dx, dy), axis=1)[0:50, :]
    fig, ax = plt.subplots(figsize=(20, 3))
    ax.plot(x_w[0], y_w[0], "b.", markersize=20)
    for arrow in arrows:
        ax.arrow(
            arrow[0],
            arrow[1],
            arrow[2],
            arrow[3],
            head_width=0.5,
            head_length=1,
            fc="blue",
            ec="blue",
        )
    ax.plot(x_w, y_w)
    ax.axis("scaled")
    plt.title("Trajectory")
    plt.xlabel("$x(m)$")
    plt.ylabel("$y(m)$")
    plt.xlim((-50, 50))
    plt.ylim((-3, 4))
    # Define ellipse parameters
    center_x = 0
    center_y = 0
    width = 60
    height = 4
    color = "red"
    fill = False
    # Create an Ellipse object
    ellipse = Ellipse((center_x, center_y), width, height, color=color, fill=fill)
    # Add the ellipse to the axis
    ax.add_patch(ellipse)
    # Define the vertices of the triangle as a list of (x, y) coordinates
    vertices = [(2, 0), (-1, 1), (-1, -1)]
    # Create a Polygon patch using the vertices and add it to the axis
    triangle = Polygon(vertices, closed=True, fill=True, color="r")
    ax.add_patch(triangle)
    # y constrain
    plt.plot(x_w, 3 * np.ones(x_w.shape), "--", color="black")
    plt.plot(x_w, -1 * np.ones(x_w.shape), "--", color="black")
    plt.show()

    ### Plot x-t figure ###
    t = np.linspace(0, 17.1, 171)
    plt.figure(figsize=(20, 3))
    plt.plot(t, xt[:, 0])
    plt.grid()
    plt.xlabel("$Time(s)$")
    plt.ylabel("$x(m)$")
    plt.show()

    ### Plot y-t figure ###
    plt.figure(figsize=(20, 3))
    plt.plot(t, xt[:, 1])
    plt.grid()
    plt.xlabel("$Time(s)$")
    plt.ylabel("$y(m)$")
    plt.show()

    ### Plot yaw-t figure ###
    plt.figure(figsize=(20, 3))
    plt.plot(t, xt[:, 2])
    plt.grid()
    plt.xlabel("$Time(s)$")
    plt.ylabel("$yaw(rad)$")
    plt.show()

    ### Plot v-t figure ###
    plt.figure(figsize=(20, 3))
    plt.plot(t, xt[:, 3])
    plt.grid()
    plt.xlabel("$Time(s)$")
    plt.ylabel("$velocity(m/s)$")
    plt.show()

    ### Plot a-t figure ###
    plt.figure(figsize=(20, 3))
    plt.plot(t[1:], ay + (0.5 * 0.6 * 9.81))
    plt.grid()
    plt.xlabel("$Time(s)$")
    plt.ylabel("$Lateral\ acc$")
    plt.plot(t, np.ones(t.shape) * 0.5 * 0.6 * 9.81, "--", color="black")
    plt.plot(t, -np.zeros(t.shape), "--", color="black")
    plt.show()
