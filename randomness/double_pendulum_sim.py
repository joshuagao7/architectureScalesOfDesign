import numpy as np


def double_pendulum_derivatives(state, L1, L2, m1, m2, g):
    theta1, omega1, theta2, omega2 = state
    delta = theta1 - theta2

    denom = 2 * m1 + m2 - m2 * np.cos(2 * delta)
    denom1 = L1 * denom
    denom2 = L2 * denom

    dtheta1 = omega1
    dtheta2 = omega2

    domega1 = (
        -g * (2 * m1 + m2) * np.sin(theta1)
        - m2 * g * np.sin(theta1 - 2 * theta2)
        - 2 * np.sin(delta) * m2
        * (omega2 * omega2 * L2 + omega1 * omega1 * L1 * np.cos(delta))
    ) / denom1

    domega2 = (
        2 * np.sin(delta)
        * (
            omega1 * omega1 * L1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(theta1)
            + omega2 * omega2 * L2 * m2 * np.cos(delta)
        )
    ) / denom2

    return np.array([dtheta1, domega1, dtheta2, domega2], dtype=float)


def rk4_step(state, dt, derivs, *args):
    k1 = derivs(state, *args)
    k2 = derivs(state + 0.5 * dt * k1, *args)
    k3 = derivs(state + 0.5 * dt * k2, *args)
    k4 = derivs(state + dt * k3, *args)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def inverse_kinematics_end_effector(x, y, L1, L2, elbow="up"):
    r2 = x * x + y * y
    r = np.sqrt(r2)
    if r > L1 + L2 or r < abs(L1 - L2):
        raise ValueError(
            f"Target (x={x:.3f}, y={y:.3f}) is unreachable for lengths L1={L1}, L2={L2}."
        )

    cos_q2 = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    sin_q2 = np.sqrt(1.0 - cos_q2 * cos_q2)
    if elbow == "down":
        sin_q2 = -sin_q2

    q2 = np.arctan2(sin_q2, cos_q2)
    q1 = np.arctan2(y, x) - np.arctan2(L2 * sin_q2, L1 + L2 * cos_q2)

    theta1 = q1 + np.pi / 2.0
    theta2 = q1 + q2 + np.pi / 2.0
    return theta1, theta2


def simulate_double_pendulum(
    theta1_0, theta2_0, L1, L2, m1, m2, g, dt, t_max
):
    steps = int(t_max / dt) + 1
    states = np.empty((steps, 4), dtype=float)
    state = np.array([theta1_0, 0.0, theta2_0, 0.0], dtype=float)
    states[0] = state

    for i in range(1, steps):
        state = rk4_step(
            state, dt, double_pendulum_derivatives, L1, L2, m1, m2, g
        )
        states[i] = state

    theta1 = states[:, 0]
    theta2 = states[:, 2]
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2


def generate_trajectories(
    x_positions, target_y, L1, L2, m1, m2, g, dt, t_max, elbow="up"
):
    trajectories = []
    for x in x_positions:
        theta1_0, theta2_0 = inverse_kinematics_end_effector(
            x, target_y, L1, L2, elbow=elbow
        )
        trajectories.append(
            simulate_double_pendulum(theta1_0, theta2_0, L1, L2, m1, m2, g, dt, t_max)
        )
    return trajectories
