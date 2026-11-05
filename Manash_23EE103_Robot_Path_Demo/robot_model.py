"""Differential-drive model using the teacher's formulas.

Values marked ASSUMED must be confirmed with the teacher.
"""
import numpy as np

R = 0.033       # wheel radius r [m]                         (ASSUMED)
B = 0.080       # half wheel base b, so 2b = 0.160 m         (ASSUMED)
MC = 0.9        # chassis mass mc [kg]                        (ASSUMED)
MW = 0.05       # mass of one wheel mw [kg]                   (ASSUMED)
IC = 0.0045     # chassis inertia Ic [kg m^2]                 (ASSUMED)

M_TOTAL = MC + 2 * MW                 # m = mc + 2 mw
I_TOTAL = IC + 2 * MW * B ** 2        # I = Ic + 2 mw b^2


def body_velocity(phi_dot_r, phi_dot_l):
    """v = r/2 (phi_r' + phi_l'),  w = r/(2b) (phi_r' - phi_l')"""
    v = R / 2 * (phi_dot_r + phi_dot_l)
    w = R / (2 * B) * (phi_dot_r - phi_dot_l)
    return v, w


def wheel_speeds(v, w):
    """Inverse of body_velocity: wheel angular speeds for a commanded (v, w)."""
    return (v + B * w) / R, (v - B * w) / R


def theta_from_wheels(theta0, phi_r, phi_l, phi_r0, phi_l0):
    """theta(t) = theta(0) + r/(2b) [(phi_r - phi_l)(t) - (phi_r - phi_l)(0)]"""
    return theta0 + R / (2 * B) * ((phi_r - phi_l) - (phi_r0 - phi_l0))


def mass_matrix():
    """M = [[m r^2/4 + I r^2/4b^2, m r^2/4 - I r^2/4b^2], [same, swapped]]"""
    a = M_TOTAL * R ** 2 / 4
    c = I_TOTAL * R ** 2 / (4 * B ** 2)
    return np.array([[a + c, a - c], [a - c, a + c]])


def wheel_torques(phi_ddot_r, phi_ddot_l):
    """tau = M * phi_ddot"""
    return mass_matrix() @ np.array([phi_ddot_r, phi_ddot_l])
