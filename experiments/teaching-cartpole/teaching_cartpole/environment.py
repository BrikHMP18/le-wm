"""Adapted single-pole physics and drawing from dmeger/TeachingCartpole.

Physics copyright (c) 2017 Juan Camilo Gamboa Higuera, Anqi Xu,
Victor Barbaros, Alex Chatron-Michaud, David Meger.
Teaching simulator by David Meger; see SOURCES.md for provenance.
"""

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
from scipy.integrate import ode
from scipy.linalg import solve_continuous_are

PARAMETERS = {"g": 9.82, "m": 0.5, "M": 0.5, "l": 0.5, "b": 1.0}
STATE_ORDER = ["x", "x_dot", "theta_dot", "theta"]


class CartPole:
    """Preserve the assignment physics; own state and solver on each reset."""

    def __init__(self, state):
        self.reset(state)

    def reset(self, state):
        self.state = np.array(state, dtype=np.float64, copy=True)
        if self.state.shape != (4,) or not np.isfinite(self.state).all():
            raise ValueError("State must have four finite elements")
        self.time = 0.0
        self.force = 0.0
        self.solver = ode(self.dynamics).set_integrator("dopri5", atol=1e-12, rtol=1e-12)
        self.solver.set_initial_value(self.state.copy(), 0.0)

    def dynamics(self, time, state):
        del time
        _, velocity, angular_velocity, theta = state
        g, m, mass, length, friction = (PARAMETERS[k] for k in ("g", "m", "M", "l", "b"))
        sine, cosine = np.sin(theta), np.cos(theta)
        a0 = m * length * angular_velocity**2 * sine
        a1 = g * sine
        a2 = self.force - friction * velocity
        denominator = 4 * (mass + m) - 3 * m * cosine**2
        return np.array(
            [
                velocity,
                (2 * a0 + 3 * m * a1 * cosine + 4 * a2) / denominator,
                -3 * (a0 * cosine + 2 * ((mass + m) * a1 + a2 * cosine)) / (length * denominator),
                angular_velocity,
            ]
        )

    def step(self, force, dt=0.005):
        if not np.isfinite(force) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Force must be finite and dt positive")
        self.force = float(force)
        self.solver.integrate(self.time + dt)
        if not self.solver.successful() or not np.isfinite(self.solver.y).all():
            raise RuntimeError("CartPole integration failed")
        self.state = self.solver.y.copy()
        self.time = float(self.solver.t)
        return self.state.copy()


def render(state, image_size=224):
    """RGB HWC uint8; fixed camera, native aspect ratio, white letterboxing."""
    surface = pygame.Surface((800, 400))
    surface.fill((255, 255, 255))
    for x in range(50, 800, 50):
        pygame.draw.line(surface, (150, 150, 150), (x, 0), (x, 400))
    for y in range(50, 400, 50):
        pygame.draw.line(surface, (150, 150, 150), (0, y), (800, y))
    centre = (int(400 + state[0] * 100), 200)
    end = (int(centre[0] + 100 * np.sin(state[3])), int(200 + 100 * np.cos(state[3])))
    pygame.draw.rect(surface, (0, 0, 0), (centre[0] - 30, 185, 60, 30))
    pygame.draw.line(surface, (0, 0, 0), centre, end, 2)
    pygame.draw.circle(surface, (150, 0, 0), centre, 5)
    pygame.draw.circle(surface, (150, 0, 0), end, 7)
    canvas = pygame.Surface((image_size, image_size))
    canvas.fill((255, 255, 255))
    height = image_size // 2
    canvas.blit(
        pygame.transform.smoothscale(surface, (image_size, height)), (0, (image_size - height) // 2)
    )
    return pygame.surfarray.array3d(canvas).transpose(1, 0, 2).copy()


def sample_initial_state(rng, mode):
    if mode == "mixed":
        mode = str(rng.choice(["upright", "downward", "uniform"]))
    angle = {
        "upright": lambda: np.pi + rng.uniform(-0.15, 0.15),
        "downward": lambda: rng.uniform(-0.3, 0.3),
        "uniform": lambda: rng.uniform(-np.pi, np.pi),
    }[mode]()
    return np.array(
        [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-1, 1), angle]
    ), mode


class Policy:
    """Held random actions or clipped local LQR with exploration noise."""

    def __init__(self, rng, name, force_limit, action_hold, noise_std):
        self.rng, self.name, self.force_limit = rng, name, force_limit
        self.action_hold, self.noise_std = action_hold, noise_std
        self.last_force = 0.0
        if name == "lqr":
            a = np.array([[0, 1, 0, 0], [0, -1.6, 0, 5.892], [0, -4.8, 0, 47.136], [0, 0, 1, 0]])
            b = np.array([[0], [1.6], [4.8], [0]])
            p = solve_continuous_are(a, b, np.diag([1, 1, 1, 10]), np.ones((1, 1)))
            self.gain = (b.T @ p).ravel()

    def action(self, state, step):
        if self.name == "random":
            if step % self.action_hold == 0:
                self.last_force = self.rng.uniform(-self.force_limit, self.force_limit)
            force = self.last_force
        elif self.name == "zero":
            force = 0.0
        else:
            error = state.copy()
            error[3] = (error[3] - np.pi + np.pi) % (2 * np.pi) - np.pi
            force = -self.gain @ error + self.rng.normal(0, self.noise_std)
        return float(np.clip(force, -self.force_limit, self.force_limit))
