"""Define a dymamical system for Point3"""
from typing import Tuple, Optional, List

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList


class Balloon(ControlAffineSystem):
    """
    Represents a point mass
    The system has state
        p = [x, y, z]
    representing the x, y, and z position of the point,
    and it has control inputs
        u = [u]
    representing the vertical control.
    The system is parameterized by
        R: radius of the wheels
        L: radius of rotation, or the distance between the two wheels
    """

    # Number of states and controls
    N_DIMS = 2
    N_CONTROLS = 2

    # State indices
    X = 0
    Y = 1
    Z = 2

    # Control indices
    UZ = 0

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.0001,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
    ):
        super().__init__(
            nominal_params,
            dt=dt,
            controller_dt=controller_dt,
            scenarios=scenarios,
            use_linearized_controller=False,
        )

    def validate_params(self, params) -> bool:
        """Check if a given set of parameters is valid
        args:
            params: a dictionary giving the parameter values for the system.
                    Requires keys ["R", "L"]
        returns:
            True if parameters are valid, False otherwise
        """
        return True

    @property
    def n_dims(self) -> int:
        return Balloon.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return Balloon.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[Balloon.X] = 5
        upper_limit[Balloon.Y] = 5

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_controls)
        upper_limit[Balloon.UX] = 1
        upper_limit[Balloon.UY] = 1
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task
        args:
            x: a tensor of points in the state space
        """
        safe_mask = x.norm(dim=-1) <= 0.8

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task
        args:
            x: a tensor of points in the state space
        """
        x = x[:][0:2]
        unsafe_mask = x.norm(dim=-1) >= 1.5

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set
        args:
            x: a tensor of points in the state space
        """
        x = x[:][0:2]
        goal_mask = x.norm(dim=-1) <= 0.3

        return goal_mask

    def _f(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.
        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        # f is a zero vector as nothing should happen when no control input is given
        f[:, Balloon.X, 0] = 0
        f[:, Balloon.Y, 0] = 0

        return f

    def _g(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-dependent part of the control-affine dynamics.
        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
        g = g.type_as(x)

        # Extract state variables
        theta = x[:, Balloon.THETA]

        # Effect on x
        g[:, Balloon.X, Balloon.V] = torch.cos(theta)

        # Effect on y
        g[:, Balloon.Y, Balloon.V] = torch.sin(theta)

        # Effect on theta
        g[:, Balloon.THETA, Balloon.THETA_DOT] = 1.0

        return g

    def u_nominal(
        self, x: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        """
        Compute the nominal control for the nominal parameters.

        args:
            x: bs x self.n_dims tensor of state
            params: the model parameters used
        returns:
            u_nominal: bs x self.n_controls tensor of controls
        """
        # The Point linearization is not well-behaved, so we create our own
        # P and K matrices (mainly as placeholders)
        self.P = torch.eye(self.n_dims)
        self.K = torch.zeros(self.n_controls, self.n_dims)

        # This controller should navigate us towards the origin. We can do this by
        # setting a velocity proportional to the inner product of the vector
        # from the Point to the origin and the vector pointing out in front of
        # the Point. If the bot is pointing away from the origin, this inner product
        # will be negative, so we'll drive backwards towards the goal. If the bot
        # is pointing towards the origin, it will drive forwards.
        u = torch.zeros(x.shape[0], self.n_controls).type_as(x)

        v_scaling = 1.0
        bot_to_origin = -x[:, : Balloon.Y + 1].reshape(-1, 1, 2)
        theta = x[:, Balloon.THETA]
        bot_facing = torch.stack((torch.cos(theta), torch.sin(theta))).T.unsqueeze(-1)
        u[:, Balloon.V] = v_scaling * torch.bmm(bot_to_origin, bot_facing).squeeze()

        # In addition to setting the velocity towards the origin, we also need to steer
        # towards the origin. We can do this via P control on the angle between the
        # Point and the vector to the origin.
        #
        # However, this angle becomes ill-defined as the bot approaches the origin, so
        # so we switch this term off if the bot is too close (and instead just control
        # theta to zero)
        phi_control_on = bot_to_origin.norm(dim=-1) >= 0.02
        phi_control_on = phi_control_on.reshape(-1)
        omega_scaling = 5.0
        angle_from_origin_to_bot = torch.atan2(x[:, Balloon.Y], x[:, Balloon.X])
        phi = theta - angle_from_origin_to_bot
        # First, wrap the angle error into [-pi, pi]
        phi = torch.atan2(torch.sin(phi), torch.cos(phi))
        # Now decrement any errors > pi/2 by pi and increment any errors < -pi / 2 by pi
        # Then P controlling the error to zero will drive the bot to point towards the
        # origin
        phi[phi > np.pi / 2.0] -= np.pi
        phi[phi < -np.pi / 2.0] += np.pi

        # Only apply this P control when the bot is far enough from the origin;
        # default to P control on theta
        u[:, Balloon.THETA_DOT] = -omega_scaling * theta
        u[phi_control_on, Balloon.THETA_DOT] = -omega_scaling * phi[phi_control_on]

        # Clamp given the control limits
        u_upper, u_lower = self.control_limits
        u = torch.clamp(u, u_lower.type_as(u), u_upper.type_as(u))

        return u
