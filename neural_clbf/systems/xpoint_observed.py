from typing import Tuple, Optional, List

import torch
import torch.nn.functional

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList

class XLayObsPoint(ControlAffineSystem):
    """
    Represents a point mass with wind observation
    The system has state
        p = [x, y, wx, wy]
    representing the x and y position of the point,
    and wind observation wx and wy,
    and it has control inputs
        u = [ux]
    representing the horizontal control.
    """

    # Number of states and controls
    N_DIMS = 4
    N_CONTROLS = 1

    # State indices
    X = 0
    Y = 1
    WX = 2
    WY = 3

    # Control indices
    UX = 0

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.001,
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

        self.P = torch.eye(self.n_dims)
        self.K = torch.zeros(self.n_controls, self.n_dims)

    def validate_params(self, params) -> bool:
        """Check if a given set of parameters is valid
        args:
            params: a dictionary giving the parameter values for the system.
        returns:
            True if parameters are valid, False otherwise
        """
        return True

    @property
    def n_dims(self) -> int:
        return XLayObsPoint.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return XLayObsPoint.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[XLayObsPoint.X] = 7
        upper_limit[XLayObsPoint.Y] = 7
        upper_limit[XLayObsPoint.WX] = 1
        upper_limit[XLayObsPoint.WY] = 1

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        upper_limit = torch.ones(self.n_controls)
        upper_limit[XLayObsPoint.UX] = 3
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def goal_point(self):
        return torch.tensor([[0.0, 5.0, 0.0, 0.0]])

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task
        args:
            x: a tensor of points in the state space
        """
        safe_bound = x[:, :2].norm(dim=-1) < 6  # Check only position (x, y)
        return safe_bound

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task
        args:
            x: a tensor of points in the state space
        """
        unsafe_bound = x[:, :2].norm(dim=-1) > 6.5  # Check only position (x, y)
        return unsafe_bound

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set
        args:
            x: a tensor of points in the state space
        """
        goal_mask = (x[:, :2] - self.goal_point[:, :2].type_as(x)).norm(dim=-1) <= 0.3
        return goal_mask.logical_and(self.safe_mask(x))

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
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1)).type_as(x)

        # Wind influence on wx and wy
        frequency_factor = 1.5
        wind_strength = 0.5

        # Position dynamics affected by wind
        f[:, XLayObsPoint.X, 0] = x[:, XLayObsPoint.WX]
        f[:, XLayObsPoint.Y, 0] = x[:, XLayObsPoint.WY]

        # Wind dynamics
        f[:, XLayObsPoint.WX, 0] = 0
        f[:, XLayObsPoint.WY, 0] = wind_strength * torch.sin(frequency_factor * x[:, XLayObsPoint.X])

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
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls)).type_as(x)

        # Only control over the x-axis
        g[:, XLayObsPoint.X, XLayObsPoint.UX] = 1.0

        return g

    def u_nominal(self, x, params=None):
        # Calculate the horizontal difference between the current state and the goal
        horizontal_diff = self.goal_point[0, XLayObsPoint.X].type_as(x) - x[:, XLayObsPoint.X]

        # Apply a proportional control gain for stability
        Kp = 0.5  # Proportional gain, you can tune this value for better stability
        u_nom = Kp * horizontal_diff

        # Clip the control values by the control limits
        upper_limit, lower_limit = self.control_limits
        upper_limit = upper_limit.to(x.device)
        lower_limit = lower_limit.to(x.device)
        u_nom = torch.clamp(u_nom, min=lower_limit, max=upper_limit)

        return u_nom.reshape(-1, self.n_controls)
