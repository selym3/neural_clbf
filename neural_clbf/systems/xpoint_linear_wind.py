"""Define a dymamical system for Point3 in Circular Wind"""
from typing import Tuple, Optional, List

import torch
import torch.nn.functional

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList


class XLinPoint(ControlAffineSystem):
    """
    Represents a point mass
    The system has state
        p = [x, y]
    representing the x and y position of the point,
    and it has control inputs
        u = [ux]
    representing the horizontal control.
    """

    # Number of states and controls
    N_DIMS = 2
    N_CONTROLS = 1

    # State indices
    X = 0
    Y = 1

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
        return XLinPoint.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return XLinPoint.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[XLinPoint.X] = 7
        upper_limit[XLinPoint.Y] = 7

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
        upper_limit[XLinPoint.UX] = 10 #100 #30 #15
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)
    
    @property
    def goal_point(self):
        return torch.tensor([[ 0.0, 5.0 ]])

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task
        args:
            x: a tensor of points in the state space
        """
        # safe_mask = x.norm(dim=-1) > 1.0
        
        # Set a safe boundary
        safe_bound = x.norm(dim=-1) < 6
        # safe_mask = safe_mask.logical_and(safe_bound)
        
        # x_lower_bound = -6
        # x_upper_bound = 6
        # y_lower_bound = -6
        # y_upper_bound = 6
        
        # within_x_bounds = (x[:, XLinPoint.X] >= x_lower_bound) & (x[:, XLinPoint.X] <= x_upper_bound)
        # within_y_bounds = (x[:, XLinPoint.Y] >= y_lower_bound) & (x[:, XLinPoint.Y] <= y_upper_bound)

        # safe_bound = within_x_bounds & within_y_bounds
        return safe_bound

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task
        args:
            x: a tensor of points in the state space
        """
        unsafe_bound = x.norm(dim=-1) > 6.5
        # x_lower_bound = -10.5
        # x_upper_bound = 6
        # y_lower_bound = -10.5
        # y_upper_bound = 10.5
        
        # within_x_bounds = (x[:, XLinPoint.X] <= x_lower_bound) | (x[:, XLinPoint.X] >= x_upper_bound)
        # within_y_bounds = (x[:, XLinPoint.Y] <= y_lower_bound) | (x[:, XLinPoint.Y] >= y_upper_bound)

        # unsafe_bound = within_x_bounds | within_y_bounds
        return unsafe_bound

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set
        args:
            x: a tensor of points in the state space
        """
        goal_mask = (x - self.goal_point.type_as(x)).norm(dim=-1) <= 0.3

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
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)
        
        f[:, XLinPoint.X, 0] = 0.0
        f[:, XLinPoint.Y, 0] = 0.1
        
        close_to_goal = (x - self.goal_point.type_as(x)).norm(dim=-1) <= 0.3
        f[close_to_goal, XLinPoint.Y, 0] = 0.0

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
        g = torch.zeros((batch_size, self.n_dims, self.n_controls)).type_as(x)
        
        # Only control over the x-axis
        g[:, XLinPoint.X, XLinPoint.UX] = 1.0
        
        return g
    
    # def u_nominal(self, x, params=None):
    #     hor = x[:, XLinPoint.X].type_as(x)
    #     hor_target = torch.zeros_like(hor).type_as(x)
    #     return (hor_target - hor).reshape(-1, self.n_controls)
    def u_nominal(self, x, params=None):
        # Calculate the horizontal difference between the current state and the goal
        horizontal_diff = self.goal_point[0, XLinPoint.X].type_as(x) - x[:, XLinPoint.X]

        # Apply a proportional control gain for stability
        Kp = 0.5  # Proportional gain, you can tune this value for better stability
        u_nom = Kp * horizontal_diff

        # Clip the control values by the control limits
        upper_limit, lower_limit = self.control_limits
        upper_limit = upper_limit.to(x.device)
        lower_limit = lower_limit.to(x.device)
        u_nom = torch.clamp(u_nom, min=lower_limit, max=upper_limit)

        return u_nom.reshape(-1, self.n_controls)
