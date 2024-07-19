"""Define a dymamical system for Point3 in Linear Wind"""
from typing import Tuple, Optional, List

import torch
import numpy as np
import torch.nn.functional

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList


class LinearWind(ControlAffineSystem):
    """
    Represents a point mass
    The system has state
        p = [x, y ]
    representing the x and y position of the point,
    and it has control inputs
        u = [ux, uy]
    representing the horizontal and vertical control.
    """

    # Number of states and controls
    N_DIMS = 2
    N_CONTROLS = 2

    # State indices
    X = 0
    Y = 1

    # Control indices
    UX = 0
    UY = 1

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
            use_linearized_controller=True,
        )

        # self.P = torch.eye(self.n_dims)
        # self.K = torch.zeros(self.n_controls, self.n_dims)

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
        return LinearWind.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return LinearWind.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """

        upper_limit = torch.ones(self.n_dims)
        upper_limit[LinearWind.X] = 5
        upper_limit[LinearWind.Y] = 5

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """

        upper_limit = torch.ones(self.n_controls)
        upper_limit[LinearWind.UX] = 10
        upper_limit[LinearWind.UY] = 10
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)
    
    @property
    def goal_point(self):
        return torch.tensor([[ 4.0, 4.0 ]])
    
    @property
    def u_eq(self):
        return torch.tensor([ [-1.0, 0.0 ] ])

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task
        args:
            x: a tensor of points in the state space
        """
        safe_mask = x.norm(dim=-1) > 1.0
        
        # Set a safe boundary
        safe_bound = x.norm(dim=-1) < 5.0
        safe_mask = safe_mask.logical_and(safe_bound)

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task
        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = x.norm(dim=-1) <= 1.0

        return unsafe_mask

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
        
        f[:, LinearWind.X, 0] = 1.0

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
        g = torch.eye(self.n_dims, self.n_controls).unsqueeze(0).repeat(batch_size, 1, 1).type_as(x)

        return g

    # def u_nominal(
    #     self, x: torch.Tensor, params: Optional[Scenario] = None
    # ) -> torch.Tensor:
    #     """
    #     Compute the nominal control for the nominal parameters.

    #     args:
    #         x: bs x self.n_dims tensor of state
    #         params: the model parameters used
    #     returns:
    #         u_nominal: bs x self.n_controls tensor of controls
    #     """
    #     to_target = self.goal_point.repeat(x.shape[0], 1).type_as(x) - x
    #     to_target = torch.nn.functional.normalize(to_target, p=2, dim=1).type_as(x) # by normalizing, always falls in allowed controls set
    #     return to_target # torch.zeros((x.shape[0], self.n_controls))
