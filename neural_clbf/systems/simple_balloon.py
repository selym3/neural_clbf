from typing import Tuple, Optional, List

import torch
import numpy as np
import torch.nn.functional

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList
class SimpleBalloon(ControlAffineSystem):

    N_DIMS = 3
    N_CONTROLS = 1

    X = 0
    Y = 1
    Z = 2

    UZ = 0

    def __init__(self):
        super().__init__(
            nominal_params = {},
            dt = 0.01,
            controller_dt= None,
            use_linearized_controller = False,
            scenarios= None,
        )

        self.P = torch.eye(3)

    def validate_params(self, params) -> bool:
        return True

    @property
    def n_dims(self) -> int:
        return SimpleBalloon.N_DIMS

    @property
    def angle_dims(self):
        return []

    @property
    def n_controls(self) -> int:
        return SimpleBalloon.N_CONTROLS

    @property
    def state_limits(self) :
        upper_limit = torch.ones(self.n_dims)
        upper_limit[SimpleBalloon.X] = 10.0
        upper_limit[SimpleBalloon.Y] = 10.0
        upper_limit[SimpleBalloon.Z] = 3.1415 * 2

        lower_limit = -1.0 * upper_limit
        lower_limit[SimpleBalloon.Z] = 0.0
        
        return (upper_limit, lower_limit)

    @property
    def control_limits(self) :
        upper_limit = torch.ones(self.n_dims)
        upper_limit[SimpleBalloon.UZ] = 10.0

        lower_limit = -1.0 * upper_limit
        
        return (upper_limit, lower_limit)

    def safe_mask(self, vectors: torch.Tensor) -> torch.Tensor:
        # Extract x, y, z components
        x = vectors[:, SimpleBalloon.X]
        y = vectors[:, SimpleBalloon.Y]
        z = vectors[:, SimpleBalloon.Z]

        # Check if (x, y) is within a circle of radius 10
        within_circle = x**2 + y**2 <= 100

        # Check if z is between 0 and 2*pi
        within_z_range = (z >= 0) & (z <= 2 * 3.1415)

        # Both conditions must be true
        result = within_circle & within_z_range

        return result.type_as(x).bool()

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x[:, 0], dtype=torch.bool)

    @property
    def u_eq(self):
        return torch.tensor([[ 0.0 ]])

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        goal_tolerance = 0.1
        return x.norm(dim=-1) <= goal_tolerance

    def _f(self, x: torch.Tensor, params):
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        # f is a zero vector as nothing should happen when no control input is given
        h =  x[:, SimpleBalloon.Z]
        f[:, SimpleBalloon.X, 0] = 0.1 * torch.cos(h) * (torch.tanh(1000.0 * h - 20.0) + 1.0)
        f[:, SimpleBalloon.Y, 0] = 0.1 * torch.sin(h) * (torch.tanh(1000.0 * h - 20.0) + 1.0)

        return f 
    
    def _g(self, x: torch.Tensor, params):
        batch_size = x.shape[0]
        values = torch.zeros((batch_size, self.n_dims, self.n_controls))
        values[:, SimpleBalloon.X, 0] = 0
        values[:, SimpleBalloon.Y, 0] = 0
        values[:, SimpleBalloon.Z, 0] = 1.0

        return values.type_as(x)

    def u_nominal(self, x, params=None):
        return torch.zeros((x.shape[0], self.n_controls)).type_as(x)
    
