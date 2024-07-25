from typing import Tuple, Optional, List

import torch
import numpy as np
import torch.nn.functional

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList
class SimpleBalloon2d(ControlAffineSystem):

    N_DIMS = 2
    N_CONTROLS = 1

    X = 0
    Z = 1

    UZ = 0

    def __init__(self):
        super().__init__(
            nominal_params = {},
            dt = 0.01,
            controller_dt= None,
            use_linearized_controller = False,
            scenarios= None,
        )

        self.P = torch.eye(2)

    def validate_params(self, params) -> bool:
        return True

    @property
    def n_dims(self) -> int:
        return SimpleBalloon2d.N_DIMS

    @property
    def angle_dims(self):
        return []

    @property
    def n_controls(self) -> int:
        return SimpleBalloon2d.N_CONTROLS

    @property
    def state_limits(self) :
        upper_limit = torch.ones(self.n_dims)
        upper_limit[SimpleBalloon2d.X] = 11
        upper_limit[SimpleBalloon2d.Z] = 11

        lower_limit = -1.0 * upper_limit
        lower_limit[SimpleBalloon2d.Z] = -1

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) :
        upper_limit = torch.ones(self.n_dims)
        upper_limit[SimpleBalloon2d.UZ] = 10.0

        lower_limit = -1.0 * upper_limit
        
        return (upper_limit, lower_limit)

    def safe_mask(self, vectors: torch.Tensor) -> torch.Tensor:
        # Extract x, y components
        x = vectors[:, SimpleBalloon2d.X]
        z = vectors[:, SimpleBalloon2d.Z]


        # (-10, 10)
        within_circle = x**2 <= 10**2

        # Check if z is between 0 and 10
        within_z_range = (z >= 0) & (z <= 10.0)

        # Both conditions must be true
        result = within_circle & within_z_range

        return result.type_as(x).bool()

    def unsafe_mask(self, vectors: torch.Tensor) -> torch.Tensor:
        x = vectors[:, SimpleBalloon2d.X]
        z = vectors[:, SimpleBalloon2d.Z]

        outside_circle = x**2 >= 10.9**2

        outside_z_range = (z <= -0.9) & (z >= 10.9)

        result = outside_circle | outside_z_range
        return result
        # return torch.zeros_like(vectors[:, 0], dtype=torch.bool) #result

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
        h =  x[:, SimpleBalloon2d.Z]
        f[:, SimpleBalloon2d.X, 0] = -0.1 * (h)
        f[:, SimpleBalloon2d.Z, 0] = 0.0

        return f 
    
    def _g(self, x: torch.Tensor, params):
        batch_size = x.shape[0]
        values = torch.zeros((batch_size, self.n_dims, self.n_controls))
        values[:, SimpleBalloon2d.X, 0] = 0
        values[:, SimpleBalloon2d.Z, 0] = 1.0

        return values.type_as(x)

    def u_nominal(self, x, params=None):
        z = x[:, SimpleBalloon2d.Z].type_as(x)
        z_target = torch.zeros_like(z).type_as(x)
        return (z_target - z).reshape(-1, self.n_controls)
    
