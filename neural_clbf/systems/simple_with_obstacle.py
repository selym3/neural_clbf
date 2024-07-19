from typing import Tuple, Optional, List

import torch
import numpy as np
import torch.nn.functional

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList
class SimpleWithObstacle(ControlAffineSystem):

    N_DIMS = 2
    N_CONTROLS = 2

    X = 0
    Y = 1

    UX = 0
    UY = 1

    def __init__(self):
        super().__init__(
            nominal_params = {},
            dt = 0.01,
            controller_dt= None,
            use_linearized_controller = True,
            scenarios= None,
        )

    def validate_params(self, params) -> bool:
        return True

    @property
    def n_dims(self) -> int:
        return SimpleWithObstacle.N_DIMS

    @property
    def angle_dims(self):
        return []

    @property
    def n_controls(self) -> int:
        return SimpleWithObstacle.N_CONTROLS

    @property
    def state_limits(self) :
        upper_limit = torch.ones(self.n_dims)
        upper_limit[SimpleWithObstacle.X] = 10.0
        upper_limit[SimpleWithObstacle.Y] = 10.0

        lower_limit = -1.0 * upper_limit
        
        return (upper_limit, lower_limit)

    @property
    def control_limits(self) :
        upper_limit = torch.ones(self.n_dims)
        upper_limit[SimpleWithObstacle.UX] = 1.0
        upper_limit[SimpleWithObstacle.UY] = 1.0

        lower_limit = -1.0 * upper_limit
        
        return (upper_limit, lower_limit)

    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        return x.norm(dim=-1) <= 9.0

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        obstacle = torch.tensor([[ 4.0, 4.0 ]])
        # TODO: in next version, add multiple obstacles
        return (x - obstacle).norm(dim=-1) <= 1.0

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        goal_tolerance = 0.1
        return x.norm(dim=-1) <= goal_tolerance

    def _f(self, x: torch.Tensor, params):
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        # f is a zero vector as nothing should happen when no control input is given
        f[:, SimpleWithObstacle.X, 0] = 0
        f[:, SimpleWithObstacle.Y, 0] = 0

        return f 
    
    def _g(self, x: torch.Tensor, params):
        batch_size = x.shape[0]
        g = torch.eye(self.n_dims, self.n_controls).unsqueeze(0).repeat(batch_size, 1, 1).type_as(x)

        return g #identity_matrix_batch
