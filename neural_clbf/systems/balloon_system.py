from typing import Tuple, Optional, List
from scipy.optimize import minimize
import torch
import numpy as np
import torch.nn.functional
import cvxpy as cp
import casadi as ca

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList
class SimpleBalloon2d(ControlAffineSystem):

    N_DIMS = 2
    N_CONTROLS = 1

    X = 0
    Z = 1

    UZ = 0

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

        self.P = torch.eye(2)
        # self.prediction_horizon = 20
        self.previous_x = None

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
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[SimpleBalloon2d.X] = 7
        upper_limit[SimpleBalloon2d.Z] = 7

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        upper_limit = torch.ones(self.n_controls)
        upper_limit[SimpleBalloon2d.UZ] = 5.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, vectors: torch.Tensor) -> torch.Tensor:
        safe_bound = vectors.norm(dim=-1) < 6

        return safe_bound

    def unsafe_mask(self, vectors: torch.Tensor) -> torch.Tensor:
        unsafe_bound = vectors.norm(dim=-1) > 6.5
        return unsafe_bound
        # return torch.zeros_like(vectors[:, 0], dtype=torch.bool) #result

    @property
    def u_eq(self):
        return torch.tensor([[ 0.0 ]])

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        goal_tolerance = 0.1
        return x.norm(dim=-1) <= goal_tolerance

    def _f(self, x: torch.Tensor, params):
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1), device=x.device)

        # Calculate the distance from the goal
        distances = torch.norm(x - self.goal_point.type_as(x), dim=1)

        # Set f to zero if within 0.3 units of the goal
        close_to_goal = distances < 0.3

        h = x[:, SimpleBalloon2d.Z]
        f[~close_to_goal, SimpleBalloon2d.X, 0] = -torch.sin(h[~close_to_goal])
        f[~close_to_goal, SimpleBalloon2d.Z, 0] = 0.0

        return f
    
    def _g(self, x: torch.Tensor, params):
        batch_size = x.shape[0]
        values = torch.zeros((batch_size, self.n_dims, self.n_controls))
        values[:, SimpleBalloon2d.X, 0] = 0
        values[:, SimpleBalloon2d.Z, 0] = 1.0

        return values.type_as(x)


    def u_nominal(self, x, params=None):
        if self.previous_x is None or self.previous_x.shape != x.shape:
            self.previous_x = x.clone()

        # Ensure goal_point is on the same device as x
        goal_point = self.goal_point.to(x.device)

        # Check if each point is approaching the goal in x-direction
        approaching_goal = (x[:, SimpleBalloon2d.X] - self.previous_x[:, SimpleBalloon2d.X]) < 0

        # Control input
        u_nom = torch.zeros((x.shape[0], self.n_controls)).type_as(x)

        # Define the control limits
        upper_limit, lower_limit = self.control_limits
        upper_limit = upper_limit.to(x.device)
        lower_limit = lower_limit.to(x.device)

        # Generate linspace of potential z values within the control limits
        z_values = torch.linspace(lower_limit[SimpleBalloon2d.UZ], upper_limit[SimpleBalloon2d.UZ], 10).to(x.device)

        # Process each state individually
        for i in range(x.shape[0]):
            if approaching_goal[i]:
                # Take the difference between current z and goal z
                z_diff = goal_point[0, SimpleBalloon2d.Z] - x[i, SimpleBalloon2d.Z]
                Kp = 0.5  # Proportional gain for z control
                u_nom[i, SimpleBalloon2d.UZ] = Kp * z_diff
            else:
                # Find the z that can switch the direction of movement along x-axis
                current_z = x[i, SimpleBalloon2d.Z]
                best_z = current_z
                min_distance = float('inf')

                for z in z_values:
                    h = z
                    sin_h_approx = h - (h**3) / 6.0

                    if (goal_point[0, SimpleBalloon2d.X] > x[i, SimpleBalloon2d.X] and sin_h_approx > 0) or (goal_point[0, SimpleBalloon2d.X] < x[i, SimpleBalloon2d.X] and sin_h_approx < 0):
                        distance = torch.abs(z - current_z)
                        if distance < min_distance:
                            min_distance = distance
                            best_z = z

                z_diff = best_z - current_z
                Kp = 0.5  # Proportional gain for z control
                u_nom[i, SimpleBalloon2d.UZ] = Kp * z_diff

        # Clip the control values by the control limits
        u_nom = torch.clamp(u_nom, min=lower_limit, max=upper_limit)

        # Update the previous state
        self.previous_x = x.clone()

        return u_nom.reshape(-1, self.n_controls)
    # def u_nominal(self, x, params=None):
    #     pass
    #     horizontal_diff = self.goal_point[0, SimpleBalloon2d.Z].type_as(x) - x[:, SimpleBalloon2d.Z]

    #     # Apply a proportional control gain for stability
    #     Kp = 0.5  # Proportional gain, you can tune this value for better stability
    #     u_nom = Kp * horizontal_diff

    #     # Clip the control values by the control limits
    #     upper_limit, lower_limit = self.control_limits
    #     upper_limit = upper_limit.to(x.device)
    #     lower_limit = lower_limit.to(x.device)
    #     u_nom = torch.clamp(u_nom, min=lower_limit, max=upper_limit)

    #     return u_nom.reshape(-1, self.n_controls)
        # return self.mpc_control(x, params)
    
    # def mpc_control(self, x: torch.Tensor, params) -> torch.Tensor:
    #     """
    #     Apply Model Predictive Control (MPC) to determine the optimal control input using CasADi.
    #     """
    #     device = x.device  # Get the device of the input tensor
    #     batch_size = x.shape[0]
    #     u_optimal = torch.zeros((batch_size, self.n_controls), device=device)  # Ensure u_optimal is on the same device

    #     for i in range(batch_size):
    #         x0 = x[i].cpu().numpy()

    #         # Define CasADi variables
    #         U = ca.MX.sym("U", self.n_controls, self.prediction_horizon)
    #         X = ca.MX.sym("X", self.n_dims, self.prediction_horizon + 1)

    #         # Define the cost function and constraints
    #         cost = 0
    #         constraints = []

    #         for t in range(self.prediction_horizon):
    #             h = X[self.Z, t]
    #             sin_h_approx = h - (h**3) / 6.0
    #             f_x = ca.vertcat(-0.1 * sin_h_approx, 0.0)
    #             g_x = ca.vertcat(0, 1)

    #             if t == 0:
    #                 constraints.append(X[:, t] == x0)

    #             cost += ca.sumsqr(X[:, t+1] - X[:, t]) + ca.sumsqr(U[:, t])
    #             constraints.append(X[:, t+1] == X[:, t] + self.dt * (f_x + g_x * U[:, t]))

    #         # Define upper and lower control limits
    #         upper_limit, lower_limit = self.control_limits
    #         constraints.append(U <= upper_limit.cpu().numpy().reshape(self.n_controls, 1))
    #         constraints.append(U >= lower_limit.cpu().numpy().reshape(self.n_controls, 1))

    #         # Concatenate all constraints into a single MX expression
    #         g = ca.vertcat(*[ca.reshape(c, -1, 1) for c in constraints])

    #         # Create the NLP problem
    #         nlp = {"x": ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1)),
    #             "f": cost,
    #             "g": g}

    #         # Create the solver
    #         opts = {"ipopt.print_level": 0, "print_time": 0}
    #         solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    #         # Solve the problem
    #         sol = solver(lbg=0, ubg=0)

    #         # Extract the optimal control input
    #         u_opt = np.array(sol["x"][:self.n_controls]).flatten()
    #         u_optimal[i, :] = torch.tensor(u_opt, device=device)  # Ensure u_opt is on the same device
    #         # print("Find u_optimal: ", u_optimal)

    #     return u_optimal

    # def u_nominal(self, x, params=None):
    #     # Define a function that returns the wind effect at a given z level
    #     def wind_effect_at_z(z):
    #         return -0.1 * torch.sin(z)
        
    #     # Determine the current position and the goal position
    #     current_z = x[:, SimpleBalloon2d.Z]
    #     goal_x = self.goal_point[0, SimpleBalloon2d.X].type_as(x)
    #     current_x = x[:, SimpleBalloon2d.X]
        
    #     # Calculate the direction towards the goal in the x-axis
    #     direction_to_goal = torch.sign(goal_x - current_x).unsqueeze(-1)  # Expand dimension for broadcasting
        
    #     # Assume z levels to check are in the range of the state limits
    #     state_upper_limit, state_lower_limit = self.state_limits
    #     z_levels = torch.linspace(state_lower_limit[SimpleBalloon2d.Z].item(), state_upper_limit[SimpleBalloon2d.Z].item(), steps=50).type_as(x)
    #     wind_effects = wind_effect_at_z(z_levels).unsqueeze(0).repeat(x.shape[0], 1)  # Expand and repeat for batch
        
    #     # Find the z level with the wind effect in the direction towards the goal
    #     valid_wind_effects = wind_effects * direction_to_goal
    #     valid_wind_effects[valid_wind_effects <= 0] = float('inf')  # Ignore wind effects in the opposite direction
        
    #     # Check if the wind effect makes progress towards the goal
    #     progress_wind_effects = wind_effects.clone()
    #     progress_wind_effects[(direction_to_goal > 0) & (wind_effects < 0)] = float('inf')
    #     progress_wind_effects[(direction_to_goal < 0) & (wind_effects > 0)] = float('inf')
        
    #     # Combine valid wind effects with progress check
    #     combined_effects = valid_wind_effects + progress_wind_effects
        
    #     # Find the z level with the combined effect closest to pushing the balloon towards the goal
    #     best_z_indices = torch.argmin(combined_effects, dim=1)
    #     best_z_levels = z_levels[best_z_indices]
        
    #     # Heuristic control gain
    #     Kp = 0.5  # Proportional gain

    #     # Calculate the control input based on the current position and the best z level
    #     vertical_diff = best_z_levels - current_z

    #     # Apply the heuristic rule: if the vertical difference is significant, apply a stronger control
    #     if torch.abs(vertical_diff).mean() > 0.5:
    #         u_nom = Kp * vertical_diff
    #     else:
    #         # If the vertical difference is small, apply a smaller control to fine-tune the position
    #         u_nom = 0.1 * vertical_diff

    #     # Clip the control values by the control limits
    #     upper_limit, lower_limit = self.control_limits
    #     upper_limit = upper_limit.to(x.device)
    #     lower_limit = lower_limit.to(x.device)
    #     u_nom = torch.clamp(u_nom, min=lower_limit, max=upper_limit)
        
    #     return u_nom.reshape(-1, self.n_controls)