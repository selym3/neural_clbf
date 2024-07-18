from argparse import ArgumentParser
from copy import copy
import subprocess

import PIL.Image
import numpy as np
import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutStateSpaceExperiment
)
from neural_clbf.systems import ControlAffineSystem

class Simple(ControlAffineSystem):

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
        return Simple.N_DIMS

    @property
    def angle_dims(self):
        return []

    @property
    def n_controls(self) -> int:
        return Simple.N_CONTROLS

    @property
    def state_limits(self) :
        upper_limit = torch.ones(self.n_dims)
        upper_limit[Simple.X] = 10.0
        upper_limit[Simple.Y] = 10.0

        lower_limit = -1.0 * upper_limit
        
        return (upper_limit, lower_limit)

    @property
    def control_limits(self) :
        upper_limit = torch.ones(self.n_dims)
        upper_limit[Simple.UX] = 1.0
        upper_limit[Simple.UY] = 1.0

        lower_limit = -1.0 * upper_limit
        
        return (upper_limit, lower_limit)

    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        return x.norm(dim=-1) <= 9.0

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x[:, 0], dtype=torch.bool)

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        goal_tolerance = 0.1
        return x.norm(dim=-1) <= goal_tolerance

    def _f(self, x: torch.Tensor, params):
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        # f is a zero vector as nothing should happen when no control input is given
        f[:, Simple.X, 0] = 0
        f[:, Simple.Y, 0] = 0

        return f 
    
    def _g(self, x: torch.Tensor, params):
        batch_size = x.shape[0]
        g = torch.eye(self.n_dims, self.n_controls).unsqueeze(0).repeat(batch_size, 1, 1).type_as(x)

        return g #identity_matrix_batch

torch.multiprocessing.set_sharing_strategy("file_system")



start_x = torch.tensor(
    [
        [0.0, 0.0],
        [1.0, -1.0],
        [3.0, 5.0],
        [8.9, 0.0]
    ]
)
controller_period = 0.01
simulation_dt = 0.01


def main(args):
    # Define the dynamics model
    dynamics_model  = Simple()

    # Initialize the DataModule
    domains = [
        (-10.0, 10.0),  # x
        (-10.0, 10.0),  # y
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        domains,
        trajectories_per_episode=1,  # disable collecting data from trajectories
        trajectory_length=1,
        fixed_samples=10000,
        max_points=100000,
        val_split=0.1,
        batch_size=64,
        quotas={"safe": 0.4, "unsafe": 0.2, "goal": 0.2},
    )

    # Define the scenarios
    scenarios = [{}]

    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(-12.0, 12.0), (-12.0, 12.0)],
        n_grid=25,
        x_axis_index=Simple.X,
        y_axis_index=Simple.Y,
        x_axis_label="x",
        y_axis_label="y",
        plot_unsafe_region=False,
    )
    rollout_experiment = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        Simple.X,
        "x",
        Simple.Y,
        "y",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=5.0,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment])

    # Initialize the controller
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite,
        clbf_hidden_layers=4,
        clbf_hidden_size=64,
        clf_lambda=0.05,
        safe_level=1.0,
        controller_period=controller_period,
        clf_relaxation_penalty=1e1,
        primal_learning_rate=1e-3,
        penalty_scheduling_rate=0,
        num_init_epochs=0,
        epochs_per_episode=100,  # disable new data-gathering
        barrier=True,  # disable fitting level sets to a safe/unsafe boundary
        disable_gurobi= True
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger("logs/simple_system/", name='first')
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, reload_dataloaders_every_epoch=True, max_epochs=500)

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)


if __name__ == "__main__":
    
    import PIL
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
