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
    RolloutStateSpaceObsExperiment
)
from neural_clbf.systems import XLayObsPoint


torch.multiprocessing.set_sharing_strategy("file_system")
def f(x):
    return 0.5*torch.cos(1.5*torch.tensor(x))
start_x = torch.tensor(
    [
        [-4.0, -4.5, 0.0, f(-4.0)],
        [ 3.0,  3.0, 0.0, f(3.0)],
        [-2.0, -3.0, 0.0, f(-2.0)],
        [ 3.5, -6.0, 0.0, f(3.5)],
        [2.0, 0.0, 0.0, f(2.0)],
        [-3.5, 0.0, 0.0, f(-3.5)],
        [1.0, -2.0, 0.0, f(1.0)]
    ]
)

controller_period = 0.01
simulation_dt = 0.01


def main(args):
    # Define the dynamics model
    nominal_params = {}
    dynamics_model = XLayObsPoint(nominal_params, dt=simulation_dt, controller_dt=controller_period)

    print(0.7)
    # Initialize the DataModule
    initial_domain = [
        (-10, 10),  # x
        (-10, 5),  # y
        (-1, 1), # wx
        (-1, 1) # wy
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_domain,
        trajectories_per_episode=10,  # disable collecting data from trajectories
        trajectory_length=3,
        fixed_samples=10000,
        max_points=100000,
        val_split=0.1,
        batch_size=128,
        quotas={"safe": 0.4, "unsafe": 0.2, "goal": 0.2},
    )

    # Define the scenarios
    scenarios = [{}]

    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(-12.0, 12.0), (-12.0, 4)],
        n_grid=25,
        x_axis_index=XLayObsPoint.X,
        y_axis_index=XLayObsPoint.Y,
        x_axis_label="$x$",
        y_axis_label="$y$",
        plot_unsafe_region=True,
    )
    rollout_state_space_experiment = RolloutStateSpaceObsExperiment(
        "Rollout State Space",
        start_x,
        plot_x_index=XLayObsPoint.X,
        plot_x_label="$x$",
        plot_y_index=XLayObsPoint.Y,
        plot_y_label="$y$",
        scenarios=[nominal_params],
        n_sims_per_start=2,
        t_sim=30.0,
        state_length=2
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_state_space_experiment])

    # Initialize the controller
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite,
        clbf_hidden_layers=3,
        clbf_hidden_size=128,
        clf_lambda=0.05,
        safe_level=1.0,
        controller_period=controller_period,
        clf_relaxation_penalty=1e1,
        primal_learning_rate=1e-3,
        penalty_scheduling_rate=0,
        num_init_epochs=0,
        epochs_per_episode=40,  # disable new data-gathering
        barrier=True,  # disable fitting level sets to a safe/unsafe boundary
        disable_gurobi= True
    )

    # Initialize the logger and trainer
    current_git_hash = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/xpoint_obs_layer_system/", name=f"commit_{current_git_hash}"
    )
    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, reload_dataloaders_every_epoch=True, max_epochs=51
    )

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
