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
from neural_clbf.systems import SimpleBalloon


torch.multiprocessing.set_sharing_strategy("file_system")



start_x = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 2.0],
        [-1.0, 1.0, 2.0],
        [1.0, +1.0, 4.0],
        [-1.0, 1.0, 4.0],
        [2.0, 2.0, 2.0],
        [-2.0, 2.0, 2.0],
        [2.0, +2.0, 4.0],
        [-2.0, 2.0, 4.0]
        
    ]
)
controller_period = 0.01
simulation_dt = 0.01


def main(args):
    # Define the dynamics model
    dynamics_model  = SimpleBalloon()

    # Initialize the DataModule
    domains = [
        (-5.0, 5.0),  # x
        (-5.0, 5.0),  # y
        (0.0, 6.283) # z
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        domains,
        trajectories_per_episode=10,  # disable collecting data from trajectories
        trajectory_length=500,
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
        domain=[(-7.0, 7.0), (-7.0, 7.0)],
        n_grid=25,
        x_axis_index=SimpleBalloon.X,
        y_axis_index=SimpleBalloon.Z,
        x_axis_label="x",
        y_axis_label="z",
        plot_unsafe_region=True,
    )
    rollout_experiment = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        SimpleBalloon.X,
        "x",
        SimpleBalloon.Z,
        "z",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=20.0,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment])

    # Initialize the controller
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite,
        clbf_hidden_layers=4,
        clbf_hidden_size=128,
        clf_lambda=0.01,
        safe_level=1.0,
        controller_period=controller_period,
        clf_relaxation_penalty=1e1,
        primal_learning_rate=1e-3,
        penalty_scheduling_rate=0,
        num_init_epochs=0,
        epochs_per_episode=100,
        barrier=True, 
        disable_gurobi= True
    )

    # Initialize the logger and trainer
    current_git_hash = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
    tb_logger = pl_loggers.TensorBoardLogger("logs/simple_balloon/", name=f'commit_{current_git_hash}')
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, reload_dataloaders_every_epoch=True, max_epochs=100)

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