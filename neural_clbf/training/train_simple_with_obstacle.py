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
from neural_clbf.systems import SimpleWithObstacle


torch.multiprocessing.set_sharing_strategy("file_system")


start_x = torch.tensor(
    [
        (-6, 1.4),
        (7.0, 9.0),
        (-2.1, -1.5),
        (0.9, -7.25),
        (4.3, 6.76),
        (-1.0, 0.0)
    ]
)
controller_period = 0.01
simulation_dt = 0.01


def main(args):
    # Define the dynamics model
    dynamics_model  = SimpleWithObstacle()

    # Initialize the DataModule
    domains = [
        (-10.0, 10.0),  # x
        (-10.0, 10.0),  # y
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        domains,
        trajectories_per_episode=5,  # disable collecting data from trajectories
        trajectory_length=1,
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
        domain=[(-12.0, 12.0), (-12.0, 12.0)],
        n_grid=25,
        x_axis_index=SimpleWithObstacle.X,
        y_axis_index=SimpleWithObstacle.Y,
        x_axis_label="x",
        y_axis_label="y",
        plot_unsafe_region=True,
    )
    rollout_experiment = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        SimpleWithObstacle.X,
        "x",
        SimpleWithObstacle.Y,
        "y",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=5.0,
    )
    experiment_suite = ExperimentSuite([ V_contour_experiment, rollout_experiment ])

    # Initialize the controller
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite,
        clbf_hidden_layers=4,
        clbf_hidden_size=128,
        clf_lambda=0.05,
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
    tb_logger = pl_loggers.TensorBoardLogger("logs/simple_system_with_wind/", name='config1')
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy=args.strategy, logger=tb_logger,  check_val_every_n_epoch=1,reload_dataloaders_every_n_epochs=1, max_epochs=500)

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)


if __name__ == "__main__":
    
    import torch.distributed as dist
    import PIL
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

    torch.set_float32_matmul_precision('medium')

    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--strategy", default=None)
    torch.set_float32_matmul_precision('medium')
    args = parser.parse_args()

    # def setup(rank, world_size):
    #     os.environ['MASTER_ADDR'] = 'localhost'
    #     os.environ['MASTER_PORT'] = '12355'

    #     # initialize the process group
    
    # rank = dist.get_rank() % 2
    # world_size = dist.get_world_size()
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)


    main(args)
