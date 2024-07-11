import matplotlib
from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.experiments import RolloutStateSpaceExperiment
from neural_clbf.systems import Point
import torch
matplotlib.use('TkAgg')

start_x = torch.tensor(
    [
        # [4.0, -4.0],
        [-4.0, 6.0],
        # [-4.5, -4.5],
        # [-4.0, -4.5],
        # [-4.5, -4.0],
        # [ 4.5,  4.5],
        # [ 4.0,  4.0]
    ]
)
def plot_point():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_file = "/home/myles/Programming/neural_clbf/neural_clbf/training/logs/point_system/commit_a4b6a97/version_4/checkpoints/epoch=44-step=6344.ckpt"
    # log_file = "neural_clbf/training/logs/point_system/commit_18395a3/version_18/checkpoints/epoch=10-step=1550.ckpt"
    # log_file = "/home/myles/Programming/neural_clbf/logs/point_system/commit_f4f996b/version_0/checkpoints/epoch=0-step=140.ckpt"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    # Tweak controller params
    neural_controller.clf_relaxation_penalty = 1
    neural_controller.controller_period = 0.01
    neural_controller.clbf_lambda = 100.0
    # neural_controller.safe_level = 100.0

    rollout_experiment = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        Point.X,
        "x",
        Point.Y,
        "y",
        scenarios=[{}],
        n_sims_per_start=1,
        t_sim=15.0,
    )

    rollout_experiment.run_and_plot(neural_controller, True)

    neural_controller.experiment_suite.run_all_and_plot(
        neural_controller, display_plots=True
    )


if __name__ == "__main__":
    plot_point()