import matplotlib
from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.experiments import RolloutStateSpaceExperiment
from neural_clbf.systems import Point
import torch
import torch.distributed as dist
import os
# matplotlib.use('TkAgg')
matplotlib.use('Agg') # on g2

start_x = torch.tensor(
    [
        # [4.0, -4.0],
        # [-4.0, 6.0],
        # [-4.5, -4.5],
        # [-4.0, -4.5],
        # [-4.5, -4.0],
        [5.0, 0.0],
        [-2.5, 8.0]
        # [5.0, 5.0],
        # [ 5,  4.5],
        # [6.0, 3.0]
        # [ 4.0,  4.0]
        # [2.0, 2.0]
    ]
)
def plot_point():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    # log_file = "neural_clbf/training/logs/point_system/commit_18395a3/version_18/checkpoints/epoch=10-step=1550.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system/commit_ca80db0/version_1/checkpoints/epoch=99-step=20858.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_sw/commit_ca80db0/version_1/checkpoints/epoch=99-step=20858.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_ba/commit_ca80db0/version_0/checkpoints/epoch=99-step=10459.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_0un/commit_ca80db0/version_1/checkpoints/epoch=99-step=20858.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_ba3/commit_ca80db0/version_0/checkpoints/epoch=99-step=10459.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_ba3/commit_ca80db0/version_1/checkpoints/epoch=99-step=10459.ckpt"
    # log_file = "logs/point_system_sws/commit_2362a64/version_0/checkpoints/epoch=59-step=4189.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_sws/commit_2362a64/version_1/checkpoints/epoch=59-step=4189.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_sws/commit_2362a64/version_2/checkpoints/epoch=59-step=4189.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_sws/commit_2362a64/version_3/checkpoints/epoch=59-step=8338.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_sws/commit_2362a64/version_5/checkpoints/epoch=59-step=8338.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_v1/commit_2362a64/version_0/checkpoints/epoch=59-step=4189.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_v1/commit_2362a64/version_1/checkpoints/epoch=59-step=4189.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_v1/commit_2362a64/version_2/checkpoints/epoch=59-step=4189.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_v1/commit_2362a64/version_3/checkpoints/epoch=59-step=3174.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_v1/commit_2362a64/version_4/checkpoints/epoch=59-step=4189.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_v1/commit_2362a64/version_5/checkpoints/epoch=59-step=4189.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_s1/commit_2362a64/version_0/checkpoints/epoch=59-step=4189.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_v2/commit_2362a64/version_0/checkpoints/epoch=59-step=4189.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_v2/commit_2362a64/version_1/checkpoints/epoch=59-step=4189.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system_v2/commit_2362a64/version_3/checkpoints/epoch=59-step=4189.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_system/commit_2362a64/version_0/checkpoints/epoch=43-step=2493.ckpt"
    log_file = "/home/jnl77/neural_clbf/logs/point_system/commit_2362a64/version_1/checkpoints/epoch=59-step=4189.ckpt"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    # Tweak controller params
    neural_controller.clf_relaxation_penalty = 1
    neural_controller.controller_period = 0.01
    # neural_controller.clbf_lambda = 100.0
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
        t_sim=40.0,
    )
    # neural_controller.experiment_suite.experiments[0].plot_unsafe_region = True
    rollout_experiment.run_and_plot(neural_controller, True)

    neural_controller.experiment_suite.run_all_and_plot(
        neural_controller, display_plots=True
    )

def setup():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12355')
    
    # gloo: cpu; nccl: gpu
    dist.init_process_group(backend='gloo', init_method=f'tcp://{master_addr}:{master_port}', rank=rank, world_size=world_size)
    # if torch.gpu_avaiable ... :
    # device = torch.cuda.device_count()
    # torch.cuda.set_device(device)

if __name__ == "__main__":
    setup()
    plot_point()