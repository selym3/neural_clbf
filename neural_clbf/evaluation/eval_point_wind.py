import matplotlib
from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.experiments import RolloutStateSpaceExperiment
from neural_clbf.systems import Point
import torch
import torch.distributed as dist
import os

matplotlib.use('TkAgg')
# matplotlib.use('Agg') # on g2

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
    log_file = "/home/myles/Programming/neural_clbf/logs/point_wind_system/commit_704a00d/version_0/checkpoints/epoch=99-step=20858.ckpt"
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
        t_sim=15.0,
    )

    rollout_experiment.run_and_plot(neural_controller, True)

    neural_controller.experiment_suite.run_all_and_plot(
        neural_controller, display_plots=True
    )
    
# def init_distributed_mode():
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
#     dist.init_process_group('nccl', rank=dist.get_rank(), world_size=dist.get_world_size())  # Use 'nccl' for GPU or 'gloo' for CPU

def setup():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12355')
    
    dist.init_process_group(backend='gloo', init_method=f'tcp://{master_addr}:{master_port}', rank=rank, world_size=world_size)
    # if torch.gpu_avaiable ... :
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    
if __name__ == "__main__":
    setup()
    plot_point()