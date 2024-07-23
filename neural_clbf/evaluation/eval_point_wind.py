import matplotlib
from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.experiments import RolloutStateSpaceExperiment
from neural_clbf.systems import Point, LinearWind, PointInWind, PointInCirWind, XCirPoint, XPoint, XPointSim
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
        # [-4.5, -1.0],
        # [ 4.5,  4.5],
        [ 4.0,  4.0],
        # [-3.0, 0.0]
        # [5.0, 0.0]
        [0.0, 2.5],
        # [-2.5, 8.0],
        # [-6, -5],
        # [5, -5]
        [-1.0, 0.0],
        [-2.5, 0.5],
        [0, -2.5],
        [2.5, 0.0]
        # [0.0, 5.0]
        # [2.0, 2.0]
        # [3.0, -1.0]
    ]
)
def plot_point():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    # log_file = "/home/jnl77/neural_clbf/logs/point_linear_system/commit_2362a64/version_0/checkpoints/epoch=69-step=5564.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_circular_wind_system/commit_ce95b94/version_0/checkpoints/epoch=69-step=5564.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_wind_system/commit_ce95b94/version_0/checkpoints/epoch=69-step=5573.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/xpoint_circular_wind_system/commit_ce95b94/version_1/checkpoints/epoch=89-step=8699.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/xpoint_system/commit_ce95b94/version_3/checkpoints/epoch=69-step=3534.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_wind_system/commit_28c3c11/version_0/checkpoints/epoch=69-step=5573.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_circular_wind_system/commit_28c3c11/version_3/checkpoints/epoch=69-step=5564.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/xpoint_system/commit_28c3c11/version_1/checkpoints/epoch=69-step=3534.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/xpoint_circular_wind_system/commit_28c3c11/version_1/checkpoints/epoch=89-step=8699.ckpt"
    # log_file = "/home/jnl77/neural_clbf/logs/point_linear_system/commit_28c3c11/version_0/checkpoints/epoch=69-step=5564.ckpt"
    log_file = "/home/jnl77/neural_clbf/logs/xpoint_sim_system/commit_cd10a61/version_2/checkpoints/epoch=69-step=2834.ckpt"
    
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    # Tweak controller params
    neural_controller.clf_relaxation_penalty = 1e1
    neural_controller.controller_period = 0.01
    # neural_controller.clbf_lambda = 100.0
    # neural_controller.safe_level = 100.0

    rollout_experiment = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        # LinearWind.X,
        # PointInCirWind.X,
        # PointInWind.X,
        # XCirPoint.X,
        # XPoint.X,
        XPointSim.X,
        "x",
        # LinearWind.Y,
        # PointInCirWind.Y,
        # PointInWind.Y,
        # XCirPoint.Y,
        # XPoint.Y,
        XPointSim.Y,
        "y",
        scenarios=[{}],
        n_sims_per_start=1,
        t_sim=30.0, #15
    )

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