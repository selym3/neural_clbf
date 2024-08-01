import matplotlib
from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.experiments import RolloutStateSpaceExperiment
from neural_clbf.systems import SimpleBalloon, SimpleBalloon2d
import torch
import torch.distributed as dist
import os
import logging
# matplotlib.use('TkAgg')
matplotlib.use('Agg') # on g2

start_x = torch.tensor(
    [
        [-4.0, -4.5],
        
    ]
)

def plot_point():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.

    log_file = "/home/jnl77/neural_clbf/logs/simple_balloon2d_system/commit_14bc260/version_11/checkpoints/epoch=50-step=2185.ckpt"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    # Tweak controller params
    neural_controller.clf_relaxation_penalty = 1
    neural_controller.controller_period = 0.01
    # neural_controller.clbf_lambda = 100.0
    # neural_controller.safe_level = 100.0

    rollout_experiment = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        SimpleBalloon2d.X,
        "x",
        SimpleBalloon2d.Z,
        "z",
        scenarios=[{}],
        n_sims_per_start=1,
        t_sim=40.0,
    )
    # neural_controller.experiment_suite.experiments[0].plot_unsafe_region = True
    rollout_experiment.run_and_plot(neural_controller, True)

    # neural_controller.experiment_suite.run_all_and_plot(
    #     neural_controller, display_plots=True
    # )

def setup():
    print("In setup")
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12355')
    print("Finish getting env variables")
    # gloo: cpu; nccl: gpu
    
    logging.basicConfig(level=logging.DEBUG)
    dist.init_process_group(backend='nccl', init_method=f'tcp://{master_addr}:{master_port}', rank=rank, world_size=world_size)
    # if torch.gpu_avaiable ... :
    # device = torch.cuda.device_count()
    # torch.cuda.set_device(device)
    print("Finish init process group")

if __name__ == "__main__":
    setup()
    print("Finished setup")
    plot_point()