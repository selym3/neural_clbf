import torch
import matplotlib
from neural_clbf.controllers import NeuralCLBFController
import os
import torch.distributed as dist

# from  
matplotlib.use('TkAgg')


def plot_simple():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    # log_file = "/home/myles/Programming/neural_clbf/logs/simple_system/first/version_2/checkpoints/epoch=151-step=14361.ckpt"
    log_file = "/home/myles/Programming/neural_clbf/logs/simple_system/first/version_4/checkpoints/epoch=28-step=2058.ckpt"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)
    neural_controller.disable_gurobi = True
    # Update parameters

    neural_controller.experiment_suite.experiments[0].plot_unsafe_region = True

    neural_controller.experiment_suite.experiments[1].start_x = torch.tensor(
        [
            [1.5, 1.5],
            [0.9, 1.5],
            [0.3, 1.5],
            [0.0, 1.5],
            [-0.3, 1.5],
            [-0.9, 1.5],
            [-1.5, 1.5],
            [1.5, -1.5],
            [0.9, -1.5],
            [0.3, -1.5],
            [0.0, -1.5],
            [-0.3, -1.5],
            [-0.9, -1.5],
            [-1.5, -1.5],
        ]
    )

    # Run the experiments and save the results
    neural_controller.experiment_suite.run_all_and_plot(
        neural_controller, display_plots=True
    )

def setup():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12355')
    
    dist.init_process_group(backend='gloo', init_method=f'tcp://{master_addr}:{master_port}', rank=rank, world_size=world_size)
    # if torch.gpu_avaiable ... :
    # device = 0 # rank % torch.cuda.device_count()
    # torch.cuda.set_device(device)



if __name__ == "__main__":
    setup()
    plot_simple()
