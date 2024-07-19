import torch
import matplotlib
from neural_clbf.controllers import NeuralCLBFController
import os
import torch.distributed as dist
from neural_clbf.systems import SimpleWithObstacle
from neural_clbf.experiments import RolloutStateSpaceExperiment

# from  
matplotlib.use('TkAgg')


start_x = torch.tensor(
    [
        (-6, 1.4),
        (7.0, 9.0),
        (-2.1, -1.5),
        (0.9, -7.25)
    ]
)



def plot_simple():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    # log_file = "/home/myles/Programming/neural_clbf/logs/simple_system/first/version_2/checkpoints/epoch=151-step=14361.ckpt"
    log_file = "/home/myles/Programming/neural_clbf/logs/simple_system_with_wind/config0/version_0/checkpoints/epoch=155-step=14925.ckpt"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)
    neural_controller.disable_gurobi = True
    # Update parameters

    neural_controller.experiment_suite.experiments[0].plot_unsafe_region = True

   


    # Run the experiments and save the results
    neural_controller.experiment_suite.run_all_and_plot(
        neural_controller, display_plots=True
    )

    rollout_experiment = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        SimpleWithObstacle.X,
        "x",
        SimpleWithObstacle.Y,
        "y",
        scenarios=[{}],
        n_sims_per_start=1,
        t_sim=15.0,
    )

    rollout_experiment.run_and_plot(neural_controller, True)

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
