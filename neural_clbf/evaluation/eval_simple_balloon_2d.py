import torch
import matplotlib
from neural_clbf.controllers import NeuralCLBFController, Controller
import os
import torch.distributed as dist
from neural_clbf.systems import SimpleBalloon2d
from neural_clbf.experiments import RolloutStateSpaceExperiment

# from  
matplotlib.use('TkAgg')


def plot_simple():
    class NoControl(Controller):
        def __init__(self, dynamics_model, experiment_suite, controller_period):
            super().__init__(dynamics_model, experiment_suite, controller_period)
        def u(self, x):
            return torch.zeros((x.shape[0], 1))

    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_file = "/home/myles/Programming/neural_clbf/logs/simple_balloon_2d/version_0/checkpoints/epoch=100-step=5352.ckpt"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)


    neural_controller.disable_gurobi = True
    neural_controller.experiment_suite.experiments[0].domain = [(-15.0, 15.0), (-15.0, 15.0)]
    # Update parameters

    neural_controller.experiment_suite.run_all_and_plot(neural_controller, True)
    
    start_x = torch.tensor(
        [
            (0.0, 0.0),
            (1.0, 1.0)
        ]
    )

    rollout_experiment = RolloutStateSpaceExperiment(
            "Rollout",
            start_x,
            SimpleBalloon2d.X,
            "x",
            SimpleBalloon2d.Z,
            "Z",
            scenarios=[{}],
            n_sims_per_start=1,
            t_sim=15.0,
        )

    rollout_experiment.run_and_plot(NoControl(neural_controller.dynamics_model, neural_controller.experiment_suite, neural_controller.controller_period), display_plots=True)
    rollout_experiment.run_and_plot(neural_controller, display_plots=True)

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
