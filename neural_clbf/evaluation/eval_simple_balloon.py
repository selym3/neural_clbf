import torch
import matplotlib
import matplotlib.pyplot as plt
from neural_clbf.controllers import NeuralCLBFController
import os
import torch.distributed as dist
import seaborn as sns

# from  
matplotlib.use('Agg')


def plot_simple():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_file = "/home/mhp58/neural_clbf/logs/simple_balloon/config0/version_3/checkpoints/epoch=124-step=7896.ckpt"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)
    neural_controller.disable_gurobi = True
    # Update parameters

    # neural_controller.controller_dt = 0.001
    neural_controller.experiment_suite.experiments[0].plot_unsafe_region = True

    neural_controller.experiment_suite.experiments[1].t_sim = 1.0

     # Run the experiments and save the results
    grid_df = neural_controller.experiment_suite.experiments[0].run(neural_controller)
    traj_df = neural_controller.experiment_suite.experiments[1].run(neural_controller)

    # Plot in 3D
    sns.set_theme(context="talk", style="white")
    ax = plt.axes(projection="3d")

    # Plot the trajectory
    print(traj_df)
    print('\n'.join(list(str(traj_df.state[i]) for i in range(2, 495, 5))))
    # print(traj_df["state"][:,0], traj_df["state"][:,1], traj_df["state"][:,2])
    ax.plot3D(traj_df["x"], traj_df["y"], traj_df["z"], "black")
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xticks(np.linspace(-0.75, 0.75, 2))
    ax.set_yticks(np.linspace(-0.75, 0.75, 2))
    ax.set_zticks(np.linspace(-0.75, 0.75, 2))
    
    ax.plot([], [], c="blue", label="V(x) = c")
    ax.tricontour(
        grid_df["x"],
        grid_df["y"],
        grid_df["V"],
        colors=["blue"],
        levels=[0.0],
    )
    
    plt.savefig('plot.png', dpi=300, bbox_inches='tight')


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
