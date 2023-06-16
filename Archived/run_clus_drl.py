import gym
from clusgym_env  import MCSEnv
from callback_simple import Callback
from tensorforce.execution import Runner
import gym.wrappers 
import numpy as np
import tensorforce
import copy
import tensorflow as tf
import os


timesteps = 200
num_parallel = 32
seed = 30
eleNames = ['Ni']
eleNums = [8]
clus_seed = 100
save_dir = './result_multi_env/'

def setup_env(recording=False, structure=None, structure_idx=None):
    
    # Set up gym
    MCS_gym = MCSEnv(eleNames=eleNames,
                     eleNums=eleNums,
                     clus_seed=clus_seed,
                     observation_fingerprints=True, 
                     save_dir = save_dir,
                     timesteps = timesteps,
                     save_every = 1,
                    )
    
    #if recording:
    # Wrap the gym to provide video rendering every 50 steps
        #MCS_gym = gym.wrappers.Monitor(MCS_gym, 
                                         #os.path.join(save_dir, 'vid'), 
                                         #force=True,
                                        #video_callable = lambda episode_id: (episode_id+1)%30==0) #every 50, starting at 51
    
    #Convert gym to tensorforce environment
    env = tensorforce.environments.OpenAIGym(MCS_gym,
                                         max_episode_timesteps=timesteps,
                                         visualize=False)
    
    return env
    
env = setup_env().environment.env
print('initial energy', env.initial_energy)

print(env.atoms)
print(env.atoms.get_positions())
from ase.io import write
write("initial_clus.traj", env.atoms)

from tensorforce.agents import Agent
tf.random.set_seed(seed)
agent = Agent.create(
    agent=dict(type='trpo'),
#     agent=dict(type='trpo', critic_network='auto', critic_optimizer=1.0),
    environment=setup_env(), 
    batch_size=1,
    learning_rate=1e-3,
    memory = 50000,
#     memory = dict(type='replay',capacity=10000),
    max_episode_timesteps = timesteps,
    exploration=dict(
        type='decaying', unit='timesteps', decay='exponential',
        initial_value=0.2, decay_steps=50000, decay_rate=0.5 #10000, 1000000
    ),
    
    parallel_interactions = num_parallel,
    
)
    


#plot_frequency --> plotting energy and trajectories frequency
callback = Callback(save_dir).episode_finish

runner2 = Runner(
    agent=agent,
    environment=setup_env(recording=False),
    max_episode_timesteps=timesteps,
)

# %prun runner.run(num_episodes=2, callback=callback, callback_episode_frequency=1)

# callback_episode_frequency --> saving results and trajs frequency
runner2.run(num_episodes=10000, callback=callback, callback_episode_frequency=1)
# runner2.run(num_episodes=100, evaluation=True)
# runner2.close()
