import matplotlib
matplotlib.use("Agg")
import gym
#from surface_seg.envs.mcs_env import MCSEnv
from clusgym_clus_move_com_ver12_mod6_expt5  import MCSEnv
import gym.wrappers
import numpy as np
import tensorforce 
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.execution import Runner
import os
import copy
from callback_simple import Callback
import random as rand

timesteps = 200
num_parallel = 32
seed = 30
eleNames = ['Cu']
eleNums = [20]
#clus_seed = 46852
save_dir = './result_multi_env_parallel_ver12_mod6_expt5/'



def setup_env(clus_seed, recording=False):
    
    # Set up gym
    #MCS_gym = MCSEnv(fingerprints=True, 
                    #permute_seed=None)
   
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
                                         #"./vid", 
                                         #force=True,
                                        #video_callable = lambda episode_id: (episode_id)%50==0) #every 50, starting at 51
    
    #Convert gym to tensorforce environment
    env = tensorforce.environments.OpenAIGym(MCS_gym,
                                         max_episode_timesteps=400,
                                         visualize=False)
    
    return env

"""
Create a environment for checking the intial energy and thermal energy
"""
#env = setup_env().environment.env
#print('initial energy', env.initial_energy)

for i in range(1000):
    clus_seed = rand.randint(0, 100000)
    

    env = setup_env(clus_seed).environment
#print('initial energy', env.initial_energy)
    print(i, clus_seed, env.initial_energy)
    with open('seed_energy.out', 'a+') as fh:
        fh.write(f"{clus_seed}, {env.initial_energy} \n")
    #print(env.atoms)
    #print(env.atoms.get_positions())

#from ase.io import write
#write("initial_clus.traj", env.atoms)


