import gym
from gym import spaces
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.optimize.bfgs import BFGS
from ase.visualize.plot import plot_atoms
from ase.io import write
from asap3 import EMT
import copy
from symmetry_function import make_snn_params, wrap_symmetry_functions
from clus_utils import checkSimilar, addAtoms, fixOverlap



DIRECTION =[np.array([1,0,0]),
           np.array([-1,0,0]),
           np.array([0,1,0]),
           np.array([0,-1,0]),
           np.array([0,0,1]),
           np.array([0,0,-1]),
          ]


class MCSEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, 
                 eleNames=None,
                 eleNums=None,
                 clus_seed=None,
                 save_dir=None,
                 observation_fingerprints = True,
                 observation_forces=True,
                 observation_positions = True,
                 descriptors = None,
                 timesteps = None,
                 save_every = None,
                 plot_every = None,
                 
                ):
        
        self.eleNames = eleNames
        self.eleNums  = eleNums
        self.eleRadii = [covalent_radii[atomic_numbers[ele]] for ele in self.eleNames]
        self.avg_radii = sum(self.eleRadii) / len(self.eleNums)
        self.clus_seed = clus_seed
        self.descriptors = descriptors
        self.timesteps = timesteps
        self.save_every = save_every
        self.plot_every = plot_every
        
        self.episodes = 0

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.history_dir = os.path.join(save_dir, 'history')
        self.plot_dir = os.path.join(save_dir, 'plots')
        self.traj_dir = os.path.join(save_dir, 'trajs')
        self.episode_min_traj_dir = os.path.join(save_dir, 'episode_min')
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        if not os.path.exists(self.traj_dir):
            os.makedirs(self.traj_dir)
        if not os.path.exists(self.episode_min_traj_dir):
            os.makedirs(self.episode_min_traj_dir)
        
        
        self.initial_atoms, self.snn_params = self._get_initial_clus()
        self.initial_atoms.set_calculator(EMT())
        self.initial_positions = self.initial_atoms.get_positions()

        self.new_initial_atoms = self.initial_atoms.copy()
        self.initial_energy = self.initial_atoms.get_potential_energy()
        
        self.relative_energy = 0.0
        self.initial_forces = self.initial_atoms.get_forces()
        
        self.atoms = self.initial_atoms.copy()
        self.clus_size = len(self.atoms)

        self.observation_positions = observation_positions
        self.observation_fingerprints = observation_fingerprints
        self.observation_forces = observation_forces


        self.fps, self.fp_length = self._get_fingerprints(self.atoms)
        self.episode_initial_fps = self.fps.copy()
        self.positions = self.atoms.get_positions()
        
        self.found_new_min = 0

        # Define the possible actions

        self.action_space = spaces.Dict({'atom_selection': spaces.Discrete(self.clus_size),
                                         'movement':spaces.Discrete(6) } ) 
        #Define the observation space
        self.observation_space = self._get_observation_space()
        
        # Set up the initial atoms
        self.reset()
        
        return

      # open AI gym API requirements
    def step(self, action):
        '''
            The agent will perform an action based on the atom selection and movement. 
            The selected atom's coordinate will be shifted from the current position.
            if the movement from the current position results in overlapped atoms,
            the reward wil be -1000
            if there is no overlapping, geometry relaxation will be performed.
            if it found  a new minimum configuration, the step will get reward based on the 
            relative energy from the initial minimum.
            if it relaxed to an existing minimum found earlier in the episode,
            the reward will be zero.
        '''
        
        reward = 0  

        self.atom_selection = action['atom_selection']
        self.movement = action['movement']
        self.shift = DIRECTION[self.movement]
       
        self.done= False
        episode_over = False

        save_path_min = None

    	#shifting the position of the selected atom in the cluster
        self.atoms[self.atom_selection].position = self.atoms.get_positions()[self.atom_selection] + self.shift * 2.5

        dist_after_move = self.atoms.get_all_distances()[self.atom_selection]
        z1 = [ dist for k, dist in enumerate(dist_after_move) if k != self.atom_selection and dist < self.avg_radii ]
	
        if len(z1) > 0:     	#checking for overlapping atoms after movement
            reward -= 1000.0
        else:			#minimization
            dyn = BFGS(atoms=self.atoms, logfile=None, trajectory= save_path_min)
            converged = dyn.run(fmax=0.02)
            self.relative_energy = self._get_relative_energy()

            self.all_minima['minima'].append(self.atoms.copy())
            self.all_minima['energies'].append(self._get_relative_energy())
            self.all_minima['timesteps'].append(self.history['timesteps'][-1] + 1)
            self.all_minima['positions'].append(self.atoms.positions.copy())
		
		#checking the similarilty of the relaxed cluster minimum  between cluster minima already found
            bool_list = []
            for clus in self.minima['minima']:
                bool_list.append(checkSimilar(self.atoms, clus)) 
		
            if any(bool_list): #if the minimum is already found, reward is zero
                reward += 0.0
            else:				# a new minima found
                reward +=  100 * np.exp((-1) * self.relative_energy)
                self.done = True
                self.found_new_min =  1
		
                self.minima['minima'].append(self.atoms.copy())
                self.minima['energies'].append(self._get_relative_energy())
                self.minima['timesteps'].append(self.history['timesteps'][-1] + 1)
                self.minima['positions'].append(self.atoms.positions.copy())
	
	
        #Fingerprints after step action
        self.fps, self.fp_length = self._get_fingerprints(self.atoms)
        print("self.fps, self.fps_length")
        print(self.fps, self.fp_length)
        
        #Get the new observation
        observation = self._get_observation()
        
        if self.done:
            episode_over = True

        #Update the history for the rendering after each step
        self.relative_energy = self._get_relative_energy()
       	
        self.trajectories.append(self.atoms.copy())

        self.history['timesteps'] = self.history['timesteps'] + [self.history['timesteps'][-1] + 1]
        self.history['energies'] = self.history['energies'] + [self.relative_energy]
        self.history['positions'] = self.history['positions'] + [self.atoms.get_positions(wrap=False).tolist()]
        self.history['scaled_positions'] = self.history['scaled_positions'] + [self.atoms.get_scaled_positions(wrap=False).tolist()]
        if self.observation_fingerprints:
            self.history['fingerprints'] = self.history['fingerprints'] + [self.fps.tolist()]
            self.history['initial_fps'] = self.history['initial_fps'] + [self.episode_initial_fps.tolist()]

        self.episode_reward += reward

        #if len(self.history['actions'])-1 >= self.total_steps:
        if len(self.history['timesteps'])-1 >= self.total_steps:
            episode_over = True
            
        if episode_over: 
            #self.total_force_calls += self.calc.force_calls
            self.min_idx = int(np.argmin(self.minima['energies']))
            if self.episodes % self.save_every == 0:
                self.save_episode()
                self.save_traj()
                
            self.episodes += 1
            
        return observation, reward, episode_over, {}

    def _get_initial_clus(self):
        self.initial_atoms, self.elements = self._generate_clus()

        if self.descriptors is None:
            Gs = {}
            Gs["G2_etas"] = np.logspace(np.log10(0.05), np.log10(5.0), num=4)
            Gs["G2_rs_s"] = [0] * 4
            Gs["G4_etas"] = [0.005]
            Gs["G4_zetas"] = [1.0]
            Gs["G4_gammas"] = [+1.0, -1]
            Gs["cutoff"] = 6.5

            G = copy.deepcopy(Gs)

            # order descriptors for simple_nn
            cutoff = G["cutoff"]
            G["G2_etas"] = [a / cutoff**2 for a in G["G2_etas"]]
            G["G4_etas"] = [a / cutoff**2 for a in G["G4_etas"]]
            descriptors = (
                G["G2_etas"],
                G["G2_rs_s"],
                G["G4_etas"],
                G["cutoff"],
                G["G4_zetas"],
                G["G4_gammas"],
            )
        self.snn_params = make_snn_params(self.elements, *descriptors)                
        return self.initial_atoms, self.snn_params
        
    
    def save_episode(self):
        save_path = os.path.join(self.history_dir, '%d_%f_%f_%f.npz' %(self.episodes, self.minima['energies'][self.min_idx],
                                                                   self.initial_energy, self.episode_reward))
        np.savez_compressed(save_path, 
             initial_energy = self.initial_energy,
             energies = self.history['energies'],
             #actions = self.history['actions'],
             scaled_positions = self.history['scaled_positions'],
             fingerprints = self.history['fingerprints'],
             initial_fps = self.history['initial_fps'],
             minima_energies = self.minima['energies'],
             minima_steps = self.minima['timesteps'],
             reward = self.episode_reward,
             episode = self.episodes,
            )
        return
    
    def save_traj(self):      
        save_path = os.path.join(self.traj_dir, '%d_%f_%f_%f_full.traj' %(self.episodes, self.minima['energies'][self.min_idx]
                                                                      , self.initial_energy, self.episode_reward))
        episode_min_path = os.path.join(self.episode_min_traj_dir, '%d_%f_%f_%f_full.traj' %(self.episodes, self.minima['energies'][self.min_idx]
                                                                      , self.initial_energy, self.episode_reward))
        trajectories = []
        for atoms in self.trajectories:
            atoms.set_calculator(EMT())
            trajectories.append(atoms)
        write(save_path, trajectories)

        write(episode_min_path, self.minima['minima'])
        
        return

    def reset(self):
        #Copy the initial atom and reset the calculator
            
        self.new_initial_atoms, self.snn_params = self._get_initial_clus()
        self.atoms = self.new_initial_atoms.copy()
        self.atoms.set_calculator(EMT())
        self.episode_reward = 0
        self.total_steps = self.timesteps
      
        #Reset the list of identified minima and their energies and positions
        self.minima = {}
        self.minima['minima'] = [self.atoms.copy()]
        self.minima['energies'] = [0.0]
        self.minima['positions'] = [self.atoms.positions.copy()]
        self.minima['timesteps'] = [0]
       

        self.all_minima = {}
        self.all_minima['minima'] = [self.atoms.copy()]
        self.all_minima['energies'] = [0.0]
        self.all_minima['positions'] = [self.atoms.positions.copy()]
        self.all_minima['timesteps'] = [0]

        #Set the energy history
        results = ['timesteps', 'energies', 'positions', 'scaled_positions', 'fingerprints', 'initial_fps']
        self.history = {}
        for item in results:
            self.history[item] = []

        self.history['timesteps'] = [0]
        self.history['energies'] = [0.0]
        self.history['positions'] = [self.atoms.get_positions().tolist()]
        self.history['scaled_positions'] = [self.atoms.get_scaled_positions().tolist()]
        if self.observation_fingerprints:
            self.fps, fp_length = self._get_fingerprints(self.atoms)
            self.initial_fps = self.fps
            self.episode_initial_fps = self.fps
            self.history['fingerprints'] = [self.fps.tolist()]
            self.history['initial_fps'] = [self.episode_initial_fps.tolist()]
        
        self.trajectories = [self.atoms.copy()]        
        
        return self._get_observation()
    

    def render(self, mode='rgb_array'):

        if mode=='rgb_array':
            # return an rgb array representing the picture of the atoms
            
            #Plot the atoms
            fig, ax1 = plt.subplots()
            plot_atoms(self.atoms, 
                       ax1, 
                       rotation='48x,-51y,-144z', 
                       show_unit_cell =0)
            
            ax1.set_ylim([0,25])
            ax1.set_xlim([-2, 20])
            ax1.axis('off')
            ax2 = fig.add_axes([0.35, 0.85, 0.3, 0.1])
            
            #Add a subplot for the energy history overlay           
            ax2.plot(self.history['timesteps'],
                     self.history['energies'])
            
            ax2.plot(self.minima['timesteps'],
                    self.minima['energies'],'o', color='r')
        

            ax2.set_ylabel('Energy [eV]')
            
            #Render the canvas to rgb values for the gym render
            plt.draw()
            renderer = fig.canvas.get_renderer()
            x = renderer.buffer_rgba()
            img_array = np.frombuffer(x, np.uint8).reshape(x.shape)
            plt.close()
            
            #return the rendered array (but not the alpha channel)
            return img_array[:,:,:3]
            
        else:
            return
    
    def close(self):
        return
    
    
    def _get_relative_energy(self):
        return self.atoms.get_potential_energy() - self.initial_energy

    def _get_observation(self):
        # helper function to get the current observation, which is just the position
        # of the free atoms as one long vector
           
        observation = {'energy':np.array(self._get_relative_energy()).reshape(1,)}
        
        if self.observation_fingerprints:
            observation['fingerprints'] = (self.fps - self.episode_initial_fps).flatten()
            
        
        observation['positions'] = self.atoms.get_scaled_positions().flatten()
            
        if self.observation_forces:
            observation['forces'] = self.atoms.get_forces().flatten()
        
        observation['found_new_min'] = np.array([self.found_new_min]).reshape(1,)
            
        return observation
    
    def _get_fingerprints(self, atoms):
        #get fingerprints from amptorch as better state space feature
        fps = wrap_symmetry_functions(self.atoms, self.snn_params)
        fp_length = fps.shape[-1]
        
        return fps, fp_length
    
    def _get_observation_space(self):  
        
        observation_space = spaces.Dict({'fingerprints': spaces.Box(low=-6,
                                            high=6,
                                            shape=(len(self.atoms)*self.fp_length, )),
                                        'positions': spaces.Box(low=-1,
                                            high=2,
                                            shape=(len(self.atoms)*3,)),
                                        'energy': spaces.Box(low=-1,
                                                    high=2.5,
                                                    shape=(1,)),
                                        'forces': spaces.Box(low= -2,
                                                            high= 2,
                                                            shape=(len(self.atoms)*3,)
                                                            ),
                                        'found_new_min': spaces.Box(low=-0.5,
                                                            high=1.5,
                                                            shape=(1,)),
                                        })

        return observation_space

    def _generate_clus(self):
        """
	Generate a random cluster configuration
	"""
        if self.clus_seed is not None:
            np.random.seed(self.clus_seed)

        ele_initial = [self.eleNames[0], self.eleNames[-1]]
        d = (self.eleRadii[0] + self.eleRadii[-1]) / 2
        clusm = Atoms(ele_initial, [(-d, 0.0, 0.0), (d, 0.0, 0.0)])
        clus = addAtoms(clusm, self.eleNames, self.eleNums, self.eleRadii, self.clus_seed)
        clus = fixOverlap(clus)
            
        elements = np.array(clus.symbols)
        _, idx = np.unique(elements, return_index=True)
        elements = list(elements[np.sort(idx)])
        
        return clus, elements
    
    
