import numpy as np
import itertools
from collections import Counter
from dscribe.descriptors import ACSF, SOAP
from ase.io import Trajectory

def generate_acsf_descriptor(traj_file,pos_list, metal_list):
    epsilon = [1,2,3,4,5,6,7]
    kappa = [0.5,1.0,1.5,2.0,2.5]
    eta = [0.01,0.03,0.06,0.1,0.2,0.4,1.0,2.5,5.0]
    R_s = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    lamb = [-1, 1]

    g2_params = [list(item) for item in itertools.product(eta, R_s)]
    g3_params = kappa
    g4_params = [list(item) for item in itertools.product(eta, epsilon, lamb)]
    #g5_params = g4_params

    # Set up: Instantiating an ACSF descripto
    species = [ 'H', "N"] + metal_list
    print('species:', species)
    acsf = ACSF(
        species=species,
        rcut=6.5,
        g2_params=g2_params,
        g3_params=g3_params,
        g4_params=g4_params,
        #g5_params=g5_params
    )

    # Create ACSF output for the system
    traj = Trajectory(traj_file)
    pos_list = pos_list
    acsf_traj  = acsf.create(traj, positions=pos_list, n_jobs=-1)

    return acsf_traj


def generate_soap_descriptor(traj_file,pos_list, metal_list):
    species = [ 'H', "N"] + metal_list
    r_cut = 6.5
    n_max = 9
    l_max = 10


    # Setting up the SOAP descriptor
    soap = SOAP(
        species=species,
        periodic=False,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,    )

    traj = Trajectory(traj_file)

    # Create ACSF output for the system
    pos_list = pos_list
    soap_traj  = soap.create(traj, positions=pos_list, n_jobs=-1)
    #print(soap_traj.shape)

    return soap_traj

