# generate lj data. 
from openmmtools import testsystems
from simtk.openmm.app import *
import simtk.unit as unit

import logging

import numpy as np

from openmmtools.constants import kB
from openmmtools import respa, utils

logger = logging.getLogger(__name__)

# Energy unit used by OpenMM unit system
from openmmtools import states, integrators
import time
import numpy as np
import sys
import os

import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--simulation', type=str, 
#                     help='What simulation to generate, candidates: lj, tip3p, tip4p, rpbe. ')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed. ')
parser.add_argument('--num_train_rounds', type=int, default=15,
                    help='Rounds of MD simulation to run. ')
parser.add_argument('--num_test_rounds', type=int, default=3,
                    help='Rounds of MD simulation to run. ')
parser.add_argument('--sequence_length', type=int, default=1000,
                    help='Length of the entire MD trajectory to generate. ')
# parser.add_argument('--train_ratio', type=float, default=0.9,
#                     help='Ratio of training set is `train_ratio`, ratio of test set is `1 - train_ratio`. ')
parser.add_argument('--window_size', type=int, default=50+200,
                    help='Window size for each piece of training/testing data, `window_size = observed_length + extrap_length`, e.g. 15 = 10 + 5. ')
# 50 is the maximum window size we 'might' be using: for training and testing, we are only using the last 200  

args = parser.parse_args()

assert(args.sequence_length % args.window_size == 0) 

# All seeds need to be fixed to make sure the generated data are consistent across all baselines
import warnings
import random
import torch
from numba import njit
warnings.filterwarnings("ignore")


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

@njit
def set_seed(value):
    random.seed(value)
    np.random.seed(value)

set_seed(args.seed)
seed_torch(args.seed)

print("Generating tip4p data...") 
print("Fix all seed to: ", args.seed)


def get_rotation_matrix():
    """ 
        Function shared by lj, tip4p, 
        Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    angles = np.random.uniform(-1.0, 1.0, size=(3,)) * np.pi
    print(f'Using angle: {angles}')
    Rx = np.array([[1., 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]], dtype=np.float32)
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]], dtype=np.float32)
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]], dtype=np.float32)
    rotation_matrix = np.matmul(Rz, np.matmul(Ry, Rx))

    return rotation_matrix

def center_positions(pos):
    offset = np.mean(pos, axis=0)
    return pos - offset, offset

BOX_SCALE = 2
DT = 2

def generate_per_round(round): 
    # for round in range(args.num_rounds):
    print(f'Running round: {round}') 
    '''
    P_NUM = 258     # Number of particles 
    fluid = testsystems.LennardJonesFluid(nparticles=P_NUM, reduced_density=0.50, shift=True)
    [topology, system, positions] = fluid.topology, fluid.system, fluid.positions
    ''' 
    waterbox = testsystems.WaterBox(
        box_edge=2 * unit.nanometers,
        model='tip4pew')
    [topology, system, positions] = [waterbox.topology, waterbox.system, waterbox.positions]
    
    p_num = positions.shape[0] // 3

    
    R = get_rotation_matrix()
    positions = positions.value_in_unit(unit.angstrom)
    positions, off = center_positions(positions)
    positions = np.matmul(positions, R)
    positions += off
    positions += np.random.randn(positions.shape[0], positions.shape[1]) * 0.005
    positions *= unit.angstrom

    timestep = DT * unit.femtoseconds
    # temperature = 100 * unit.kelvin
    temperature = 300 * unit.kelvin # for tip4p 
    chain_length = 10
    # friction = 25. / unit.picosecond
    friction = 1. / unit.picosecond # for tip4p 
    num_mts = 5
    num_yoshidasuzuki = 5

    integrator1 = integrators.NoseHooverChainVelocityVerletIntegrator(system,
                                                                      temperature,
                                                                      friction,
                                                                      timestep, chain_length, num_mts, num_yoshidasuzuki)

    ''' If seed is set to 0 (which is the default value assigned), a unique seed is 
    chosen when a Context is created from this Force. This is done to ensure that 
    each Context receives unique random seeds without you needing to set them explicitly.

    current_seed = random.randint(-1000, 1000) 
    print("Seed for this round:", current_seed) 
    integrator1.setRandomNumberSeed(current_seed)
    '''

    simulation = Simulation(topology, system, integrator1)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    simulation.minimizeEnergy(tolerance=1*unit.kilojoule/unit.mole)
    simulation.step(1)

    ''' please re-factor the following code in compliance with the LG-ODE input format 
    os.makedirs(f'./lj_data/', exist_ok=True)
    dataReporter_gt = StateDataReporter(f'./log_nvt_lj_{seed}.txt', 50, totalSteps=50000,
        step=True, time=True, speed=True, progress=True, elapsedTime=True, remainingTime=True,
        potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True,
                                     separator='\t')
    simulation.reporters.append(dataReporter_gt)

    for t in range(1000):
        if (t+1)%100 == 0:
            print(f'Finished {(t+1)*50} steps')
        state = simulation.context.getState(getPositions=True,
                                             getVelocities=True,
                                             getForces=True,
                                             enforcePeriodicBox=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        vel = state.getVelocities(asNumpy=True).value_in_unit(unit.meter / unit.second)
        force = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)
        np.savez(f'../md_dataset/lj_data/data_{seed}_{t}.npz',
                 pos=pos,
                 vel=vel,
                 forces=force)
        simulation.step(50)
    '''

    dataReporter_gt = StateDataReporter(f'{log_path}/log_nvt_lj_{round}.txt', 50, totalSteps=50000,
        step=True, time=True, speed=True, progress=True, elapsedTime=True, remainingTime=True,
        potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True,
                                     separator='\t')
    simulation.reporters.append(dataReporter_gt)
    


    loc_acc = list()
    vel_acc = list()
    force_acc = list()
    # edges = list() # sorry, but such edge does not exist on the global scale 
    # timestamps_acc = np.arange(1000)
    # timestamps_acc = np.repeat(timestamps_acc, P_NUM, axis=-1) # shape: (1000, P_NUM)



    for i in range(args.sequence_length): 
        t = time.time()



        if (i + 1) % 100 == 0:
            print("Iter: {}, Simulation time: {}; finished {} steps".format(i, time.time() - t, (i + 1) * 50))

        state = simulation.context.getState(getPositions=True,
                                             getVelocities=True,
                                             getForces=True,
                                             enforcePeriodicBox=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        vel = state.getVelocities(asNumpy=True).value_in_unit(unit.meter / unit.second)
        force = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)

        loc_acc.append(pos)
        vel_acc.append(vel)
        force_acc.append(force)

        simulation.step(50)
                

        # radius_graph = None # TBD
        # edges.append(radius_graph)
    
    loc_acc = np.asarray(loc_acc) # [1000, 258, 3]
    vel_acc = np.asarray(vel_acc) # [1000, 258, 3]
    force_acc = np.asarray(force_acc) # [1000, 258, 3]
    
    # train_range = args.sequence_length * args.train_ratio

    loc_list, vel_list, force_list, timestamps_list = [], [], [], []

    # Now, the different windows are non-overlapping 
    for i in range(args.window_size, args.sequence_length + 1, args.window_size): # generating training/testing data pieces 
        loc_piece = loc_acc[i - args.window_size : i]
        vel_piece = vel_acc[i - args.window_size : i]
        force_piece = force_acc[i - args.window_size : i]
        # timestamps_piece = timestamps_acc[i - args.window_size : i]

        assert(len(loc_piece) == args.window_size) 

        loc_list.append(loc_piece)
        vel_list.append(vel_piece)
        force_list.append(force_piece)
        # timestamps_list.append(timestamps_piece)

        '''
        if (i < train_range): 
            loc_train.append(loc_piece)
            vel_train.append(vel_piece)
            force_train.append(force_piece)
            timestamps_train.append(timestamps_piece)
        elif (i >= train_range + args.window_size): 
            loc_test.append(loc_piece)
            vel_test.append(vel_piece)
            force_test.append(force_piece)
            timestamps_test.append(timestamps_piece)
        '''


    loc_list = np.asarray(loc_list) # [4, 250, 258, 3]
    vel_list = np.asarray(vel_list) # [4, 250, 258, 3]
    force_list = np.asarray(force_list) # [4, 250, 258, 3]
    # timestamps_list = np.asarray(timestamps_list) # [981, 20, 258]
    
    return loc_list, vel_list, force_list # , timestamps_list 

'''
loc_train = np.asarray(loc_train)
vel_train = np.asarray(vel_train)
force_train = np.asarray(force_train)
timestamps_train = np.asarray(timestamps_train)

loc_test = np.asarray(loc_test)
vel_test = np.asarray(vel_test)
force_test = np.asarray(force_test)
timestamps_test = np.asarray(timestamps_test)
'''


suffix = '_tip4p' # '_lj 
store_path = f'data/larger_tip4p_data_{args.window_size}'
log_path = f'data/larger_tip4p_data_{args.window_size}/logs'
os.makedirs(store_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)
print(args)
with open(store_path + '/' + 'generation_specs.txt', 'w') as f: 
    print(args, file=f)


# lg-ode style file format 
loc_train = list()
vel_train = list()
force_train = list()
# edges = list() # sorry, but such edge does not exist on the global scale 
# timestamps_train = list()

for round in range(args.num_train_rounds): 
    # loc_list, vel_list, force_list = generate_per_round(round)
    while True:
        try:
            loc_list, vel_list, force_list = generate_per_round(round) 
            # If no error occurs, break out of the loop
            break
        except ValueError:
            loc_list, vel_list, force_list = None, None, None 
            print("Encountering ValueError, retrying...")
    loc_train.append(loc_list)
    vel_train.append(vel_list)
    force_train.append(force_list)
    # timestamps_train.append(timestamps_list)

loc_train = np.concatenate(loc_train) # [15 * 4, 250, 258, 3]
vel_train = np.concatenate(vel_train) # [15 * 4, 250, 258, 3]
force_train = np.concatenate(force_train) # [15 * 4, 250, 258, 3]
# timestamps_train = np.concatenate(timestamps_train) # [9*981, 20, 258]

loc_train = np.swapaxes(loc_train, 1, 2) # [15 * 4, 258, 250, 3]
vel_train = np.swapaxes(vel_train, 1, 2) # [15 * 4, 258, 250, 3]
force_train = np.swapaxes(force_train, 1, 2) # [15 * 4, 258, 250, 3]
# timestamps_train = np.swapaxes(timestamps_train, 1, 2) # [9*981, 258, 20]

m, n, k, _ = loc_train.shape
# timestamps_train = np.repeat(np.arange(k), m*n).reshape(m, n, k)
timestamps_train = np.repeat(np.arange(k).reshape(1, -1), m*n, axis=0).reshape(m, n, k)


loc_test = list()
vel_test = list()
force_test = list()
# edges = list() # sorry, but such edge does not exist on the global scale 
# timestamps_test = list()

for round in range(args.num_test_rounds): 
    loc_list, vel_list, force_list = generate_per_round(round + args.num_train_rounds)
    loc_test.append(loc_list)
    vel_test.append(vel_list)
    force_test.append(force_list)
    # timestamps_test.append(timestamps_list)

loc_test = np.concatenate(loc_test)
vel_test = np.concatenate(vel_test)
force_test = np.concatenate(force_test)
# timestamps_test = np.concatenate(timestamps_test)

loc_test = np.swapaxes(loc_test, 1, 2)
vel_test = np.swapaxes(vel_test, 1, 2)
force_test = np.swapaxes(force_test, 1, 2)
# timestamps_test = np.swapaxes(timestamps_test, 1, 2)

m, n, k, _ = loc_test.shape
# timestamps_test = np.repeat(np.arange(k), m*n).reshape(m, n, k)
timestamps_test = np.repeat(np.arange(k).reshape(1, -1), m*n, axis=0).reshape(m, n, k)




# Dump training dataset 
np.save(store_path + '/loc_train' + suffix + '.npy', loc_train)
np.save(store_path + '/vel_train' + suffix + '.npy', vel_train)
np.save(store_path + '/force_train' + suffix + '.npy', force_train)
np.save(store_path + '/times_train' + suffix + '.npy', timestamps_train)

# Dump testing dataset 
np.save(store_path + '/loc_test' + suffix + '.npy', loc_test)
np.save(store_path + '/vel_test' + suffix + '.npy', vel_test)
np.save(store_path + '/force_test' + suffix + '.npy', force_test)
np.save(store_path + '/times_test' + suffix + '.npy', timestamps_test)





