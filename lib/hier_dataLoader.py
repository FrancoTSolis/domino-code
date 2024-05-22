import numpy as np
import torch
from torch_geometric.data import DataLoader, Data
from torch.utils.data import DataLoader as Loader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
import scipy.sparse as sp
import random

import mdshare
import mdtraj as md
import os 

from lib.protein_data import get_ala2_atom_numbers, get_lj_atom_numbers, get_tip3p_atom_numbers, get_tip4p_atom_numbers 

# from run_models import fprint 

class ParseHierData(object): 
    def __init__(self, dataset_path, args, suffix):
        self.dataset_path = dataset_path
        self.suffix = suffix
        self.args = args 

        self.observe_scope = args.observe_scope
        self.extrap_scope = args.extrap_scope 
        self.entire_scope = self.observe_scope + self.extrap_scope
        self.raw_num = args.raw_num 

        assert (self.entire_scope <= self.raw_num)

        if args.data == "ala2":
            self.file_names = {
                "train": ["alanine-dipeptide-0-250ns-nowater.xtc", "alanine-dipeptide-1-250ns-nowater.xtc", ],
                "test": ["alanine-dipeptide-2-250ns-nowater.xtc", ],
                "topology": "alanine-dipeptide-nowater.pdb", 
            }
            self.atom_numbers = get_ala2_atom_numbers(distinguish=(not args.indistinguishable))
        elif args.data == "lj": 
            self.atom_numbers = get_lj_atom_numbers(distinguish=(not args.indistinguishable)) 
        elif args.data == "tip3p": 
            self.atom_numbers = get_tip3p_atom_numbers(distinguish=(not args.indistinguishable)) 
        elif args.data == "tip4p": 
            self.atom_numbers = get_tip4p_atom_numbers(distinguish=(not args.indistinguishable)) 

    def load_data(self, batch_size, data_type="train", device=None, fprint=None):
        self.batch_size = batch_size

        if self.suffix in ['_lj', '_tip3p', '_tip4p']:
            # Loading raw data 
            loc = np.load(self.dataset_path + '/loc_' + data_type + self.suffix + '.npy', allow_pickle=True) 
            vel = np.load(self.dataset_path + '/vel_' + data_type + self.suffix + '.npy', allow_pickle=True) # TODO: use vel!!!!!!!!!!!!! check github commit! 
            times = np.load(self.dataset_path + '/times_' + data_type + self.suffix + '.npy', allow_pickle=True) 

        
        if self.suffix == '_ala2':
            fprint(f"Loading {data_type} data from files {self.file_names[data_type]}...") 
            fprint(f"Using topology file {self.file_names['topology']}...") 

            topology = mdshare.fetch(self.file_names['topology'], self.dataset_path) 
            trajs = [md.load_xtc(os.path.join(self.dataset_path, fn), topology) for fn in self.file_names[data_type]]
            trajs = [t.center_coordinates().xyz for t in trajs] 

            loc = np.stack(trajs, axis=0) # TODO: check axis alignment    
            # swap the 1st and 2nd axes 
            loc = np.swapaxes(loc, 1, 2) # [n_data, n_atoms, seq_len, n_dim] 
        
        if (self.suffix == '_tip4p'): # remove the pseudo-atom 
            loc = loc[:, (np.mod(np.arange(loc.shape[1]), 4) < 3), :, :] 
            vel = vel[:, (np.mod(np.arange(vel.shape[1]), 4) < 3), :, :] 
            times = times[:, (np.mod(np.arange(times.shape[1]), 4) < 3), :] 
        
        assert (loc.shape[2] == self.raw_num) 
        if (self.suffix in ['_lj', '_tip3p', '_tip4p']): 
            assert (vel.shape[2] == self.raw_num)
            assert (times.shape[2] == self.raw_num)
        
        # We are taking elements from the front this time 
        loc = loc[:,:,:self.entire_scope,:]
        if (self.suffix in ['_lj', '_tip3p', '_tip4p']): 
            vel = vel[:,:,:self.entire_scope,:]
            times = times[:,:,:self.entire_scope]

        preserved_batch_idx = np.random.permutation(loc.shape[0])
        loc = loc[preserved_batch_idx]
        if (self.suffix in ['_lj', '_tip3p', '_tip4p']): 
            vel = vel[preserved_batch_idx]
            times = times[preserved_batch_idx]
        coord = loc # Unnormalized location features 

        if data_type == "train": 
            # find the maximum and minimum values for loc and vel respectively
            self.max_loc = np.max(loc[:,:,:self.observe_scope,:])
            self.min_loc = np.min(loc[:,:,:self.observe_scope,:])
            if (self.suffix in ['_lj', '_tip3p', '_tip4p']): 
                self.max_vel = np.max(vel[:,:,:self.observe_scope,:])
                self.min_vel = np.min(vel[:,:,:self.observe_scope,:])

            # shorthand operator for de-normalization
            # correct the excessive shifts when adding multiple levels of normalized values 
            '''
            corr_loc = (self.max_loc + self.min_loc) / (self.max_loc - self.min_loc) 
            corr_vel = (self.max_vel + self.min_vel) / (self.max_vel - self.min_vel) 
            self.correction_tsr = torch.tensor([corr_loc, corr_loc, corr_loc, corr_vel, corr_vel, corr_vel])  
            ''' 

        # Normalize to [-1, 1] 
        loc = (loc - self.min_loc) * 2 / (self.max_loc - self.min_loc) - 1 
        if (self.suffix in ['_lj', '_tip3p', '_tip4p']): 
            vel = (vel - self.min_vel) * 2 / (self.max_vel - self.min_vel) - 1

        self.num_graph = loc.shape[0]
        self.num_atoms = loc.shape[1]
        if (self.suffix in ['_lj', '_tip3p', '_tip4p']): 
            self.feature = loc[0][0][0].shape[0] + vel[0][0][0].shape[0]
            assert (self.feature == 6) 
        else: 
            self.feature = loc[0][0][0].shape[0] #  + vel[0][0][0].shape[0]
            assert (self.feature == 3)
        
        fprint("number graph in", data_type, "is %d" % self.num_graph) 
        fprint("number atoms in", data_type, "is %d" % self.num_atoms) 

        # Create a single DataLoader
        if (self.suffix in ['_lj', '_tip3p', '_tip4p']): # small moleucle dataset
            combined_data_loader = HierarchicalDataLoader(loc, vel, coord, times, self.entire_scope, self.args, data_type, device, self.atom_numbers) 
        else: # protein dataset 
            combined_data_loader = HierarchicalDataLoader(loc, None, coord, None, self.entire_scope, self.args, data_type, device, self.atom_numbers) 
        num_batch = len(combined_data_loader)

        return combined_data_loader, num_batch


class HierarchicalDataLoader(IterableDataset):
    def __init__(self, loc, vel, coord, times, entire_scope, args, data_type, device, atom_numbers): 
        super().__init__() 

        if (args.data not in ['lj', 'tip3p', 'tip4p']): 
            assert (vel is None); assert (times is None) 

        self.loc = loc
        # self.vel = vel
        self.coord = coord
        # self.times = times
        self.entire_scope = entire_scope 
        self.args = args 
        self.data_type = data_type 

        if (args.data in ['lj', 'tip3p', 'tip4p']): 
            self.vel = vel 
            # self.times = times 

        self.batch_size = args.batch_size 

        self.observe_scope = args.observe_scope
        self.extrap_scope = args.extrap_scope 

        self.num_levels = args.num_levels 
        self.max_segments = args.max_segments 

        self.device = device 

        self.select = None # use this to select certain time points to evaluate 

        self.selected_regions = None 


        self.atom_numbers = atom_numbers 

        # Process each level of ODE
        self.forecast_freqs = [None for _ in range(self.num_levels)]
        self.lookahead_freqs = [None for _ in range(self.num_levels)]
        self.forecast_scopes = [None for _ in range(self.num_levels)] 
        self.lookahead_scopes = [None for _ in range(self.num_levels)] 
        for level in range(self.num_levels):
            forecast_freq = args.ode_forecast_frequencies[level]
            lookahead_freq = int(forecast_freq * args.encoder_sampling_dilations[level]) 
            forecast_scope = self.extrap_scope if (level == 0) else args.ode_forecast_frequencies[level - 1] 
            lookahead_scope = int(forecast_scope * args.encoder_lookahead_scopes[level])  
        
            self.forecast_freqs[level] = forecast_freq 
            self.lookahead_freqs[level] = lookahead_freq 
            self.forecast_scopes[level] = forecast_scope 
            self.lookahead_scopes[level] = lookahead_scope 

    def set_select(self, select, total_dt=None): 
        self.select = select 
        if select is None: 
            self.selected_segment = None 
        else: 
            # raise Exception("deprecated")
            self.selected_segment = {} 
            for i in range(self.num_levels): # find the `[,)` domain that contains the selected time point 
                start = int(np.ceil(self.select / self.forecast_scopes[i]) - 1) * self.forecast_scopes[i] + self.observe_scope 
                end = start + self.forecast_scopes[i] 
                self.selected_segment[i] = (start, end) 
        
        self.selected_regions = None 

        if total_dt is not None: 
            self.selected_regions = [[] for _ in range(self.num_levels)] 
            for t in total_dt: 
                for i in range(self.num_levels): 
                    start = int(np.ceil(t / self.forecast_scopes[i]) - 1) * self.forecast_scopes[i] + self.observe_scope 
                    end = start + self.forecast_scopes[i] 
                    if ((start, end) not in self.selected_regions[i]): 
                        self.selected_regions[i].append((start, end)) 
        
        return 

    def __iter__(self):
        for batch_idx in range(0, len(self.loc), self.batch_size):

            # Initialize batches for each level
            batch_data = {
                "loc": torch.tensor(self.loc[batch_idx:batch_idx + self.batch_size], dtype=torch.float32, device=self.device),
                # "vel": torch.tensor(self.vel[batch_idx:batch_idx + self.batch_size], dtype=torch.float32, device=self.device),
                "coord": torch.tensor(self.coord[batch_idx:batch_idx + self.batch_size], dtype=torch.float32, device=self.device), 
                # "times": torch.tensor(self.times[batch_idx:batch_idx + self.batch_size], device=self.device),
            }

            if (self.args.data in ['lj', 'tip3p', 'tip4p']): 
                batch_data["vel"] = torch.tensor(self.vel[batch_idx:batch_idx + self.batch_size], dtype=torch.float32, device=self.device) 
                # batch_data["times"] = torch.tensor(self.times[batch_idx:batch_idx + self.batch_size], device=self.device) 


            # code from chatgpt, correctness to be verified 
            for level in range(self.num_levels): 
                batch_data[f"level_{level}"] = {
                    "encoder_mask": [],
                    "decoder_mask": [],
                    "encoder_decoder_segments": [], 
                }

            for _ in range(self.batch_size): 
                prev_decoder_timepoints = set()  # To keep track of decoder time points from the previous level
                for level in range(self.num_levels): 
                    # Initialize masks and segments for the current level
                    encoder_mask = np.zeros(self.entire_scope, dtype=bool)
                    decoder_mask = np.zeros(self.entire_scope, dtype=bool)
                    encoder_decoder_segments = []

                    if level == 0:
                        # Level 1: One segment, fixed time points
                        decoder_timepoints = set(range(self.observe_scope, self.entire_scope, self.forecast_freqs[level]))
                        encoder_timepoints = set(range(0, self.lookahead_scopes[level], self.lookahead_freqs[level]))
                        encoder_decoder_segments.append((0, self.observe_scope, self.entire_scope))
                    else:
                        # Level 2 and above: Sample segments
                        if (self.select is None) and (self.selected_regions is None):  
                            possible_segments = self.get_possible_segments(prev_decoder_timepoints, self.forecast_scopes[level])
                            assert(len(possible_segments) > self.max_segments) 
                            sampled_segments = random.sample(possible_segments, min(self.max_segments, len(possible_segments)))
                        elif (self.selected_regions is not None): 
                            assert (self.select is None)  
                            sampled_segments = self.selected_regions[level] 
                        else: 
                            # raise Exception("deprecated")  
                            start, end = self.selected_segment[level] 
                            sampled_segments = [(start, end)] 
                        decoder_timepoints = set()
                        encoder_timepoints = set()

                        for start, end in sampled_segments:
                            decoder_timepoints.update(range(start, end, self.forecast_freqs[level]))
                            # assert(start >= self.lookahead_scopes[level]) 
                            # encoder_start = max(0, start - self.lookahead_scopes[level])
                            # encoder_timepoints.update(range(encoder_start, start, self.lookahead_freqs[level]))
                            # encoder_decoder_segments.append((encoder_start, start, end))
                            # assert (self.lookahead_scopes[level] is None) 
                            # print("lookahead scope should be None here") 
                            encoder_timepoints.update([start, ]) # We are not doing lookahead for the higher levels 
                            encoder_decoder_segments.append((None, start, end))

                    # Update masks
                    encoder_mask[list(encoder_timepoints)] = True
                    decoder_mask[list(decoder_timepoints)] = True
                    prev_decoder_timepoints = decoder_timepoints

                    batch_data[f"level_{level}"]["encoder_mask"].append(torch.tensor(encoder_mask, device=self.device)) 
                    batch_data[f"level_{level}"]["decoder_mask"].append(torch.tensor(decoder_mask, device=self.device)) 
                    batch_data[f"level_{level}"]["encoder_decoder_segments"].append(encoder_decoder_segments) 


            yield batch_data

    def get_possible_segments(self, prev_decoder_timepoints, forecast_scope):
        segments = []
        sorted_timepoints = sorted(prev_decoder_timepoints)
        '''
        for i in range(len(sorted_timepoints) - 1):
            start = sorted_timepoints[i]
            end = min(sorted_timepoints[i + 1], start + forecast_scope)
            if end - start >= forecast_scope:
                segments.append((start, end))
        '''
        for i in range(len(sorted_timepoints)): 
            start = sorted_timepoints[i] 
            # assuming the next time point is always within the forecast scope 
            # assuming the current-layer forecast scope is equal to the previous-layer forecast frequency 
            end = start + forecast_scope 
            segments.append((start, end))
        return segments

    def __len__(self):
        if len(self.loc) % self.batch_size == 0:
            return len(self.loc) // self.batch_size
        else: 
            return len(self.loc) // self.batch_size + 1
    
    def transfer_one_graph(self, data, coord, loc, vel, time, time_begin=0, enc_scale=None):
        '''
        Regarding the source of coord, loc, vel: 
        1) From ground truth (teacher forcing) 
        2) From the base model's prediction (free running, requires backprop) 
        '''
        num_atoms, seq_length, _ = loc.shape  # Assuming loc and vel have the same shape

        if data not in ['lj', 'tip3p', 'tip4p']: 
            assert (vel is None)

        encoder_start, encoder_end = enc_scale 
        assert (time_begin == 0) # useless for now: we do not rely on time_begin to do rescaling 

        assert (coord.shape[-1] == 3) 
        assert (loc.shape[-1] == 3) 
        if data in ['lj', 'tip3p', 'tip4p']: 
            assert (vel.shape[-1] == 3) 
            x = torch.cat([loc, vel], dim=-1).view(-1, loc.shape[-1] + vel.shape[-1]) 
        else: 
            x = loc.view(-1, loc.shape[-1]) 

        x_coord = coord.view(-1, coord.shape[-1]) 

        # Adjust time by time_begin and flatten
        x_pos = (time - time_begin).repeat(num_atoms) # .flatten()

        '''
        Caution! Time normalization subject to change 
        '''
        # x_pos = x_pos / self.entire_scope # Normalize to [0, 1] 
        # change to local normalization 
        assert (encoder_start < encoder_end) 
        assert (time[0] == encoder_start) 
        assert (time[-1] < encoder_end) 
        x_pos = (x_pos - encoder_start) / (encoder_end - encoder_start) # Normalize to [0, 1]

        # Number of time steps per atom (all atoms have the same number of time steps)
        y = torch.full((num_atoms,), seq_length, dtype=torch.long) 

        # Unsqueeze to add a new dimension and then repeat
        atom_number = self.atom_numbers.unsqueeze(1).repeat(1, int(x.shape[0] / self.atom_numbers.shape[0])) 

        # beware of the difference between x and x_coord 
        graph_data = Data(x=x, y=y, pos=x_pos, x_coord=x_coord, atom_number=atom_number)  
        return graph_data

    def transfer_one_latent(self, latent, graph_latent):
        num_atoms, seq_length, _ = latent.shape


        y = torch.full((num_atoms,), seq_length, dtype=torch.long)

        graph_data = Data(x=latent.view(-1, latent.shape[-1]), y=y, graph_x=graph_latent.view(-1, graph_latent.shape[-1])) 

        return graph_data 

