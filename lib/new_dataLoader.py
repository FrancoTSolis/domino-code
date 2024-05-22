raise Exception("Deprecated. ") 

import numpy as np
import torch
from torch_geometric.data import DataLoader,Data
# https://pytorch-geometric.readthedocs.io/en/1.3.1/_modules/torch_geometric/data/dataloader.html#DenseDataLoader 
# from torch_geometric.data import DenseDataLoader,Data
from torch.utils.data import DataLoader as Loader
import scipy.sparse as sp
from tqdm import tqdm
import lib.utils as utils
from torch.nn.utils.rnn import pad_sequence

from lib.moving_average import decompose_trajectory 
from torch_geometric.data import Batch

class ParseData(object):

    def __init__(self, dataset_path,args,suffix='_springs5',mode="interp"):
        self.dataset_path = dataset_path
        self.suffix = suffix
        self.mode = mode
        self.random_seed = args.random_seed
        self.args = args
        self.total_step = args.total_ode_step
        self.cutting_edge = args.cutting_edge
        self.num_pre = args.extrap_num
        self.num_entire = args.entire_num 
        self.num_raw = args.raw_num 

        # decompose trajectory 
        self.n_hiers = args.n_hiers
        self.filter_sizes = args.filter_sizes

        self.max_loc = None
        self.min_loc = None
        self.max_vel = None
        self.min_vel = None

        self.correction_tsr = None 

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)



    def load_data(self,sample_percent,batch_size,data_type="train",batch_percent=None):
        self.batch_size = batch_size
        self.sample_percent = sample_percent
        print("Sampling is deprecated. ")

        # `cut_num` is used to limit the overall size of training / testing data
        # Deprecate this feature for now 
        '''
        if data_type == "train":
            cut_num = 20000
        else:
            cut_num = 5000
        '''

        # Loading Data
        loc = np.load(self.dataset_path + '/loc_' + data_type + self.suffix + '.npy', allow_pickle=True) # [:cut_num]
        vel = np.load(self.dataset_path + '/vel_' + data_type + self.suffix + '.npy', allow_pickle=True) # [:cut_num]
        # forces = np.load(self.dataset_path + '/forces_' + data_type + self.suffix + '.npy', allow_pickle=True) # [:cut_num]
        # edges = np.load(self.dataset_path + '/edges_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num]  # [500,5,5]
        times = np.load(self.dataset_path + '/times_' + data_type + self.suffix + '.npy', allow_pickle=True) # [:cut_num]  # 【500，5]

        if batch_percent is not None: 
            print("Keeping %f of %s data" % (batch_percent, data_type)) 
            preserved_batch_idx = np.random.choice(np.arange(loc.shape[0]), int(loc.shape[0] * batch_percent), replace=False)
            # print(loc.shape[0])
            # print(len(preserved_batch_idx))
            # print(preserved_batch_idx)
        else: 
            print("Keeping full %s data" % data_type) 
            preserved_batch_idx = np.random.permutation(loc.shape[0])
        
        loc = loc[preserved_batch_idx]
        vel = vel[preserved_batch_idx]
        times = times[preserved_batch_idx]

        
        self.num_graph = loc.shape[0]
        self.num_atoms = loc.shape[1]
        self.feature = loc[0][0][0].shape[0] + vel[0][0][0].shape[0]
        print("number graph in   "+data_type+"   is %d" % self.num_graph)
        print("number atoms in   " + data_type + "   is %d" % self.num_atoms)
    
        loc_hiers = decompose_trajectory(loc, self.n_hiers, self.filter_sizes) 
        vel_hiers = decompose_trajectory(vel, self.n_hiers, self.filter_sizes) 
        
        


        if self.suffix == "_springs5" or self.suffix == "_charged5" or self.suffix == "_lj":
            # Normalize features to [-1, 1], across test and train dataset

            coord = loc # Unnormalized location features 
            
            if self.max_loc == None:
                loc, max_loc, min_loc = self.normalize_features(loc,
                                                                self.num_atoms)  # [num_sims,num_atoms, (timestamps,2)]
                vel, max_vel, min_vel = self.normalize_features(vel, self.num_atoms)
                self.max_loc = max_loc
                self.min_loc = min_loc
                self.max_vel = max_vel
                self.min_vel = min_vel

                corr_loc = 2 * (max_loc + min_loc) / (max_loc - min_loc) 
                corr_vel = 2 * (max_vel + min_vel) / (max_vel - min_vel) 
                self.correction_tsr = torch.tensor([corr_loc, corr_loc, corr_loc, corr_vel, corr_vel, corr_vel])  
            else:
                loc = (loc - self.min_loc) * 2 / (self.max_loc - self.min_loc) - 1
                vel = (vel - self.min_vel) * 2 / (self.max_vel - self.min_vel) - 1
            
            for i in range(self.n_hiers): 
                loc_hiers[i] = (loc_hiers[i] - self.min_loc) * 2 / (self.max_loc - self.min_loc) - 1 
                vel_hiers[i] = (vel_hiers[i] - self.min_vel) * 2 / (self.max_vel - self.min_vel) - 1

        else:
            raise Exception("Unsupported suffix. ")
            self.timelength = 49



        # split data w.r.t interp and extrap, also normalize times
        if self.mode=="interp":
            raise Exception("interpolation not supported. ")
            loc_en,vel_en,times_en = self.interp_extrap(loc,vel,times,self.mode,data_type)
            loc_de = loc_en
            vel_de = vel_en
            times_de = times_en
        elif self.mode == "extrap":
            coord_en,loc_en,vel_en,times_en,coord_de,loc_de,vel_de,times_de = self.interp_extrap(coord,loc,vel,times,self.mode,data_type)
            
            loc_hiers_en = [None for _ in range(self.n_hiers)]  
            vel_hiers_en = [None for _ in range(self.n_hiers)]  
            loc_hiers_de = [None for _ in range(self.n_hiers)]  
            vel_hiers_de = [None for _ in range(self.n_hiers)]  
            
            for i in range(self.n_hiers): 
                _,loc_hiers_en[i],vel_hiers_en[i],_,_,loc_hiers_de[i],vel_hiers_de[i],_ = self.interp_extrap(coord,loc_hiers[i],vel_hiers[i],times,self.mode,data_type)
                
                

        #Encoder dataloader
        series_list_observed, coord_observed, loc_observed, vel_observed, times_observed = self.split_data(coord_en, loc_en, vel_en, times_en)
        
        loc_hiers_observed = [None for _ in range(self.n_hiers)] 
        vel_hiers_observed = [None for _ in range(self.n_hiers)] 
        for i in range(self.n_hiers): 
            _, _, loc_hiers_observed[i], vel_hiers_observed[i], _ = self.split_data(coord_en, loc_hiers_en[i], vel_hiers_en[i], times_en)
        

        if self.mode == "interp":
            raise Exception("interpolation not supported. ")
            time_begin = 0
        else:
            # time_begin = 1
            time_begin = 0
        '''
        encoder_data_loader, graph_data_loader = self.transfer_data(loc_observed, vel_observed, edges,
                                                                    times_observed, time_begin=time_begin)
        '''
        encoder_data_loader = self.transfer_data(coord_observed, loc_observed, vel_observed, loc_hiers_observed, vel_hiers_observed, edges=None, times=times_observed, time_begin=time_begin)

        ''' Radius graph should be constructed during runtime. But how to contruct the static graph?
        # Graph Dataloader --USING NRI
        edges = np.reshape(edges, [-1, self.num_atoms ** 2])
        edges = np.array((edges + 1) / 2, dtype=np.int64)
        edges = torch.LongTensor(edges)
        # Exclude self edges
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((self.num_atoms, self.num_atoms)) - np.eye(self.num_atoms)),
            [self.num_atoms, self.num_atoms])

        edges = edges[:, off_diag_idx]
        graph_data_loader = Loader(edges, batch_size=self.batch_size)

        '''


        # Decoder Dataloader
        if self.mode=="interp":
            raise Exception("interpolation not supported. ")
            series_list_de = series_list_observed
        elif self.mode == "extrap":
            # Well, we don't need to record the decoder coordinates. 
            # Instead, we use the coordinates from the last observed snapshot to contruct the radius graph. 
            series_list_de = self.decoder_data(loc_de,vel_de,loc_hiers_de,vel_hiers_de,times_de)
        decoder_data_loader = Loader(series_list_de, batch_size=self.batch_size * self.num_atoms, shuffle=False,
                                     collate_fn=lambda batch: self.variable_time_collate_fn_activity(
                                         batch))  # num_graph*num_ball [tt,vals,masks]


        num_batch = len(decoder_data_loader)
        encoder_data_loader = utils.inf_generator(encoder_data_loader)
        # graph_data_loader = utils.inf_generator(graph_data_loader)
        decoder_data_loader = utils.inf_generator(decoder_data_loader)

        # return encoder_data_loader, decoder_data_loader, graph_data_loader, num_batch
        return encoder_data_loader, decoder_data_loader, num_batch




    def interp_extrap(self,coord,loc,vel,times,mode,data_type):
        loc_observed = np.ones_like(loc)
        vel_observed = np.ones_like(vel)
        times_observed = np.ones_like(times)
        if mode =="interp":
            raise Exception("interpolation not supported. ")
            if data_type== "test":
                # get ride of the extra nodes in testing data.
                for i in range(self.num_graph):
                    for j in range(self.num_atoms):
                        loc_observed[i][j] = loc[i][j][:-self.num_pre]
                        vel_observed[i][j] = vel[i][j][:-self.num_pre]
                        times_observed[i][j] = times[i][j][:-self.num_pre]

                return loc_observed,vel_observed,times_observed/self.total_step
            else:
                return loc,vel,times/self.total_step


        elif mode == "extrap":# split into 2 parts and normalize t seperately

            assert (data_type == "test" or data_type == "train")

            assert (self.num_raw == coord.shape[2]) 

            coord_observed = coord[:,:,-self.num_entire:-self.num_pre,:]
            loc_observed = loc[:,:,-self.num_entire:-self.num_pre,:]
            vel_observed = vel[:,:,-self.num_entire:-self.num_pre,:]
            times_observed = times[:,:,-self.num_entire:-self.num_pre]

            coord_extrap = coord[:,:,-self.num_pre:,:]
            loc_extrap = loc[:,:,-self.num_pre:,:]
            vel_extrap = vel[:,:,-self.num_pre:,:]
            times_extrap = times[:,:,-self.num_pre:]

            # times_observed = times_observed/self.total_step
            # times_extrap = (times_extrap - self.total_step)/self.total_step

            times_observed = (times_observed - (self.num_raw - self.num_entire))/self.total_step
            times_extrap = (times_extrap - (self.num_raw - self.num_pre))/self.total_step


            '''
            coord_observed = np.ones_like(coord)
            loc_observed = np.ones_like(loc)
            vel_observed = np.ones_like(vel)
            times_observed = np.ones_like(times)

            coord_extrap = np.ones_like(coord)
            loc_extrap = np.ones_like(loc)
            vel_extrap = np.ones_like(vel)
            times_extrap = np.ones_like(times)

            if data_type == "test" or data_type == "train":
                for i in range(self.num_graph):
                    for j in range(self.num_atoms):
                        coord_observed[i][j] = coord[i][j][:-self.num_pre]
                        loc_observed[i][j] = loc[i][j][:-self.num_pre]
                        vel_observed[i][j] = vel[i][j][:-self.num_pre]
                        times_observed[i][j] = times[i][j][:-self.num_pre]

                        coord_extrap[i][j] = coord[i][j][-self.num_pre:]
                        loc_extrap[i][j] = loc[i][j][-self.num_pre:]
                        vel_extrap[i][j] = vel[i][j][-self.num_pre:]
                        times_extrap[i][j] = times[i][j][-self.num_pre:]
                times_observed = times_observed/self.total_step
                times_extrap = (times_extrap - self.total_step)/self.total_step
            else:
                raise Exception("Deprecate the complicated training data spliting scheme in LG-ODE. ")
                for i in range(self.num_graph):
                    for j in range(self.num_atoms):
                        times_current = times[i][j]
                        times_current_mask = np.where(times_current<self.total_step//2,times_current,0)
                        num_observe_current = np.argmax(times_current_mask)+1

                        loc_observed[i][j] = loc[i][j][:num_observe_current]
                        vel_observed[i][j] = vel[i][j][:num_observe_current]
                        times_observed[i][j] = times[i][j][:num_observe_current]

                        loc_extrap[i][j] = loc[i][j][num_observe_current:]
                        vel_extrap[i][j] = vel[i][j][num_observe_current:]
                        times_extrap[i][j] = times[i][j][num_observe_current:]

                times_observed = times_observed / self.total_step
                times_extrap = (times_extrap - self.total_step//2) / self.total_step
            
            '''

            return coord_observed,loc_observed,vel_observed,times_observed,coord_extrap,loc_extrap,vel_extrap,times_extrap


    def split_data(self,coord,loc,vel,times):
        coord_observed = np.ones_like(coord)
        loc_observed = np.ones_like(loc)
        vel_observed = np.ones_like(vel)
        times_observed = np.ones_like(times)

        # split encoder data
        coord_list = []
        loc_list = []
        vel_list = []
        times_list = []

        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                '''
                coord_list.append(coord[i][j][1:])
                loc_list.append(loc[i][j][1:])  # [2500] num_train * num_ball
                vel_list.append(vel[i][j][1:])
                times_list.append(times[i][j][1:])
                '''
                coord_list.append(coord[i][j])
                loc_list.append(loc[i][j])  # [2500] num_train * num_ball
                vel_list.append(vel[i][j])
                times_list.append(times[i][j])






        series_list = []
        odernn_list = []
        for i, loc_series in enumerate(loc_list):
            # for encoder data
            graph_index = i // self.num_atoms
            atom_index = i % self.num_atoms
            length = len(loc_series)
            '''
            preserved_idx = sorted(
                np.random.choice(np.arange(length), int(length * self.sample_percent), replace=False))
            '''
            preserved_idx = np.arange(length) # Deprecate sampling. 
            coord_observed[graph_index][atom_index] = coord_list[i][preserved_idx]
            loc_observed[graph_index][atom_index] = loc_series[preserved_idx]
            vel_observed[graph_index][atom_index] = vel_list[i][preserved_idx]
            times_observed[graph_index][atom_index] = times_list[i][preserved_idx]

            # for odernn encoder
            feature_observe = np.zeros((self.timelength, self.feature))  # [T,D]
            times_observe = -1 * np.ones(self.timelength)  # maximum #[T], padding -1
            mask_observe = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed

            times_observe[:len(times_list[i][preserved_idx])] = times_list[i][preserved_idx]
            feature_observe[:len(times_list[i][preserved_idx])] = np.concatenate(
                (loc_series[preserved_idx], vel_list[i][preserved_idx]), axis=1)
            mask_observe[:len(times_list[i][preserved_idx])] = 1

            tt_observe = torch.FloatTensor(times_observe)
            vals_observe = torch.FloatTensor(feature_observe)
            masks_observe = torch.FloatTensor(mask_observe)

            odernn_list.append((tt_observe, vals_observe, masks_observe))

            # for decoder data, padding and mask
            feature_predict = np.zeros((self.timelength, self.feature))  # [T,D]
            times_predict = -1 * np.ones(self.timelength)  # maximum #[T], padding = 0, if have initial, then padding -1
            mask_predict = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed

            times_predict[:len(times_list[i])] = times_list[i]
            feature_predict[:len(times_list[i])] = np.concatenate((loc_series, vel_list[i]), axis=1)
            mask_predict[:len(times_list[i])] = 1

            tt = torch.FloatTensor(times_predict)
            vals = torch.FloatTensor(feature_predict)
            masks = torch.FloatTensor(mask_predict)

            series_list.append((tt, vals, masks))


        return series_list, coord_observed, loc_observed, vel_observed, times_observed

    def decoder_data(self, loc, vel, loc_hiers, vel_hiers, times):

        # split decoder data
        loc_list = []
        vel_list = []
        times_list = []
        
        loc_hiers_list = [] 
        vel_hiers_list = [] 

        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                loc_list.append(loc[i][j])  # [2500] num_train * num_ball
                vel_list.append(vel[i][j])
                times_list.append(times[i][j])

        for layer in range(self.n_hiers): 
            loc_hiers_list.append([]) 
            vel_hiers_list.append([]) 
            for i in range(self.num_graph):
                for j in range(self.num_atoms):
                    loc_hiers_list[layer].append(loc_hiers[layer][i][j])  # [2500] num_train * num_ball
                    vel_hiers_list[layer].append(vel_hiers[layer][i][j])             
            

        series_list = []
        for i, loc_series in enumerate(loc_list):
            # for decoder data, padding and mask
            feature_predict = np.zeros((self.timelength, self.feature))  # [T,D]
            times_predict = -1 * np.ones(self.timelength)  # maximum #[T], padding = 0, if have initial, then padding -1
            mask_predict = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed

            feature_hier_predict = [None for _ in range(self.n_hiers)]  
            hiers_vals = [None for _ in range(self.n_hiers)]  
            for layer in range(self.n_hiers): 
                feature_hier_predict[layer] = np.zeros((self.timelength, self.feature))  # [T,D] 
                feature_hier_predict[layer][:len(times_list[i])] = np.concatenate((loc_hiers_list[layer][i], vel_hiers_list[layer][i]), axis=1)
                hiers_vals[layer] = torch.FloatTensor(feature_hier_predict[layer]) 

            times_predict[:len(times_list[i])] = times_list[i]
            feature_predict[:len(times_list[i])] = np.concatenate((loc_series, vel_list[i]), axis=1)
            mask_predict[:len(times_list[i])] = 1

            

            tt = torch.FloatTensor(times_predict)
            vals = torch.FloatTensor(feature_predict)
            masks = torch.FloatTensor(mask_predict)

            # series_list.append((tt, vals, masks))
            series_list.append((tt, vals, hiers_vals, masks))

        return series_list

    
    def custom_enc_collate(data_list):
        # Assuming each dictionary in `data_list` has the same keys
        keys = data_list[0].keys()

        # Initialize a dictionary to hold the batched data
        batched_data = {}

        for key in keys:
            # Extract the list of Data objects for the current key
            data_objects = [data[key] for data in data_list]

            # Collate the list of Data objects into a single Batch
            batched_data[key] = Batch.from_data_list(data_objects)

        return batched_data

    def transfer_data(self, coord, loc, vel, loc_hiers, vel_hiers, edges, times, time_begin=0):
        '''edges input disabled'''
        assert (edges is None)
        data_list = []
        # graph_list = []
        # edge_size_list = []

        for i in tqdm(range(self.num_graph)):
            '''
            data_per_graph, edge_data, edge_size = self.transfer_one_graph(loc[i], vel[i], edges[i], times[i],
                                                                           time_begin=time_begin)
            '''
            # data_per_graph = self.transfer_one_graph(coord[i], loc[i], vel[i], None, times[i], time_begin=time_begin)
            # data_list.append(data_per_graph)

            data_per_graph = {} 
            data_per_graph["agg"] = self.transfer_one_graph(coord[i], loc[i], vel[i], None, times[i], time_begin=time_begin) 
            for layer in range(self.n_hiers): 
                data_per_graph[f'layer_{layer}'] = self.transfer_one_graph(coord[i], loc_hiers[layer][i], vel_hiers[layer][i], None, times[i], time_begin=time_begin) 


            data_list.append(data_per_graph)
            
            # graph_list.append(edge_data)
            # edge_size_list.append(edge_size)

        # print("average number of edges per graph is %.4f" % np.mean(np.asarray(edge_size_list)))
        data_loader = DataLoader(data_list, batch_size=self.batch_size, collate_fn=self.custom_enc_collate) 
        # graph_loader = DataLoader(graph_list, batch_size=self.batch_size)

        return data_loader # , graph_loader

    
    def transfer_one_graph(self, coord, loc, vel, edge, time, time_begin=0, mask=True, forward=False):
        # Creating x : [N,D]
        # Creating edge_index
        # Creating edge_attr
        # Creating edge_type
        # Creating y: [N], value= num_steps
        # Creeating pos 【N】
        # forward: t0=0;  otherwise: t0=tN/2

        '''edge input disabled'''
        assert (edge is None)

        '''
        # compute cutting window size:
        if self.cutting_edge:
            if self.suffix == "_springs5" or self.suffix == "_charged5":
                max_gap = (self.total_step - 40*self.sample_percent) /self.total_step
            else:
                max_gap = (self.total_step - 30 * self.sample_percent) / self.total_step
        else:
            max_gap = 100


        if self.mode=="interp":
            forward= False
        else:
            forward=True
        '''

        y = np.zeros(self.num_atoms)
        x = list()
        x_coord = list()
        x_pos = list()
        node_number = 0
        node_time = dict()
        ball_nodes = dict()

        # Creating x, y, x_pos
        for i, ball in enumerate(loc):
            coord_ball = coord[i]
            loc_ball = ball
            vel_ball = vel[i]
            time_ball = time[i]

            # Creating y
            y[i] = len(time_ball)

            # Creating x and x_pos, by tranverse each ball's sequence
            for j in range(loc_ball.shape[0]):
                xj_feature = np.concatenate((loc_ball[j], vel_ball[j]))
                x.append(xj_feature)
                x_coord.append(coord_ball[j])

                x_pos.append(time_ball[j] - time_begin)
                node_time[node_number] = time_ball[j]

                if i not in ball_nodes.keys():
                    ball_nodes[i] = [node_number]
                else:
                    ball_nodes[i].append(node_number)

                node_number += 1

        '''
         matrix computing
         '''
        # Adding self-loop

        ''' Deprecate all edge-related ops
        edge_with_self_loop = edge + np.eye(self.num_atoms, dtype=int)

        edge_time_matrix = np.concatenate([np.asarray(x_pos).reshape(-1, 1) for i in range(len(x_pos))],
                                          axis=1) - np.concatenate(
            [np.asarray(x_pos).reshape(1, -1) for i in range(len(x_pos))], axis=0)
        edge_exist_matrix = np.zeros((len(x_pos), len(x_pos)))

        for i in range(self.num_atoms):
            for j in range(self.num_atoms):
                if edge_with_self_loop[i][j] == 1:
                    sender_index_start = int(np.sum(y[:i]))
                    sender_index_end = int(sender_index_start + y[i])
                    receiver_index_start = int(np.sum(y[:j]))
                    receiver_index_end = int(receiver_index_start + y[j])
                    if i == j:
                        edge_exist_matrix[sender_index_start:sender_index_end,
                        receiver_index_start:receiver_index_end] = 1
                    else:
                        edge_exist_matrix[sender_index_start:sender_index_end,
                        receiver_index_start:receiver_index_end] = -1

        if mask == None:
            edge_time_matrix = np.where(abs(edge_time_matrix)<=max_gap,edge_time_matrix,-2)
            edge_matrix = (edge_time_matrix + 2) * abs(edge_exist_matrix)  # padding 2 to avoid equal time been seen as not exists.
        elif forward == True:  # sender nodes are thosewhose time is larger. t0 = 0
            edge_time_matrix = np.where((edge_time_matrix >= 0) & (abs(edge_time_matrix)<=max_gap), edge_time_matrix, -2) + 2
            edge_matrix = edge_time_matrix * abs(edge_exist_matrix)
        elif forward == False:  # sender nodes are thosewhose time is smaller. t0 = tN/2
            edge_time_matrix = np.where((edge_time_matrix <= 0) & (abs(edge_time_matrix)<=max_gap), edge_time_matrix, -2) + 2
            edge_matrix = edge_time_matrix * abs(edge_exist_matrix)

        _, edge_attr_same = self.convert_sparse(edge_exist_matrix * edge_matrix)
        edge_is_same = np.where(edge_attr_same > 0, 1, 0).tolist()

        edge_index, edge_attr = self.convert_sparse(edge_matrix)
        edge_attr = edge_attr - 2
        edge_index_original, _ = self.convert_sparse(edge)

        '''



        # converting to tensor
        x = torch.FloatTensor(x)
        # edge_index = torch.LongTensor(edge_index)
        # edge_attr = torch.FloatTensor(edge_attr)
        # edge_is_same = torch.FloatTensor(np.asarray(edge_is_same))

        x_coord = torch.FloatTensor(x_coord)

        y = torch.LongTensor(y)
        x_pos = torch.FloatTensor(x_pos)

        # graph_index_original = torch.LongTensor(edge_index_original)
        # edge_data = Data(x = torch.ones(self.num_atoms),edge_index = graph_index_original)


        # graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=x_pos, edge_same=edge_is_same)
        graph_data = Data(x=x, y=y, pos=x_pos, x_coord=x_coord)
        # edge_size = edge_index.shape[1]

        return graph_data # ,edge_data,edge_size

    def variable_time_collate_fn_activity(self,batch):
        """
        Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
            - record_id is a patient id
            - tt is a 1-dimensional tensor containing T time values of observations.
            - vals is a (T, D) tensor containing observed values for D variables.
            - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise. Since in human dataset, it join the data of four tags (belt, chest, ankles) into a single time series
            - labels is a list of labels for the current patient, if labels are available. Otherwise None.
        Returns:
            combined_tt: The union of all time observations.
            combined_vals: (M, T, D) tensor containing the observed values.
            combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
        """
        D = self.feature
        combined_tt, inverse_indices = torch.unique(torch.cat([ex[0] for ex in batch]), sorted=True,
                                                    return_inverse=True) #【including 0 ]
        offset = 0
        combined_vals = torch.zeros([len(batch), len(combined_tt), D])
        combined_mask = torch.zeros([len(batch), len(combined_tt), D])
        
        combined_hiers_vals = dict() 
        for layer in range(self.n_hiers): 
            combined_hiers_vals[f'layer_{layer}'] = torch.zeros([len(batch), len(combined_tt), D]) 
                    

        for b, ( tt, vals, hiers_vals, mask) in enumerate(batch):

            indices = inverse_indices[offset:offset + len(tt)]

            offset += len(tt)

            combined_vals[b, indices] = vals
            combined_mask[b, indices] = mask 

            for layer in range(self.n_hiers): 
                combined_hiers_vals[f'layer_{layer}'][b, indices] = hiers_vals[layer] 
            
            

        # ????????? 
        # print("please verify!!!!!!!!!!!!!!") 
        for layer in range(self.n_hiers): 
            combined_hiers_vals[f'layer_{layer}'] = combined_hiers_vals[f'layer_{layer}'][:,1:,:]
                

        # get rid of the padding timepoint
        combined_tt = combined_tt[1:]
        combined_vals = combined_vals[:,1:,:]
        combined_mask = combined_mask[:,1:,:]

        combined_tt = combined_tt.float()


        data_dict = {
            "data": combined_vals,
            "hiers_data": combined_hiers_vals, 
            "time_steps": combined_tt,
            "mask": combined_mask,
            }
        return data_dict

    def normalize_features(self,inputs, num_balls):
        '''

        :param inputs: [num-train, num-ball,(timestamps,2)]
        :return:
        '''
        value_list_length = [balls[i].shape[0] for i in range(num_balls) for balls in inputs]  # [2500] num_train * num_ball
        self.timelength = max(value_list_length)
        value_list = [torch.tensor(balls[i]) for i in range(num_balls) for balls in inputs]
        value_padding = pad_sequence(value_list,batch_first=True,padding_value = 0)
        max_value = torch.max(value_padding).item()
        min_value = torch.min(value_padding).item()

        # Normalize to [-1, 1]
        inputs = (inputs - min_value) * 2 / (max_value - min_value) - 1
        return inputs,max_value,min_value

    def convert_sparse(self,graph):
        graph_sparse = sp.coo_matrix(graph)
        edge_index = np.vstack((graph_sparse.row, graph_sparse.col))
        edge_attr = graph_sparse.data
        return edge_index, edge_attr





