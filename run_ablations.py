import os
import sys
# from lib.new_dataLoader import ParseData 
from lib.hier_dataLoader import ParseHierData 
from tqdm import tqdm
import argparse
import numpy as np
from random import SystemRandom
import torch
import torch.optim as optim
import lib.utils as utils
from torch.distributions.normal import Normal
from lib.create_latent_ode_model import create_LatentODE_model
from lib.utils import compute_loss_all_batches

from torch_geometric.data import Batch, Data 
import torch.nn as nn 


import sys
import datetime
import argparse
from contextlib import contextmanager

import json 


# import multiprocessing

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
# parser.add_argument('--n-balls', type=int, default=5,
#                     help='Number of objects in the dataset.')
parser.add_argument('--niters', type=int, default=50)
parser.add_argument('--lr',  type=float, default=5e-4, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=16)
parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
# parser.add_argument('--save-graph', type=str, default='plot/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="name of ckpt. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")
# more protein datasets to be added. 
parser.add_argument('--data', type=str, default='ala2', help="protein datasets, including ala2, etc. (only consider the alpha carbons)")
# parser.add_argument('--z0-encoder', type=str, default='GTrans', help="GTrans")
parser.add_argument('--z0-encoder', type=str, default='ParticleNet', help="ParticleNet") 
parser.add_argument('-l', '--latents', type=int, default=16, help="Size of the latent state")
# parser.add_argument('--rec-dims', type=int, default= 64, help="Dimensionality of the recognition model .")
# parser.add_argument('--ode-dims', type=int, default=128, help="Dimensionality of the ODE func")
parser.add_argument('--rec-layers', type=int, default=2, help="Number of layers in recognition model ")
# parser.add_argument('--n-heads', type=int, default=1, help="Number of heads in GTrans")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers  ODE func ")
# parser.add_argument('--extrap', type=str,default="True", help="Set extrapolation mode. If this flag is not set, run interpolation mode.")

parser.add_argument('--dropout', type=float, default=0.2,help='Dropout rate (1 - keep probability).')
# parser.add_argument('--sample-percent-train', type=float, default=0.6,help='Percentage of training observtaion data')
# parser.add_argument('--sample-percent-test', type=float, default=0.6,help='Percentage of testing observtaion data')
parser.add_argument('--augment_dim', type=int, default=64, help='augmented dimension')
parser.add_argument('--edge_types', type=int, default=2, help='edge number in NRI')
parser.add_argument('--odenet', type=str, default="NRI", help='NRI')
parser.add_argument('--solver', type=str, default="rk4", help='dopri5,rk4,euler')
parser.add_argument('--l2', type=float, default=1e-3, help='l2 regulazer')
parser.add_argument('--optimizer', type=str, default="AdamW", help='Adam, AdamW')
parser.add_argument('--clip', type=float, default=10, help='Gradient Norm Clipping')
# parser.add_argument('--cutting_edge', type=bool, default=True, help='True/False')

# parser.add_argument('--extrap_num', type=int, default=8000, help='extrap num ')
# parser.add_argument('--total_ode_step', type=int, default=8000, help='total ode step ') # ? 
# parser.add_argument('--entire_num', type=int, default=10000, help='entire num ')

parser.add_argument('--observe_scope', type=int, default=2000, help='the scope of the observed part in the raw data. ') 
parser.add_argument('--extrap_scope', type=int, default=8000, help='the scope of the part to be extrapolated in the raw data. ') 
parser.add_argument('--raw_num', type=int, default=250000, help='the number of data points in the raw data. ')

parser.add_argument('--rec_attention', type=str, default="attention")
parser.add_argument('--alias', type=str, default="run")
# parser.add_argument('--use_gpu', type=bool, default=False)
# parser.add_argument('--use_gpu', action='store_true')
# parser.add_argument('--full_dataset', action='store_true')
parser.add_argument('--no_gpu', action='store_true')
# parser.add_argument('--no_multi_device', action='store_true')

# parser.add_argument('--partial_dataset', action='store_true')
# parser.add_argument('--teacher_forcing', action='store_true')
# parser.add_argument('--supervise_lookahead', action='store_true')
parser.add_argument('--use_vae', action='store_true')

parser.add_argument('--num_levels', type=int, default=3, help="Num of level of hierarchies to split the data ") 
# parser.add_argument('--filter_sizes', type=list, default=[10,5], help="Filter sizes when doing moving average. ")
# parser.add_argument('--filter_sizes', type=int, action='append',
#                     help='add an integer to the list of Filter sizes when doing moving average.')

parser.add_argument("--ode_forecast_frequencies", type=int, nargs='+', default=[400, 20, 1], 
                    help="Specifies the frequency at which each level of the ODE function forecasts unobserved time points.")
parser.add_argument("--encoder_sampling_dilations", type=float, nargs='+', default=[0.5, 0.5, 1.0], 
                    help="Sets the sampling frequency dilation for each level of the encoder.")
parser.add_argument("--encoder_lookahead_scopes", type=float, nargs='+', default=[0.25, 1.5, 2.0], 
                    help="Defines the relative look-ahead scope for each encoder level.")
parser.add_argument('--max_segments', type=int, default=5, help="Maximum number of segments to train on each level. ") 

parser.add_argument("--level_weights", type=float, nargs='+', default=[0.1, 0.2, 0.7], 
                    help="Specifies the weights for each level in the final loss term.")
parser.add_argument('--log_feature', type=str, default="trial", help='Feature name for the log file') 

parser.add_argument('--proj_aggregate', type=str, default="concat", help='Projector aggregation schemes: concat, add, single, zero. ') 
parser.add_argument('--dec_aggregate', type=str, default="concat", help='Decoder aggregation schemes: concat, add, single. ') 


# fprint("What is the required relationship between `extrap_num`, `total_ode_step`, and `entire_num`? ")

# For ala2 dataset 
# do not specify n_features explicitly, make it compliant with the rest of the dataset 
# parser.add_argument('--n_features', type=int, default=64, help="Number of features for the model. ") 
parser.add_argument('--max_lag', type=int, default=1000, help="Maximum lag to consider in the ALA2 dataset. ") 
parser.add_argument('--diff_steps', type=int, default=1000, help="Number of diffusion steps in the model.") 
parser.add_argument("--indistinguishable", action='store_true', help="Enable this flag to treat atoms as indistinguishable.")
parser.add_argument("--iterative_training", action='store_true', help="Enable this flag to perform iterative training.") 


args = parser.parse_args()
# assert(int(args.rec_dims%args.n_heads) ==0)

# assert(args.n_hiers == len(args.filter_sizes) + 1) 

assert(args.num_levels == len(args.ode_forecast_frequencies)) 
assert(args.num_levels == len(args.encoder_sampling_dilations)) 
assert(args.num_levels == len(args.encoder_lookahead_scopes)) 
assert(int(args.extrap_scope * args.encoder_lookahead_scopes[0]) == args.observe_scope) 

@contextmanager
def log_to_file(filename):
    # Save the original stdout
    original_stdout = sys.stdout
    try:
        # Open the log file and set stdout to it
        with open(filename, 'a') as file:
            sys.stdout = file
            yield
    finally:
        # Reset stdout to its original state
        sys.stdout = original_stdout

# Construct the log file name
current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%m%d%H%M')
log_filename = f"{formatted_time}_{args.log_feature}.log"
ckpt_filename = f"{formatted_time}_{args.log_feature}.pt" 

def fprint(*msg): 
    with log_to_file(log_filename): 
        print(*msg, flush=True)  


fprint("Arguments:", args)

for i in range(args.num_levels): 
    fprint(f'Level {i}: ') 
    fprint(f"ODE sampling frequency: once every {args.ode_forecast_frequencies[i]} timepoints")
    fprint(f"Encoder sampling frequency: once every {int(args.ode_forecast_frequencies[i] * args.encoder_sampling_dilations[i])} timepoints")
    ext_scope = args.extrap_scope if (i == 0) else args.ode_forecast_frequencies[i - 1] 
    enc_scope = int(ext_scope * args.encoder_lookahead_scopes[i]) 
    fprint(f"ODE prediction scope: {ext_scope}")
    fprint(f"Encoder observation scope: {enc_scope}")

assert (sum(args.level_weights) == 1.0) 

'''
Q: what does time step in ode solving mean exactly? Can you provide a simple example?

A: In solving ordinary differential equations (ODEs), a "time step" refers to the increment in the independent variable (often time) used in numerical methods. The concept of a time step is used in iterative methods, where the solution is progressed step by step.

The size of the time step can affect the accuracy and stability of the numerical solution. A smaller time step usually leads to a more accurate solution, but it requires more computational resources. Conversely, a larger time step can speed up the computation, but it may lead to less accurate or unstable solutions.

Here's a simple example using the Euler method to solve the ODE dy/dt = y, with the initial condition y(0) = 1. This ODE has an exact solution y(t) = e^t.

1. Choose a time step size, for example, dt = 0.1.
2. Start at the initial condition, t = 0, y = 1.
3. Use the Euler method to find the next value: y_new = y + dt * dy/dt = y + dt * y.
4. Update t and y: t = t + dt, y = y_new.
5. Repeat steps 3 and 4 until you reach the desired final time.

This method will give an approximate solution to the ODE. If you compare it to the exact solution, you will see that the approximation improves as you decrease the time step size.
'''

if args.data == "lj": 
    # raise Exception("LJ data is not supported. ") 
    args.dataset = f'data/larger_lj_data_{args.raw_num}' 
    args.suffix = '_lj'
    # args.total_ode_step = 10 
    args.n_balls = 258
    args.cutoff = 7.5 
elif args.data == "tip3p":
    # raise Exception("TIP3P data is not supported. ") 
    args.dataset = f'data/larger_tip3p_data_{args.raw_num}' 
    args.suffix = '_tip3p'
    args.n_balls = 258 * 3 # TODO: fill in the number of balls 
    args.cutoff = 4.2 
elif args.data == "tip4p": 
    # raise Exception("TIP4P data is not supported. ") 
    args.dataset = f'data/larger_tip4p_data_{args.raw_num}' 
    args.suffix = '_tip4p'
    args.n_balls = 251 * 3 # TODO: fill in the number of balls 
    args.cutoff = 4.2 
elif args.data == "ala2": 
    args.dataset = f'data/ala2_data_{args.raw_num}' 
    args.suffix = '_ala2'
    args.n_balls = 22  
    args.cutoff = None 
else: 
    raise Exception("Unsupported dataset. ")




############ CPU AND GPU related, Mode related, Dataset Related
if torch.cuda.is_available() and (args.no_gpu is False):
    fprint("Using GPU" + "-"*80)
    device = torch.device("cuda:0")
    fprint("Number of available cuda devices:", torch.cuda.device_count()) 
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "" 
    fprint("Using CPU" + "-" * 80)
    device = torch.device("cpu")


fprint("Running in extraploation mode. ") 


#####################################################################################################

if __name__ == '__main__':
    # torch.manual_seed(args.random_seed)
    # np.random.seed(args.random_seed)

    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random() * 100000)
        fprint(f"Creating new experiment ID: {experimentID}") 


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
    
    set_seed(args.random_seed)
    seed_torch(args.random_seed)
    
    fprint("Fix all seed to: ", args.random_seed)
    fprint("Why there is still randomness even if I set the seed? ") 



    ############ Saving Path and Preload.
    file_name = os.path.basename(__file__)[:-3]  # run_models
    utils.makedirs(args.save)
    # utils.makedirs(args.save_graph)


    ############ Loading Data
    fprint("Loading dataset: " + args.dataset)

    dataloader = ParseHierData(args.dataset, args, args.suffix)

    train_combined_loader, train_batch = dataloader.load_data(batch_size=args.batch_size,  
                                                              data_type="train", device=device, fprint=fprint)
    test_combined_loader, test_batch = dataloader.load_data(batch_size=args.batch_size,    
                                                            data_type="test", device=device, fprint=fprint) 



    input_dim = dataloader.feature

    ############ Command Related
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)

    ############ Model Select
    # Create the model
    obsrv_std = 0.01
    obsrv_std = torch.Tensor([obsrv_std]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
    fprint("`z0_prior` is used to calculate the KL divergence \
          between the prior and the posterior. It is probably \
            not trainable, so we can probably share it for now.") 

    model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, args.use_vae)
    
    '''
    if (args.no_multi_device is False) and (torch.cuda.device_count() > 1):
        fprint("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        model.to(device)
    ''' 


    ##################################################################
    # Load checkpoint and evaluate the model
    if args.load is not None:
        fprint("Loading model from: ", args.load) 
        ckpt_path = args.load 
        utils.get_ckpt_model(ckpt_path, model, device)
        #exit()

    ##################################################################
    # Training

    ''' 
    log_path = "logs/" + args.alias +"_" + args.z0_encoder+ "_" + args.data + "_" +str(args.sample_percent_train)+ "_" + args.mode + "_" + str(experimentID) + ".log"
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info(str(args))
    logger.info(args.alias)
    ''' 

    # Optimizer
    fprint("Parameters registered to the optimizer: (beware of unregistred parameters!!!!!)")
    '''
    for name, param in model.named_parameters():
        fprint(f"{name}: shape={param.shape}, number of elements={param.numel()}")
    ''' 
        
    if args.optimizer == "AdamW":
        optimizer =optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.l2)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)


    wait_until_kl_inc = 10
    best_test_mse = np.inf
    n_iters_to_viz = 1

    best_mse = float('inf')  # Initialize with a very high value
    best_epoch = 0


    def train_single_batch(model,batch_dict_encoder,batch_dict_decoder,batch_dict_graph,kl_coef):

        raise Exception("This function is not used. ") 

        assert (batch_dict_graph is None)
        
        optimizer.zero_grad()
        # train_res = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, batch_dict_graph,
        #                                      n_traj_samples=3, kl_coef=kl_coef)
        
        train_res = {}
        pred_y = {}

        # train_res['aggr'] = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, None, layer_name='aggr', n_traj_samples=3, kl_coef=kl_coef)

        '''
        def train_on_gpu(model, batch_dict_encoder, batch_dict_decoder, kl_coef, gpu_id, layer, results_dict):
            # Move the model to the specified GPU
            model.to(f'cuda:{gpu_id}')

            # Perform computation
            pred_y, train_res = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, None, layer_name=f'layer_{layer}', n_traj_samples=3, kl_coef=kl_coef)

            # Store results in the shared dictionary
            results_dict[f'layer_{layer}'] = (pred_y, train_res)     

        # Dictionary to store results from each process
        manager = multiprocessing.Manager()
        results_dict = manager.dict()

        # List to keep track of processes
        processes = []

        for layer in range(args.n_hiers):
            # Assign each layer to a different GPU
            gpu_id = (layer + 1) % torch.cuda.device_count()  # This assumes args.n_hiers <= number of GPUs

            # Create and start a new process for each layer
            p = multiprocessing.Process(target=train_on_gpu, args=(model, batch_dict_encoder, batch_dict_decoder, kl_coef, gpu_id, layer, results_dict))
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()   

        for key, value in results_dict.items(): 
            fprint(key) 
            train_res[key] = value[1]
            pred_y[key] = value[0] 
        '''

        
        for layer in range(args.n_hiers): 
            pred_y[f'layer_{layer}'], train_res[f'layer_{layer}'] = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, None, layer_name=f'layer_{layer}', n_traj_samples=3, kl_coef=kl_coef)
        
        
        # loss = train_res["loss"] 
        pred_y_original = torch.zeros_like(pred_y[f'layer_{0}']) 
        for layer in range(args.n_hiers): 
            pred_y_original = pred_y_original + pred_y[f'layer_{layer}'] 
        pred_y_original += dataloader.correction_tsr.view((1,1,1,-1)).to(device) 
        mse_original = model.get_mse(batch_dict_decoder["data"], pred_y_original, mask=batch_dict_decoder["mask"])  # [1]
        mae_original = model.get_mae(batch_dict_decoder["data"], pred_y_original, mask=batch_dict_decoder["mask"])  # [1]
        mape_original = model.get_mape(batch_dict_decoder["data"], pred_y_original, mask=batch_dict_decoder["mask"])  # [1]
            
        loss = 0 
        for layer in range(args.n_hiers): 
            loss += train_res[f'layer_{layer}']['loss'] # ?????? 
        loss = loss / args.n_hiers 

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        loss_value = loss.data.item()

        mse_original_value = mse_original.data.item() 
        mae_original_value = mae_original.data.item() 
        mape_original_value = mape_original.data.item() 

        del loss; del mse_original 
        torch.cuda.empty_cache()
        # train_res, loss
        # return loss_value,train_res["mse"],train_res["likelihood"],train_res["kl_first_p"],train_res["std_first_p"]
        return loss_value, mse_original_value, mae_original_value, mape_original_value, train_res 

    def get_weights_for_epoch(epoch, schedules):
        """ Determine the weight configuration for a given epoch. """
        for schedule in schedules:
            start = schedule['epochs']['start']
            end = schedule['epochs']['end']
            if epoch >= start and ((end == "remaining") or (epoch < end)):
                return schedule['weights']
        return None  # Default case, if no schedule matches


    def train_epoch(epo, mode="ablation", designated_level=None, select=None): 

        global best_mse 
        global best_epoch 

        '''
        if mode == "train":
            model.train()
            combined_loader = train_combined_loader
            combined_loader.set_select(select) 
        elif mode == "test": 
            model.eval() 
            combined_loader = test_combined_loader 
            combined_loader.set_select(select) 
        elif mode == "select": 
            model.eval() 
            combined_loader = test_combined_loader 
            combined_loader.set_select(select) 
        ''' 
        # ablation study mode 
        assert mode == "ablation" 
        assert select is None 
        model.eval() 
        combined_loader = test_combined_loader 
        combined_loader.set_select(select) 
        
        loss_list = []

        loss_hier_list = [[] for _ in range(args.num_levels)] 
        mse_hier_list = [[] for _ in range(args.num_levels)] 
        mae_hier_list = [[] for _ in range(args.num_levels)] 
        mape_hier_list = [[] for _ in range(args.num_levels)] 
        likelihood_hier_list = [[] for _ in range(args.num_levels)] 
        kl_first_p_hier_list = [[] for _ in range(args.num_levels)] 
        std_first_p_hier_list = [[] for _ in range(args.num_levels)] 

        lookahead_mse_hier_list = [[] for _ in range(args.num_levels - 1)] 
        lookahead_mae_hier_list = [[] for _ in range(args.num_levels - 1)] 
        lookahead_mape_hier_list = [[] for _ in range(args.num_levels - 1)] 
        lookahead_likelihood_hier_list = [[] for _ in range(args.num_levels - 1)] 
        lookahead_kl_first_p_hier_list = [[] for _ in range(args.num_levels - 1)] 
        lookahead_std_first_p_hier_list = [[] for _ in range(args.num_levels - 1)] 

        


        torch.cuda.empty_cache()



        for itr, raw_batch in enumerate(tqdm(combined_loader)): 

            # Load raw data 
            raw_loc = raw_batch["loc"]
            # raw_vel = raw_batch["vel"]
            raw_coord = raw_batch["coord"] 
            # raw_times = raw_batch["times"] 

            if (args.data in ["lj", "tip3p", "tip4p"]): 
                raw_vel = raw_batch["vel"] 
                # raw_times = raw_batch["times"] 

            # number of sequences in the data 
            num_graphs = raw_loc.shape[0] 

            '''
            # should be decoded from `cum_latent` 
            pred_loc = {f'level_{i}': torch.zeros_like(raw_loc, device=device) for i in range(args.num_levels)} 
            pred_vel = {f'level_{i}': torch.zeros_like(raw_vel, device=device) for i in range(args.num_levels)} 
            pred_coord = {f'level_{i}': torch.zeros_like(raw_coord, device=device) for i in range(args.num_levels)}
            ''' 

            ''' accumulation will no longer be carried out on the physical space 
            cum_loc = {f'level_{i}': torch.zeros_like(raw_loc, device=device) for i in range(args.num_levels)} 
            cum_vel = {f'level_{i}': torch.zeros_like(raw_vel, device=device) for i in range(args.num_levels)} 
            cum_coord = {f'level_{i}': torch.zeros_like(raw_coord, device=device) for i in range(args.num_levels)}
            ''' 

            # print(args.latents) 
            # print(raw_loc.shape) 
            # create zero tensor, the first 3 dimensions are the same as `raw_loc`, and the last dimension is `args.latents` 
            pred_latent = {f'level_{i}': torch.zeros((raw_loc.shape[0], raw_loc.shape[1], raw_loc.shape[2], args.latents), device=device) for i in range(args.num_levels)} 
            cum_latent = {f'level_{i}': torch.zeros((raw_loc.shape[0], raw_loc.shape[1], raw_loc.shape[2], args.latents*(i+1)), device=device) for i in range(args.num_levels)} 
            
            add_latent = {f'level_{i}': torch.zeros((raw_loc.shape[0], raw_loc.shape[1], raw_loc.shape[2], args.latents), device=device) for i in range(args.num_levels)} 


            # KL Coeff. initialization 
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else: # wonder this condition will be triggered in our dataset 
                kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))

            info = {} 
            temporal_weights = {} 
            for level in range(args.num_levels): 
                # Load masks marking the time-points we are interested in 
                encoder_mask = raw_batch[f"level_{level}"]["encoder_mask"] 
                # decoder_mask = raw_batch[f"level_{level}"]["decoder_mask"] 
                encoder_decoder_segments = raw_batch[f"level_{level}"]["encoder_decoder_segments"] 

                union_decoder_mask = [None for _ in range(num_graphs)] 
                for i in range(num_graphs): 
                    union_decoder_mask[i] = torch.zeros_like(raw_batch[f"level_{level}"]["decoder_mask"][i], dtype=bool, device=device) 
                    for j in range(level, args.num_levels): 
                        union_decoder_mask[i] |= raw_batch[f"level_{j}"]["decoder_mask"][i] 
                        if (j == level): continue 
                        union_decoder_mask[i] |= raw_batch[f"level_{j}"]["encoder_mask"][i]



                latent_enc = [] 
                graph_latent_enc = [] 
                loc_enc, coord_enc, times_enc, times_dec = [], [], [], [] 
                if (args.data in ["lj", "tip3p", "tip4p"]):
                    vel_enc = [] 
                enc_scales, dec_scales = [], [] 

                if (level ==  0): 
                    for i in range(num_graphs): 
                        encoder_start, encoder_end, decoder_end = 0, dataloader.observe_scope, dataloader.entire_scope 
                        loc_enc.append(raw_loc[i, :, encoder_mask[i], :]) 
                        if (args.data in ["lj", "tip3p", "tip4p"]): 
                            vel_enc.append(raw_vel[i, :, encoder_mask[i], :]) 
                        coord_enc.append(raw_coord[i, :, encoder_mask[i], :])
                        times_enc.append(torch.arange(encoder_mask[i].shape[-1], device=device)[encoder_mask[i]]) 

                        times_dec.append(torch.arange(union_decoder_mask[i].shape[-1], device=device)[union_decoder_mask[i]])

                        # scaling information for encoder and decoder data: part 1 
                        assert(times_enc[-1][0] == encoder_start)
                        assert(times_enc[-1][-1] < encoder_end) 
                        if (times_dec[-1][0] == encoder_end): 
                            pass 
                        elif (times_dec[-1][0] < encoder_end):
                            print("predict back in time") 
                        else: 
                            assert False 
                        assert(times_dec[-1][-1] < decoder_end) 
                        enc_scales.append((float(encoder_start), float(encoder_end)))  
                        dec_scales.append((float(encoder_end), float(decoder_end))) 

                elif (level > 0): # load data from previous predictions 
                    for i in range(num_graphs): 
                        for segment in encoder_decoder_segments[i]: 
                            # encoder_start, encoder_end, decoder_end = segment 
                            _, encoder_end, decoder_end = segment; assert (_ is None) 
                            '''
                            enc_position_mask = torch.zeros_like(encoder_mask[i], dtype=bool, device=device) 
                            enc_position_mask[encoder_start:encoder_end] = True 
                            enc_position_mask &= encoder_mask[i] 
                            ''' 
                            # no lookahead now, just the intial state itself 
                            enc_position_mask = torch.zeros_like(encoder_mask[i], dtype=bool, device=device) 
                            enc_position_mask[encoder_end] = True 
                            assert (encoder_mask[i][encoder_end] == True) 

                            dec_position_mask = torch.zeros_like(union_decoder_mask[i], dtype=bool, device=device) 
                            dec_position_mask[encoder_end:decoder_end] = True 
                            dec_position_mask &= union_decoder_mask[i] 
                            
                            '''
                            if (mode == "train") and (args.teacher_forcing is True): 
                                loc_enc.append(raw_loc[i, :, enc_position_mask, :]) 
                                vel_enc.append(raw_vel[i, :, enc_position_mask, :]) 
                                coord_enc.append(raw_coord[i, :, enc_position_mask, :]) 
                            else: 
                                loc_enc.append(cum_loc[f'level_{level-1}'][i, :, enc_position_mask, :]) 
                                vel_enc.append(cum_vel[f'level_{level-1}'][i, :, enc_position_mask, :]) 
                                coord_enc.append(cum_coord[f'level_{level-1}'][i, :, enc_position_mask, :]) 
                            ''' 

                            if (args.proj_aggregate == "concat"): 
                                latent_enc.append(cum_latent[f'level_{level-1}'][i, :, enc_position_mask, :])  
                            elif (args.proj_aggregate == "add"): # TODO: check alignment 
                                latent_enc.append(add_latent[f'level_{level-1}'][i, :, enc_position_mask, :]) 
                            elif (args.proj_aggregate == "single"): 
                                latent_enc.append(pred_latent[f'level_{level-1}'][i, :, enc_position_mask, :]) 
                            elif (args.proj_aggregate == "zero"): 
                                latent_enc.append(torch.zeros_like(pred_latent[f'level_{level-1}'][i, :, enc_position_mask, :], device=device)) 
                            else: 
                                raise Exception("Unsupported projector aggregation scheme. ") 
                            
                            if (args.dec_aggregate == "concat"): 
                                graph_latent_enc.append(cum_latent[f'level_{level-1}'][i, :, enc_position_mask, :])  
                            elif (args.dec_aggregate == "add"): # TODO: check alignment 
                                graph_latent_enc.append(add_latent[f'level_{level-1}'][i, :, enc_position_mask, :]) 
                            elif (args.dec_aggregate == "single"): 
                                graph_latent_enc.append(pred_latent[f'level_{level-1}'][i, :, enc_position_mask, :]) 
                            else:
                                raise Exception("Unsupported decoder aggregation scheme. ")     
                                                   
                            times_enc.append(torch.arange(encoder_mask[i].shape[-1], device=device)[enc_position_mask])
                            times_dec.append(torch.arange(union_decoder_mask[i].shape[-1], device=device)[dec_position_mask])


                            # scaling information for encoder and decoder data: part 2 
                            '''
                            assert(times_enc[-1][0] == encoder_start)
                            assert(times_enc[-1][-1] < encoder_end) 
                            if (times_dec[-1][0] == encoder_end): 
                                pass 
                            elif (times_dec[-1][0] < encoder_end):
                                print("predict back in time") 
                            else: 
                                assert False 
                            ''' 

                            assert (times_enc[-1][0] == encoder_end) 

                            assert(times_dec[-1][-1] < decoder_end) 
                            # enc_scales.append((float(encoder_start), float(encoder_end)))  
                            enc_scales.append((None, float(encoder_end))) 
                            dec_scales.append((float(encoder_end), float(decoder_end)))         
                            

                        '''
                        acc_loc_enc = raw_loc[i, :, encoder_mask[i], :] 
                        acc_vel_enc = raw_vel[i, :, encoder_mask[i], :] 
                        acc_coord_enc = raw_coord[i, :, encoder_mask[i], :] 
                        acc_times_enc = torch.arange(encoder_mask[i].shape[-1], device=device)[encoder_mask[i]] 
                        acc_times_dec = torch.arange(union_decoder_mask[i].shape[-1], device=device)[union_decoder_mask[i]] 

                        enc_idx, dec_idx = 0, 0 
                        for segment in encoder_decoder_segments[i]: 
                            encoder_start, encoder_end, decoder_end = segment 
                            loc_enc.append(acc_loc_enc[enc_idx:enc_idx+encoder_end-encoder_start])  
                            vel_enc.append(acc_vel_enc[enc_idx:enc_idx+encoder_end-encoder_start]) 
                            coord_enc.append(acc_coord_enc[enc_idx:enc_idx+encoder_end-encoder_start]) 
                            times_enc.append(acc_times_enc[enc_idx:enc_idx+encoder_end-encoder_start]) 
                            times_dec.append(acc_times_dec[dec_idx:dec_idx+decoder_end-encoder_end]) 
                            enc_idx += encoder_end - encoder_start 
                            dec_idx += decoder_end - encoder_end 
                        ''' 

                
                encoder_batch = []
                # batch_actual_length = len(loc_enc) 
                batch_actual_length = len(times_enc)  
                if (level == 0): 
                    assert (batch_actual_length == num_graphs)
                    assert (batch_actual_length == len(loc_enc)) 
                    for i in range(batch_actual_length):  
                        if (args.data in ["lj", "tip3p", "tip4p"]):
                            graph_data = train_combined_loader.transfer_one_graph(data=args.data, coord=coord_enc[i], loc=loc_enc[i], vel=vel_enc[i], time=times_enc[i], time_begin=0, enc_scale=enc_scales[i])  
                        else: 
                            graph_data = train_combined_loader.transfer_one_graph(data=args.data, coord=coord_enc[i], loc=loc_enc[i], vel=None, time=times_enc[i], time_begin=0, enc_scale=enc_scales[i])  
                        encoder_batch.append(graph_data) 
                elif (level > 0): 
                    if mode is not "select": 
                        assert (batch_actual_length == num_graphs * args.max_segments)
                    assert (batch_actual_length == len(latent_enc)) 
                    assert (batch_actual_length == len(graph_latent_enc)) 
                    for i in range(batch_actual_length): 
                        graph_data = train_combined_loader.transfer_one_latent(latent=latent_enc[i], graph_latent=graph_latent_enc[i]) # TODO: implement transfer_one_latent 
                        encoder_batch.append(graph_data)   
                encoder_batch = Batch.from_data_list(encoder_batch).to(device) 

                # decoder_batch = {"time_steps": torch.stack(times_dec, dim=0)}
                '''
                Caution! Time normalization subject to change 
                '''
                # decoder_batch = {"time_steps": [x/dataloader.entire_scope for x in times_dec]}
                decoder_batch = {"time_steps": []} 
                for dec, scale in zip(times_dec, dec_scales): 
                    decoder_batch["time_steps"].append((dec - scale[0]) / (scale[1] - scale[0])) 

                '''
                if (args.no_multi_device is False) and (torch.cuda.device_count() > 1): 
                    pred_y, info[f'level_{level}'], temporal_weights[f'level_{level}'] = model.module.get_reconstruction(encoder_batch, decoder_batch, f'level_{level}', n_traj_samples=1)
                else: 
                    pred_y, info[f'level_{level}'], temporal_weights[f'level_{level}'] = model.get_reconstruction(encoder_batch, decoder_batch, f'level_{level}', n_traj_samples=1)
                '''
                pred_x_latent, info[f'level_{level}'], temporal_weights[f'level_{level}'] = model.get_reconstruction(encoder_batch, decoder_batch, f'level_{level}', n_traj_samples=1, 
                                                                                                                     max_loc=dataloader.max_loc, min_loc=dataloader.min_loc) 

                
                # to be corrected 
                # times_dec_reg = decoder_batch["time_steps"].view(batch_actual_length, -1) 
                # times_dec_reg = torch.stack(times_dec, dim=0) 

                # pred_y += dataloader.correction_tsr.view((1,1,1,-1)).to(device) # ??? 
                for l in range(batch_actual_length): 
                    times_dec_reg = times_dec[l] 
                    ''' we are not doing the aggregation on the physical space anymore 
                    pred_y_reg = pred_y[l] 
                    idx = int(l / batch_actual_length * num_graphs) 
                    pred_loc[f'level_{level}'][idx, :, times_dec_reg, :] = pred_y_reg[..., :3]
                    pred_vel[f'level_{level}'][idx, :, times_dec_reg, :] = pred_y_reg[..., 3:6]
                    pred_coord[f'level_{level}'][idx, :, times_dec_reg, :] = pred_y_reg[..., :3] * (dataloader.max_loc - dataloader.min_loc) / 2 + (dataloader.max_loc + dataloader.min_loc) / 2 

                    cum_loc[f'level_{level}'][idx, :, times_dec_reg, :] += level * dataloader.correction_tsr[:3].view((1,1,-1)).to(device)
                    cum_vel[f'level_{level}'][idx, :, times_dec_reg, :] += level * dataloader.correction_tsr[3:].view((1,1,-1)).to(device)

                    for i in range(level, args.num_levels):  
                        cum_loc[f'level_{i}'][idx, :, times_dec_reg, :] += pred_y_reg[..., :3]
                        cum_vel[f'level_{i}'][idx, :, times_dec_reg, :] += pred_y_reg[..., 3:6]
                        cum_coord[f'level_{i}'][idx, :, times_dec_reg, :] += pred_y_reg[..., :3] * (dataloader.max_loc - dataloader.min_loc) / 2 + (dataloader.max_loc + dataloader.min_loc) / 2
                    ''' 

                    # we are doing the aggregation on the latent space 
                    pred_latent_reg = pred_x_latent[l] 
                    idx = int(l / batch_actual_length * num_graphs) 
                    pred_latent[f'level_{level}'][idx, :, times_dec_reg, :] = pred_latent_reg 
                    for i in range(level, args.num_levels): 
                        x = cum_latent[f'level_{i}'] 

                        new_shape = x.shape[:-1] + (i+1, x.shape[-1] // (i+1))
                        x = x.view(new_shape) 
                        # x[idx, :, times_dec_reg, i, :] = pred_latent_reg # TODO: that's right, we need to add an extra dimension at -2 here
                        x[idx, :, times_dec_reg, level, :] = pred_latent_reg # ?????? 
                    
                    # to support the addition aggregation scheme 
                    # TODO: check correctness 
                    for i in range(level, args.num_levels): 
                        x = add_latent[f'level_{i}'] 
                        x[idx, :, times_dec_reg, :] += pred_latent_reg 

            y_prediction, y_value = {}, {} 
            for level in range(args.num_levels): 
                # predicted_loc, predicted_vel = [], [] 
                if (args.data in ["lj", "tip3p", "tip4p"]): 
                    predicted_loc_and_vel = [] 
                    groundtruth_loc, groundtruth_vel = [], [] 
                else: 
                    predicted_loc = [] 
                    groundtruth_loc = [] 
                decoder_mask = raw_batch[f"level_{level}"]["decoder_mask"] 

                '''
                # Supervision for the lookahead, to enforce consistent performance 
                # between train and test, especially in the presence of teacher forcing
                if (level + 1 < args.num_levels) and (mode == "train") and (args.supervise_lookahead is True): 
                    assert (len(decoder_mask) == num_graphs) 
                    for i in range(num_graphs): 
                        decoder_mask[i] = decoder_mask[i] | raw_batch[f"level_{level+1}"]["encoder_mask"][i] 
                ''' 

                for i in range(num_graphs): 
                    '''
                    predicted_loc.append(cum_loc[f'level_{level}'][i, :, decoder_mask[i], :]) 
                    predicted_vel.append(cum_vel[f'level_{level}'][i, :, decoder_mask[i], :]) 
                    ''' 
                    '''
                    if (args.data in ["lj", "tip3p", "tip4p"]): 
                        if (args.dec_aggregate == "concat"): 
                            predicted_loc_and_vel.append(model.decoder_phys[f'level_{level}'](cum_latent[f'level_{level}'][i, :, decoder_mask[i], :])) 
                        elif (args.dec_aggregate == "add"): 
                            predicted_loc_and_vel.append(model.decoder_phys[f'level_{level}'](add_latent[f'level_{level}'][i, :, decoder_mask[i], :]))
                        elif (args.dec_aggregate == "single"): # TODO: verify correctness and alignment 
                            predicted_loc_and_vel.append(model.decoder_phys[f'level_{level}'](pred_latent[f'level_{level}'][i, :, decoder_mask[i], :]))
                        else: 
                            raise Exception("Unsupported decoder aggregation scheme. ") 
                    else: 
                        if (args.dec_aggregate == "concat"): 
                            predicted_loc.append(model.decoder_phys[f'level_{level}'](cum_latent[f'level_{level}'][i, :, decoder_mask[i], :])) 
                        elif (args.dec_aggregate == "add"): 
                            predicted_loc.append(model.decoder_phys[f'level_{level}'](add_latent[f'level_{level}'][i, :, decoder_mask[i], :]))
                        elif (args.dec_aggregate == "single"): # TODO: verify correctness and alignment 
                            predicted_loc.append(model.decoder_phys[f'level_{level}'](pred_latent[f'level_{level}'][i, :, decoder_mask[i], :]))
                        else: 
                            raise Exception("Unsupported decoder aggregation scheme. ") 
                    ''' 

                    # use the designated level latent and decoder for all levels of prediction 
                    assert mode == "ablation" 
                    if (args.data in ["lj", "tip3p", "tip4p"]): 
                        if (args.dec_aggregate == "concat"): 
                            predicted_loc_and_vel.append(model.decoder_phys[f'level_{designated_level}'](cum_latent[f'level_{designated_level}'][i, :, decoder_mask[i], :])) 
                        elif (args.dec_aggregate == "add"): 
                            predicted_loc_and_vel.append(model.decoder_phys[f'level_{designated_level}'](add_latent[f'level_{designated_level}'][i, :, decoder_mask[i], :]))
                        elif (args.dec_aggregate == "single"): # TODO: verify correctness and alignment 
                            predicted_loc_and_vel.append(model.decoder_phys[f'level_{designated_level}'](pred_latent[f'level_{designated_level}'][i, :, decoder_mask[i], :]))
                        else: 
                            raise Exception("Unsupported decoder aggregation scheme. ") 
                    else: 
                        if (args.dec_aggregate == "concat"): 
                            predicted_loc.append(model.decoder_phys[f'level_{designated_level}'](cum_latent[f'level_{designated_level}'][i, :, decoder_mask[i], :])) 
                        elif (args.dec_aggregate == "add"): 
                            predicted_loc.append(model.decoder_phys[f'level_{designated_level}'](add_latent[f'level_{designated_level}'][i, :, decoder_mask[i], :]))
                        elif (args.dec_aggregate == "single"): # TODO: verify correctness and alignment 
                            predicted_loc.append(model.decoder_phys[f'level_{designated_level}'](pred_latent[f'level_{designated_level}'][i, :, decoder_mask[i], :]))
                        else: 
                            raise Exception("Unsupported decoder aggregation scheme. ") 


                    groundtruth_loc.append(raw_loc[i, :, decoder_mask[i], :]) 
                    if (args.data in ["lj", "tip3p", "tip4p"]): 
                        groundtruth_vel.append(raw_vel[i, :, decoder_mask[i], :]) 
                '''
                predicted_loc = torch.cat(predicted_loc, dim=0) 
                predicted_vel = torch.cat(predicted_vel, dim=0) 
                ''' 
                groundtruth_loc = torch.cat(groundtruth_loc, dim=0) 
                if (args.data in ["lj", "tip3p", "tip4p"]): 
                    groundtruth_vel = torch.cat(groundtruth_vel, dim=0) 
                    y_prediction[f'level_{level}'] = torch.cat(predicted_loc_and_vel, dim=0)
                    y_value[f'level_{level}'] = torch.cat([groundtruth_loc, groundtruth_vel], dim=-1) 
                else: 
                    # y_prediction[f'level_{level}'] = torch.cat([predicted_loc, predicted_vel], dim=-1) 
                    y_prediction[f'level_{level}'] = torch.cat(predicted_loc, dim=0) 
                    # y_value[f'level_{level}'] = torch.cat([groundtruth_loc, groundtruth_vel], dim=-1) 
                    y_value[f'level_{level}'] = groundtruth_loc 

            train_res = {} 

            # print("Does sharing the same `info` across levels cause any problem? ") 
            info_shared = info[f'level_{0}'] 

            for level in range(args.num_levels): 
                # train_res[f'level_{level}'] = model.compute_losses(groundtruth=y_value[f'level_{level}'], pred_y=y_prediction[f'level_{level}'], info=info[f'level_{level}'], temporal_weights=temporal_weights[f'level_{level}'], kl_coef=kl_coef, use_vae=args.use_vae)
                train_res[f'level_{level}'] = model.compute_losses(groundtruth=y_value[f'level_{level}'], pred_y=y_prediction[f'level_{level}'], info=info_shared, temporal_weights=temporal_weights[f'level_{level}'], kl_coef=kl_coef, use_vae=args.use_vae)
            
            
            loss = torch.tensor(0.0).to(device) 
            if (args.iterative_training is True): 
                scheduled_weights = get_weights_for_epoch(epo, schedules) 
                print(f"Epoch {epo}: Training with scheduled weights {scheduled_weights}")
                for level in range(args.num_levels): 
                    loss += scheduled_weights[level] * train_res[f'level_{level}']['loss']
            else: 
                print(f"Epoch {epo}: Training with uniform weights {args.level_weights}")
                for level in range(args.num_levels): 
                    loss += args.level_weights[level] * train_res[f'level_{level}']['loss'] # ?????? 
            # loss = loss / args.num_levels 

            if mode == "train":
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) 
                optimizer.step()

            #saving results
            loss_list.append(loss.detach().cpu().numpy())  

            
            for level in range(args.num_levels): 
                loss_hier_list[level].append(train_res[f'level_{level}']['loss'])
                mse_hier_list[level].append(train_res[f'level_{level}']['mse'])
                mae_hier_list[level].append(train_res[f'level_{level}']['mae'])
                mape_hier_list[level].append(train_res[f'level_{level}']['mape'])
                likelihood_hier_list[level].append(train_res[f'level_{level}']['likelihood'])
                kl_first_p_hier_list[level].append(train_res[f'level_{level}']['kl_first_p'])
                std_first_p_hier_list[level].append(train_res[f'level_{level}']['std_first_p'])

            '''
            profiling for the accuracy of lookahead values themselves 
            '''
            # take note: "lookahead" is no longer "lookhead" in the current context 
            # TODO: you need to be careful! This is not the predicted value we want! This prediction is "one level short". 
            # i.e. it is used to construct the graph used for prediction, and it does not take into account the current level 
            # results that has just been computed. 
            with torch.no_grad():  
                lookahead_y_prediction, lookahead_y_value = {}, {} 
                for level in range(args.num_levels - 1): 
                    # lookahead_predicted_loc, lookahead_predicted_vel = [], [] 
                    if (args.data in ["lj", "tip3p", "tip4p"]): 
                        lookahead_predicted_loc_and_vel = [] 
                        lookahead_groundtruth_loc, lookahead_groundtruth_vel = [], [] 
                    else: 
                        lookahead_predicted_loc = [] 
                        lookahead_groundtruth_loc = [] 
                    lookahead_mask = raw_batch[f"level_{level+1}"]["encoder_mask"] 
                    for i in range(num_graphs): 
                        '''
                        lookahead_predicted_loc.append(cum_loc[f'level_{level}'][i, :, lookahead_mask[i], :]) 
                        lookahead_predicted_vel.append(cum_vel[f'level_{level}'][i, :, lookahead_mask[i], :]) 
                        ''' 
                        """                     
                        if (args.data in ["lj", "tip3p", "tip4p"]): 
                            if (args.dec_aggregate == "concat"): 
                                lookahead_predicted_loc_and_vel.append(model.decoder_phys[f'level_{level}'](cum_latent[f'level_{level}'][i, :, lookahead_mask[i], :]))
                            elif (args.dec_aggregate == "add"): 
                                lookahead_predicted_loc_and_vel.append(model.decoder_phys[f'level_{level}'](add_latent[f'level_{level}'][i, :, lookahead_mask[i], :])) 
                            elif (args.dec_aggregate == "single"): # TODO: verify correctness and alignment 
                                lookahead_predicted_loc_and_vel.append(model.decoder_phys[f'level_{level}'](pred_latent[f'level_{level}'][i, :, lookahead_mask[i], :])) 
                            else: 
                                raise Exception("Unsupported decoder aggregation scheme. ") 

                        else: # protein dataset 
                            if (args.dec_aggregate == "concat"): 
                                lookahead_predicted_loc.append(model.decoder_phys[f'level_{level}'](cum_latent[f'level_{level}'][i, :, lookahead_mask[i], :]))
                            elif (args.dec_aggregate == "add"): 
                                lookahead_predicted_loc.append(model.decoder_phys[f'level_{level}'](add_latent[f'level_{level}'][i, :, lookahead_mask[i], :])) 
                            elif (args.dec_aggregate == "single"): # TODO: verify correctness and alignment 
                                lookahead_predicted_loc.append(model.decoder_phys[f'level_{level}'](pred_latent[f'level_{level}'][i, :, lookahead_mask[i], :])) 
                            else: 
                                raise Exception("Unsupported decoder aggregation scheme. ") 
                        """

                        # use the designated level latent and decoder for all levels of prediction 
                        assert (mode == "ablation") 
                        if (args.data in ["lj", "tip3p", "tip4p"]): 
                            if (args.dec_aggregate == "concat"): 
                                lookahead_predicted_loc_and_vel.append(model.decoder_phys[f'level_{designated_level}'](cum_latent[f'level_{designated_level}'][i, :, lookahead_mask[i], :]))
                            elif (args.dec_aggregate == "add"): 
                                lookahead_predicted_loc_and_vel.append(model.decoder_phys[f'level_{designated_level}'](add_latent[f'level_{designated_level}'][i, :, lookahead_mask[i], :])) 
                            elif (args.dec_aggregate == "single"): # TODO: verify correctness and alignment 
                                lookahead_predicted_loc_and_vel.append(model.decoder_phys[f'level_{designated_level}'](pred_latent[f'level_{designated_level}'][i, :, lookahead_mask[i], :])) 
                            else: 
                                raise Exception("Unsupported decoder aggregation scheme. ") 

                        else: # protein dataset 
                            if (args.dec_aggregate == "concat"): 
                                lookahead_predicted_loc.append(model.decoder_phys[f'level_{designated_level}'](cum_latent[f'level_{designated_level}'][i, :, lookahead_mask[i], :]))
                            elif (args.dec_aggregate == "add"): 
                                lookahead_predicted_loc.append(model.decoder_phys[f'level_{designated_level}'](add_latent[f'level_{designated_level}'][i, :, lookahead_mask[i], :])) 
                            elif (args.dec_aggregate == "single"): # TODO: verify correctness and alignment 
                                lookahead_predicted_loc.append(model.decoder_phys[f'level_{designated_level}'](pred_latent[f'level_{designated_level}'][i, :, lookahead_mask[i], :])) 
                            else: 
                                raise Exception("Unsupported decoder aggregation scheme. ") 

                        lookahead_groundtruth_loc.append(raw_loc[i, :, lookahead_mask[i], :]) 
                        if (args.data in ["lj", "tip3p", "tip4p"]): 
                            lookahead_groundtruth_vel.append(raw_vel[i, :, lookahead_mask[i], :]) 
                    '''
                    lookahead_predicted_loc = torch.cat(lookahead_predicted_loc, dim=0) 
                    lookahead_predicted_vel = torch.cat(lookahead_predicted_vel, dim=0) 
                    ''' 
                    lookahead_groundtruth_loc = torch.cat(lookahead_groundtruth_loc, dim=0) 
                    if (args.data in ["lj", "tip3p", "tip4p"]): 
                        lookahead_groundtruth_vel = torch.cat(lookahead_groundtruth_vel, dim=0) 
                        lookahead_y_prediction[f'level_{level}'] = torch.cat(lookahead_predicted_loc_and_vel, dim=0)  
                        lookahead_y_value[f'level_{level}'] = torch.cat([lookahead_groundtruth_loc, lookahead_groundtruth_vel], dim=-1) 
                    else: 
                        # lookahead_y_prediction[f'level_{level}'] = torch.cat([lookahead_predicted_loc, lookahead_predicted_vel], dim=-1) 
                        lookahead_y_prediction[f'level_{level}'] = torch.cat(lookahead_predicted_loc, dim=0)  
                        # lookahead_y_value[f'level_{level}'] = torch.cat([lookahead_groundtruth_loc, lookahead_groundtruth_vel], dim=-1) 
                        lookahead_y_value[f'level_{level}'] = lookahead_groundtruth_loc 

                lookahead_train_res = {} 
                for level in range(args.num_levels - 1): 
                    # lookahead_train_res[f'level_{level}'] = model.compute_losses(groundtruth=lookahead_y_value[f'level_{level}'], pred_y=lookahead_y_prediction[f'level_{level}'], info=info[f'level_{level}'], temporal_weights=temporal_weights[f'level_{level}'], kl_coef=kl_coef, use_vae=args.use_vae) 
                    lookahead_train_res[f'level_{level}'] = model.compute_losses(groundtruth=lookahead_y_value[f'level_{level}'], pred_y=lookahead_y_prediction[f'level_{level}'], info=info_shared, temporal_weights=temporal_weights[f'level_{level}'], kl_coef=kl_coef, use_vae=args.use_vae) 

                for level in range(args.num_levels - 1): 
                    lookahead_mse_hier_list[level].append(lookahead_train_res[f'level_{level}']['mse'])
                    lookahead_mae_hier_list[level].append(lookahead_train_res[f'level_{level}']['mae'])
                    lookahead_mape_hier_list[level].append(lookahead_train_res[f'level_{level}']['mape'])
                    lookahead_likelihood_hier_list[level].append(lookahead_train_res[f'level_{level}']['likelihood'])
                    lookahead_kl_first_p_hier_list[level].append(lookahead_train_res[f'level_{level}']['kl_first_p'])
                    lookahead_std_first_p_hier_list[level].append(lookahead_train_res[f'level_{level}']['std_first_p'])




        message_train = []
        message_train.append('Epoch {:04d} [{} {} seq (cond on sampled tp)] | Backprop Loss {:.6f} |'.format(epo, mode, designated_level, np.mean(loss_list))) 
        for level in range(args.num_levels): 
            loss_level_detached = [x.detach().data.item() for x in loss_hier_list[level]]
            message_train.append('Level {:04d}: Loss {:.6f} | MSE {:.6f} | RMSE {:.6f} | MAE {:.6f} | MAPE {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
                level, np.mean(loss_level_detached), np.mean(mse_hier_list[level]), np.sqrt(np.mean(mse_hier_list[level])), np.mean(mae_hier_list[level]), np.mean(mape_hier_list[level]), np.mean(likelihood_hier_list[level]),  
                np.mean(kl_first_p_hier_list[level]), np.mean(std_first_p_hier_list[level]) 
            ))

        current_mse = 0.0 
        horizons = []  
        for level in range(args.num_levels): 
            if (level == 0): 
                current_horizon = args.extrap_scope / args.ode_forecast_frequencies[level] 
            else: 
                current_horizon = args.extrap_scope / args.ode_forecast_frequencies[level] - args.extrap_scope / args.ode_forecast_frequencies[level-1] 
            horizons.append(current_horizon) 
            current_mse += current_horizon * np.mean(mse_hier_list[level]) 
        
        current_mse /= np.sum(horizons) 
        message_train.append("Weighted overall MSE {:.6f}".format(current_mse)) 
        
        # Check if the current MSE is the best so far
        if (mode == "test") and (current_mse < best_mse): 
            best_mse = current_mse
            best_epoch = epo
            # Save the model checkpoint
            # You need to replace 'model' with the actual model variable name and specify the file path
            torch.save(model.state_dict(), ckpt_filename) 
            message_train.append("Best model saved at epoch {:04d}".format(epo)) 


        '''
        profiling for the accuracy of lookahead values themselves 
        '''
        lookahead_message_train = []
        lookahead_message_train.append("Lookahead accuracy profiling: ") 
        for level in range(args.num_levels - 1): 
            lookahead_message_train.append('Level {:04d} (+1): MSE {:.6f} | RMSE {:.6f} | MAE {:.6f} | MAPE {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
                level, np.mean(lookahead_mse_hier_list[level]), np.sqrt(np.mean(lookahead_mse_hier_list[level])), np.mean(lookahead_mae_hier_list[level]), np.mean(lookahead_mape_hier_list[level]), np.mean(lookahead_likelihood_hier_list[level]),  
                np.mean(lookahead_kl_first_p_hier_list[level]), np.mean(lookahead_std_first_p_hier_list[level]) 
            ))

        ''' 
        del loss_list; del loss_hier_list; del mse_hier_list; del mae_hier_list; del mape_hier_list; del likelihood_hier_list; del kl_first_p_hier_list; del std_first_p_hier_list 
        del pred_loc; del pred_vel; del pred_coord; del cum_loc; del cum_vel; del cum_coord 
        del raw_loc; del raw_vel; del raw_coord; del raw_times 
        del pred_y; del info; del temporal_weights 
        del y_prediction; del y_value 
        del train_res; del loss 
        del predicted_loc; del predicted_vel; del groundtruth_loc; del groundtruth_vel 
        del encoder_mask; del decoder_mask; del union_decoder_mask; del times_enc; del times_dec 
        del loc_enc; del vel_enc; del coord_enc; 
        del enc_position_mask; del dec_position_mask 
        del encoder_batch; del decoder_batch 
        del pred_y_reg; del times_dec_reg 
        del batch_actual_length 
        del raw_batch 
        ''' 
        torch.cuda.empty_cache() 

        return '\n'.join(message_train), '\n'.join(lookahead_message_train), kl_coef

    
    # Load the JSON file
    with open('hyperparameters.json', 'r') as file:
        hyperparameters = json.load(file)

    # Access the total_dt list
    total_dt = hyperparameters['total_dt'] 
    schedules = hyperparameters['schedules'] 

    with torch.no_grad(): 
        for epo in range(1, args.niters + 1): 
            for designated_level in range(args.num_levels): 
                message_abalation, lookahead_message_ablation, kl_coef = train_epoch(epo, mode="ablation", designated_level=designated_level)  
                fprint(message_abalation) 
                fprint(lookahead_message_ablation)

                torch.cuda.empty_cache()

    '''
    for epo in range(1, args.niters + 1):
        message_train, lookahead_message_train, kl_coef = train_epoch(epo, mode="train")  
        fprint(message_train) 
        fprint(lookahead_message_train)

        with torch.no_grad(): 
            message_test, lookahead_message_test, kl_coef = train_epoch(epo, mode="test") 
            fprint(message_test) 
            fprint(lookahead_message_test) 

            for select in total_dt: 
                message_select, lookahead_message_select, kl_coef = train_epoch(epo, mode="select", select=select) 
                fprint(message_select) 
                fprint(lookahead_message_select) 

        torch.cuda.empty_cache()
    ''' 


    '''
        if epo % n_iters_to_viz == 0:
            model.eval()
            test_res = compute_loss_all_batches(model, test_encoder, None, test_decoder,
                                                n_batches=test_batch, device=device,
                                                n_traj_samples=3, kl_coef=kl_coef, 
                                                dataloader=dataloader)


            message_test = []
            message_test.append('Epoch {:04d} [Test seq (cond on sampled tp)] | Backprop Loss {:.6f} | Original MSE {:.6f} | Original RMSE {:.6f} | Original MAE {:.6f} | Original MAPE {:.6f} |'.format(epo, test_res["loss"], test_res["mse_original"], torch.sqrt(test_res["mse_original"]), test_res["mae_original"], test_res["mape_original"])) 
            for layer in range(args.n_hiers): 
                message_test.append('Layer {:01d}: Loss {:.6f} | MSE {:.6f} | RMSE {:.6f} | MAE {:.6f} | MAPE {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
                    layer, test_res[f'layer_{layer}']['loss'], test_res[f'layer_{layer}']['mse'], np.sqrt(test_res[f'layer_{layer}']['mse']), test_res[f'layer_{layer}']['mae'], test_res[f'layer_{layer}']['mape'], test_res[f'layer_{layer}']['likelihood'], 
                    test_res[f'layer_{layer}']['kl_first_p'], test_res[f'layer_{layer}']['std_first_p'], 
                ))
            
            message_test = '\n'.join(message_test) 

            logger.info("Experiment " + str(experimentID))
            logger.info(message_train)
            logger.info(message_test)
            logger.info("KL coef: {}".format(kl_coef))
            fprint("data: %s, encoder: %s, sample: %s, mode:%s" % (
                args.data, args.z0_encoder, str(args.sample_percent_train), args.mode))

            if test_res["mse_original"] < best_test_mse:
                best_test_mse = test_res["mse_original"]
                message_best = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Best original mse {:.6f}|'.format(epo,
                                                                                                        best_test_mse)
                logger.info(message_best)
                ckpt_path = os.path.join(args.save, "experiment_" + str(
                    experimentID) + "_" + args.z0_encoder + "_" + args.data + "_" + str(
                    args.sample_percent_train) + "_" + args.mode + "_epoch_" + str(epo) + "_mse_" + str(
                    best_test_mse) + '.ckpt')
                torch.save({
                    'args': args,
                    'state_dict': model.state_dict(),
                }, ckpt_path)
        '''


    fprint("Training summary: ")
    fprint("Best MSE: {:.6f} at epoch {:04d}".format(best_mse, best_epoch)) 










