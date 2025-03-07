import os
import logging
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def makedirs(dirname):
	if not os.path.exists(dirname):
		os.makedirs(dirname)

def save_checkpoint(state, save, epoch):
	if not os.path.exists(save):
		os.makedirs(save)
	filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
	torch.save(state, filename)

	
def get_logger(logpath, filepath, package_files=[],
			   displaying=True, saving=True, debug=False):
	logger = logging.getLogger()
	if debug:
		level = logging.DEBUG
	else:
		level = logging.INFO
	logger.setLevel(level)
	if saving:
		info_file_handler = logging.FileHandler(logpath, mode='w')
		info_file_handler.setLevel(level)
		logger.addHandler(info_file_handler)
	if displaying:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)
		logger.addHandler(console_handler)
	logger.info(filepath)

	for f in package_files:
		logger.info(f)
		with open(f, 'r') as package_f:
			logger.info(package_f.read())

	return logger


def inf_generator(iterable):
	"""Allows training with DataLoaders in a single infinite loop:
		for i, (x, y) in enumerate(inf_generator(train_loader)):
	"""
	iterator = iterable.__iter__()
	while True:
		try:
			yield iterator.__next__()
		except StopIteration:
			iterator = iterable.__iter__()

def dump_pickle(data, filename):
	with open(filename, 'wb') as pkl_file:
		pickle.dump(data, pkl_file)

def load_pickle(filename):
	with open(filename, 'rb') as pkl_file:
		filecontent = pickle.load(pkl_file)
	return filecontent

def init_network_weights(net, std = 0.1):
	for m in net.modules():
		if isinstance(m, nn.Linear):
			nn.init.normal_(m.weight, mean=0, std=std)
			nn.init.constant_(m.bias, val=0)

def flatten(x, dim):
	return x.reshape(x.size()[:dim] + (-1, ))


def get_device(tensor):
	device = torch.device("cpu")
	if tensor.is_cuda:
		device = tensor.get_device()
	return device

def sample_standard_gaussian(mu, sigma):
	device = get_device(mu)

	d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
	r = d.sample(mu.size()).squeeze(-1)
	return r * sigma.float() + mu.float()

def get_dict_template():
	return {"data": None,
			"time_setps": None,
			"mask": None
			}
def get_next_batch_new(dataloader,device, is_dict):
	data_dict = dataloader.__next__()
	
	# used in combination with `custom_enc_collate()`  
	if is_dict == True: 
		assert (type(data_dict) is dict)
		for key in data_dict.keys(): 
			data_dict[key] = data_dict[key].to(device) 
		return data_dict 
	#device_now = data_dict.batch.device
	return data_dict.to(device)

def get_next_batch(dataloader,device):
	# Make the union of all time points and perform normalization across the whole dataset
	data_dict = dataloader.__next__()

	batch_dict = get_dict_template()


	batch_dict["data"] = data_dict["data"].to(device)
	batch_dict["time_steps"] = data_dict["time_steps"].to(device)
	batch_dict["mask"] = data_dict["mask"].to(device)

	batch_dict["hiers_data"] = dict() 
	for x in data_dict["hiers_data"].keys(): 
		batch_dict["hiers_data"][x] = data_dict["hiers_data"][x].to(device) 

	return batch_dict


def get_ckpt_model(ckpt_path, model, device):
	if not os.path.exists(ckpt_path):
		raise Exception("Checkpoint " + ckpt_path + " does not exist.")
	# Load checkpoint.
	state_dict = torch.load(ckpt_path)
	# ckpt_args = checkpt['args']
	# state_dict = checkpt['state_dict']
	model_dict = model.state_dict()

	# 1. filter out unnecessary keys
	state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(state_dict) 
	# 3. load the new state dict
	model.load_state_dict(state_dict)
	model.to(device)


def update_learning_rate(optimizer, decay_rate = 0.999, lowest = 1e-3):
	for param_group in optimizer.param_groups:
		lr = param_group['lr']
		lr = max(lr * decay_rate, lowest)
		param_group['lr'] = lr


def linspace_vector(start, end, n_points):
	# start is either one value or a vector
	size = np.prod(start.size())

	assert(start.size() == end.size())
	if size == 1:
		# start and end are 1d-tensors
		res = torch.linspace(start, end, n_points)
	else:
		# start and end are vectors
		res = torch.Tensor()
		for i in range(0, start.size(0)):
			res = torch.cat((res, 
				torch.linspace(start[i], end[i], n_points)),0)
		res = torch.t(res.reshape(start.size(0), n_points))
	return res

def reverse(tensor):
	idx = [i for i in range(tensor.size(0)-1, -1, -1)]
	return tensor[idx]

def create_net(n_inputs, n_outputs, n_layers = 1,
	n_units = 100, nonlinear = nn.Tanh):
	layers = [nn.Linear(n_inputs, n_units)]
	for i in range(n_layers):
		layers.append(nonlinear())
		layers.append(nn.Linear(n_units, n_units))

	layers.append(nonlinear())
	layers.append(nn.Linear(n_units, n_outputs))
	return nn.Sequential(*layers)




def compute_loss_all_batches(model,
	encoder,graph,decoder,
	n_batches, device,
	n_traj_samples = 1, kl_coef = 1., 
	dataloader=None):

	assert (graph is None)

	'''
	total = {}
	total["loss"] = 0
	total["likelihood"] = 0
	total["mse"] = 0
	total["kl_first_p"] = 0
	total["std_first_p"] = 0
	'''


	n_test_batches = 0

	n_hiers = len(model.diffeq_solver) 

	total = {} 
	total["loss"] = 0.0 
	total["mse_original"] = 0.0 
	total["mae_original"] = 0.0 
	total["mape_original"] = 0.0 
	for layer in range(n_hiers): 
		total[f'layer_{layer}'] = {
			"loss": 0.0, 
			"likelihood": 0.0, 
			"mse": 0.0, 
			"mae": 0.0, 
			"mape": 0.0, 
			"kl_first_p": 0.0, 
			"std_first_p": 0.0, 
		}



	model.eval()
	print("Computing loss... ")
	with torch.no_grad():
		for i in tqdm(range(n_batches)):
			batch_dict_encoder = get_next_batch_new(encoder, device, is_dict=True)
			# batch_dict_graph = get_next_batch_new(graph, device)
			batch_dict_decoder = get_next_batch(decoder, device)

			'''
			results = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, batch_dict_graph,
											   n_traj_samples=n_traj_samples, kl_coef=kl_coef)
			'''

			'''
			results = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, None,
											   n_traj_samples=n_traj_samples, kl_coef=kl_coef)
			''' 


			test_res = {}
			pred_y = {}

			# train_res['aggr'] = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, None, layer_name='aggr', n_traj_samples=3, kl_coef=kl_coef)

			for layer in range(n_hiers): 
				pred_y[f'layer_{layer}'], test_res[f'layer_{layer}'] = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, None, layer_name=f'layer_{layer}', n_traj_samples=3, kl_coef=kl_coef)

			# loss = train_res["loss"] 
			pred_y_original = torch.zeros_like(pred_y[f'layer_{0}']) 
			for layer in range(n_hiers): 
				pred_y_original = pred_y_original + pred_y[f'layer_{layer}'] 
			pred_y_original += dataloader.correction_tsr.view((1,1,1,-1)).to(device) 
			mse_original = model.get_mse(batch_dict_decoder["data"], pred_y_original, mask=batch_dict_decoder["mask"])  # [1]
			mae_original = model.get_mae(batch_dict_decoder["data"], pred_y_original, mask=batch_dict_decoder["mask"])  # [1]
			mape_original = model.get_mape(batch_dict_decoder["data"], pred_y_original, mask=batch_dict_decoder["mask"])  # [1]
	
			loss = 0.0 
			for layer in range(n_hiers): 
				loss += test_res[f'layer_{layer}']['loss']  
			loss = loss / n_hiers 

			total["loss"] += loss
			total["mse_original"] += mse_original
			total["mae_original"] += mae_original
			total["mape_original"] += mape_original

			for layer in range(n_hiers): 
				for key in total[f'layer_{layer}'].keys(): 
					if key in test_res[f'layer_{layer}']: 
						var = test_res[f'layer_{layer}'][key]
						if isinstance(var, torch.Tensor): 
							var = var.detach().item() 
						total[f'layer_{layer}'][key] += var 



			'''			
			for key in total.keys():
				if key in results:
					var = results[key]
					if isinstance(var, torch.Tensor):
						var = var.detach().item()
					total[key] += var
			''' 

			n_test_batches += 1

			del batch_dict_encoder,batch_dict_decoder,test_res

		if n_test_batches > 0:
			total["loss"] /= n_test_batches 
			total["mse_original"] /= n_test_batches 
			total["mae_original"] /= n_test_batches 
			total["mape_original"] /= n_test_batches 

			for layer in range(n_hiers): 
				for key, _ in total[f'layer_{layer}'].items(): 
					total[f'layer_{layer}'][key] /= n_test_batches


	return total






