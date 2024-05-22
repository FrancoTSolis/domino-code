from lib.base_models import VAE_Baseline
import lib.utils as utils
import torch

class LatentGraphODE(VAE_Baseline):
	def __init__(self, input_dim, latent_dim, encoder_z0, decoder_phys, decoder_latent, diffeq_solver,
				 z0_prior, device, obsrv_std=None, use_vae=False):

		super(LatentGraphODE, self).__init__(
			input_dim=input_dim, latent_dim=latent_dim,
			z0_prior=z0_prior,
			device=device, obsrv_std=obsrv_std)

		self.encoder_z0 = encoder_z0
		self.diffeq_solver = diffeq_solver
		self.decoder_phys = decoder_phys 
		self.decoder_latent = decoder_latent 
		self.latent_dim =latent_dim
		self.use_vae = use_vae 




	def get_reconstruction(self, batch_en, batch_de, level_name, n_traj_samples=1, max_loc=None, min_loc=None): 
		assert (self.use_vae == True) 
		
		#Encoder:
		if level_name == "level_0":  
			# first_point_mu, first_point_std, edge_idx_latest_observed, x_coord_latest_observed = \
			# 	self.encoder_z0[level_name](batch_en.x, batch_en.x_coord, batch_en.pos, batch_en.batch, batch_en.y) 						  

			# first_point_mu, first_point_std, edge_idx_latest_observed, x_coord_latest_observed = self.encoder_z0[level_name](batch_en) 				  
			first_point_mu, first_point_std, x_coord_latest_observed = self.encoder_z0[level_name](batch_en) 				  


			means_z0 = first_point_mu.repeat(n_traj_samples,1,1) #[3,num_ball,10]
			sigmas_z0 = first_point_std.repeat(n_traj_samples,1,1) #[3,num_ball,10]
			first_point_enc = utils.sample_standard_gaussian(means_z0, sigmas_z0) #[3,num_ball,10]

			first_point_std = first_point_std.abs() 
			assert (torch.sum(first_point_std < 0) == 0.)
		else: 
			curr_level = int(level_name.split("_")[-1])
			# for subsequent levels, 
			# the initial states are refined from the corresponding 0th-level outputs 
			# the graph structure are decoded from the 0th-level outputs 
			first_point_enc = self.decoder_latent[f'level_{curr_level-1}'](batch_en.x).repeat(n_traj_samples,1,1) # `x` is the concatenated latent states from the previous levels  
			# pred_x = self.decoder_phys[f'level_{curr_level-1}'](batch_en.x) # need to cope with some more details here, just as in the encoder_z0 function 
			pred_x = self.decoder_phys[f'level_{curr_level-1}'](batch_en.graph_x) # projector and decoder aggregation options can vary 
			pred_coord = pred_x[..., :3] * (max_loc - min_loc) / 2 + (max_loc + min_loc) / 2
			# print("assuming sequence length to be 1. Already the last time step, no need for further truncation. ")
			x_coord_latest_observed = pred_coord.reshape((batch_en.num_graphs, -1, 3)) # need to be corrected 
			# level - 1????????
			# TODO: implement the output_graph function in the encoder_z0 class 

		'''
		else: 
			first_point_enc, edge_idx_latest_observed, x_coord_latest_observed = \
				self.encoder_z0[level_name](batch_en.x, batch_en.x_coord, batch_en.pos, batch_en.batch, batch_en.y) 						  

			first_point_enc = first_point_enc.repeat(n_traj_samples,1,1) #[3,num_ball,10]
		''' 			
		

		time_steps_to_predict = batch_de["time_steps"] # combined_tt 

		first_point_enc_ = first_point_enc.view(n_traj_samples, len(time_steps_to_predict), first_point_enc.shape[-2] // len(time_steps_to_predict), self.latent_dim) #[3,num_ball*10,10]

		assert (not torch.isnan(first_point_enc).any())

		pred_x = [] 
		assert (n_traj_samples == 1) 
		for i in range(len(time_steps_to_predict)): 
			# print("should update latest coordinates once! ") 
			sol = self.diffeq_solver[level_name](first_point_enc_[:, i], time_steps_to_predict[i], None, None, x_coord_latest_observed[i:i+1]) 
			# pred = self.decoder[level_name](sol) 
			# pred_x.append(pred.squeeze(0)) 		
			pred_x.append(sol.squeeze(0)) 



		# ODE:Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
		# sol_y = self.diffeq_solver[level_name](first_point_enc, time_steps_to_predict, None, edge_idx_latest_observed, x_coord_latest_observed)


        # Decoder:
		# pred_x = self.decoder[level_name](sol_y)

		if level_name == "level_0": 
			all_extra_info = {
				"first_point": (torch.unsqueeze(first_point_mu,0), torch.unsqueeze(first_point_std,0), first_point_enc),
				# "latent_traj": sol_y.detach()
			}
		else: 
			all_extra_info = None 

		return pred_x, all_extra_info, None









