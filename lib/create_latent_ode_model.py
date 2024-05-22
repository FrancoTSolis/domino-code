from lib.gnn_models import GNN
from lib.latent_ode import LatentGraphODE
from lib.encoder_decoder import *
from lib.diffeq_solver import DiffeqSolver,GraphODEFunc

import argparse
# from code.LJ.train_network_lj import ParticleNetLightning
# from gamd_code.LJ import ParticleNetLightning

# https://stackoverflow.com/a/21995949 # Importing files from different folder



def create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, use_vae):
	'''
	if args.data == "lj": 
		from gamd_code.LJ import ParticleNetLightning
	elif args.data == "tip3p":
		from gamd_code.tip3p import ParticleNetLightning 
	elif args.data == "tip4p": 
		from gamd_code.tip4p import ParticleNetLightning # TODO: add tip4p 
	elif args.data == "ala2":
		from lib.cpainn import PaiNNEncoder, PaiNNODENet 
	else: 
		raise Exception("Unsupported dataset. ")
	'''
	from lib.cpainn import PaiNNEncoder, PaiNNODENet 


	# dim related
	latent_dim = args.latents # ode output dimension
	# rec_dim = args.rec_dims
	input_dim = input_dim
	# ode_dim = args.ode_dims #ode gcn dimension
	# n_heads = args.n_heads 

	#encoder related

	''' Deprecate the LG-ODE GNN Encoder 
	encoder_z0 = GNN(in_dim=input_dim, n_hid=rec_dim, out_dim=latent_dim, n_heads=args.n_heads,
						 n_layers=args.rec_layers, dropout=args.dropout, conv_name=args.z0_encoder,
						 aggregate=args.rec_attention)  # [b,n_ball,e]
	'''

	enc_parser = argparse.ArgumentParser(description="arguments for the ParticleNetLighting encoder from GAMD. ")
	# enc_parser.add_argument('--in_feats', type=int)
	enc_parser.add_argument('--encoding_size', type=int)
	enc_parser.add_argument('--out_feats', type=int)
	enc_parser.add_argument('--hidden_dim', type=int)
	enc_parser.add_argument('--final_dim', type=int)
	enc_parser.add_argument('--edge_embedding_dim', type=int)
	enc_parser.add_argument('--drop_edge', action='store_true', help='Description for flag')
	enc_parser.add_argument('--use_layer_norm', action='store_true', help='Description for flag')
	enc_parser.add_argument('--aggregate', type=str)
	# enc_parser.add_argument('--n_heads', type=int)
	enc_parser.add_argument("--indistinguishable", action='store_true')
	enc_parser.add_argument('--data', type=str) 

	enc_args = enc_parser.parse_args([
		# '--in_feats', str(input_dim), # 6, size of the original node feature 
		'--encoding_size', '128', # embedding size of each node 
		'--out_feats', '128', 
		'--hidden_dim', '128', # 128 for LJ
		'--final_dim', str(latent_dim), # 16 
		'--edge_embedding_dim', '128', # 128 for LJ 
		# '--drop_edge', 
		'--use_layer_norm', 
		'--aggregate', args.rec_attention, # 'attention'
		# '--n_heads', str(args.n_heads), 
		'--data', args.data, ]
		+ (["--indistinguishable"] if args.indistinguishable else []) 
	)
	# encoder_z0 = ParticleNetLightning(args=enc_args)
	enc_args.cutoff = args.cutoff 

	#ODE related
	if args.augment_dim > 0:
		ode_input_dim = latent_dim + args.augment_dim
	else:
		ode_input_dim = latent_dim


	''' Deprecate the LG-ODE GNN ode function 
	ode_func_net = GNN(in_dim = ode_input_dim,n_hid =ode_dim,out_dim = ode_input_dim,n_heads=args.n_heads,n_layers=args.gen_layers,dropout=args.dropout,conv_name = args.odenet,aggregate="add")
	'''
	ode_parser = argparse.ArgumentParser(description="arguments for the ODE Function. ")
	ode_parser.add_argument('--in_feats', type=int)
	ode_parser.add_argument('--encoding_size', type=int)
	ode_parser.add_argument('--out_feats', type=int)
	ode_parser.add_argument('--hidden_dim', type=int)
	ode_parser.add_argument('--final_dim', type=int)
	ode_parser.add_argument('--edge_embedding_dim', type=int)
	ode_parser.add_argument('--drop_edge', action='store_true', help='Description for flag')
	ode_parser.add_argument('--use_layer_norm', action='store_true', help='Description for flag')
	ode_parser.add_argument('--aggregate', type=str)
	ode_parser.add_argument("--indistinguishable", action='store_true')
	ode_parser.add_argument('--data', type=str) 
	

	# ode_parser.add_argument('--n_heads', type=int)

	ode_args = ode_parser.parse_args([
		'--in_feats', str(ode_input_dim), # 80 
		'--encoding_size', '128', # embedding size of each node 
		'--out_feats', '128', 
		'--hidden_dim', '128', # 128 for LJ 
		'--final_dim', str(ode_input_dim), 
		'--edge_embedding_dim', '128', # 128 for LJ 
		# '--drop_edge', 
		'--use_layer_norm', 
		'--aggregate', 'add', 
		# '--n_heads', str(args.n_heads), 
		'--data', args.data, ] 
		+ (["--indistinguishable"] if args.indistinguishable else []) 
	)
	ode_args.cutoff = args.cutoff 

	encoder_z0 = nn.ModuleDict() 
	ode_func_net = nn.ModuleDict() 
	gen_ode_func = nn.ModuleDict() 
	diffeq_solver = nn.ModuleDict() 
	# decoder = nn.ModuleDict() 
	decoder_phys = nn.ModuleDict() 
	decoder_latent = nn.ModuleDict() 

	# for v2, we need to create multiple encoders, ode functions, and decoders 
	score_model_kwargs = {
        # "n_features": args.n_features,
        "max_lag": args.max_lag,
        "diff_steps": args.diff_steps,
		"num_of_atoms": args.n_balls, 
    }
	
	for level in range(args.num_levels):  
		encoder_z0[f'level_{level}'] = PaiNNEncoder(enc_args, **score_model_kwargs, use_vae=use_vae) 
		ode_func_net[f'level_{level}'] = PaiNNODENet(ode_args, **score_model_kwargs) 
		gen_ode_func[f'level_{level}'] = GraphODEFunc(ode_func_net=ode_func_net[f'level_{level}'], device=device).to(device)
		diffeq_solver[f'level_{level}'] = DiffeqSolver(gen_ode_func[f'level_{level}'], args.solver, args=args,odeint_rtol=1e-2, odeint_atol=1e-2, device=device)
		# decoder[f'level_{level}'] = Decoder(latent_dim, input_dim).to(device) 
		if (args.dec_aggregate == "add" or args.dec_aggregate == "single"): 
			decoder_phys[f'level_{level}'] = Decoder(latent_dim, input_dim).to(device)
		elif args.dec_aggregate == "concat": 
			decoder_phys[f'level_{level}'] = Decoder(latent_dim*(level+1), input_dim).to(device) 
		else: 
			raise Exception("Unsupported decoder aggregate method. ") 
		
		if level + 1 < args.num_levels: 
			if (args.proj_aggregate == "add" or args.proj_aggregate == "single"): 
				decoder_latent[f'level_{level}'] = Decoder(latent_dim, latent_dim).to(device) 
			elif args.proj_aggregate == "concat": 
				decoder_latent[f'level_{level}'] = Decoder(latent_dim*(level+1), latent_dim).to(device) 
			elif args.proj_aggregate == "zero": 
				decoder_latent[f'level_{level}'] = nn.Identity().to(device) 
			else: 
				raise Exception("Unsupported projector aggregate method. ") 


	'''
	ode_func_net = ParticleNetLightning(args=ode_args)


	gen_ode_func = GraphODEFunc(
		ode_func_net=ode_func_net,
		device=device).to(device)

	diffeq_solver = DiffeqSolver(gen_ode_func, args.solver, args=args,odeint_rtol=1e-2, odeint_atol=1e-2, device=device)
	'''

    #Decoder related
	# decoder = Decoder(latent_dim, input_dim).to(device)

	# encoder_z0 and decoder are lists now 
	model = LatentGraphODE(
		input_dim = input_dim,
		latent_dim = args.latents, 
		encoder_z0 = encoder_z0, 
		# decoder = decoder, 
		decoder_phys = decoder_phys, 
		decoder_latent = decoder_latent, 
		diffeq_solver = diffeq_solver, 
		z0_prior = z0_prior, 
		device = device,
		obsrv_std = obsrv_std,
		use_vae=use_vae
		).to(device)

	return model
