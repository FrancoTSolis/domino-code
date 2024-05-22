import warnings

import torch
from torch_scatter import scatter
import torch.nn as nn 

import lib.utils as utils 
from lib.gnn_models import TemporalEncoding 

import lib.embedding as embedding 

from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

from types import SimpleNamespace 

from lib.protein_data import get_ala2_atom_numbers, get_lj_atom_numbers, get_tip3p_atom_numbers, get_tip4p_atom_numbers 



class PaiNNEncoder(torch.nn.Module):
    def __init__(
        self,
        args, 
        # n_features=32,
        n_layers=2,
        max_lag=1000,
        diff_steps=1000,
        n_neighbors=100,
        # n_types=167,
        dist_encoding="positional_encoding",
        num_of_atoms=22, 
        use_vae=False, 
    ):
        super().__init__()

        n_features = args.out_feats 
        
        self.data = args.data 

        if self.data == "ala2": 
            self.atom_numbers = get_ala2_atom_numbers(distinguish=(not args.indistinguishable))  
        elif self.data == "lj": 
            self.atom_numbers = get_lj_atom_numbers(distinguish=(not args.indistinguishable)) 
        elif self.data == "tip3p": 
            self.atom_numbers = get_tip3p_atom_numbers(distinguish=(not args.indistinguishable)) 
        elif self.data == "tip4p": 
            self.atom_numbers = get_tip4p_atom_numbers(distinguish=(not args.indistinguishable)) 
        self.num_of_atoms = num_of_atoms 

        # n_type is the maximum value in self.atom_numbers 
        n_types = max(self.atom_numbers) + 1 
        if self.data == "ala2": 
            n_types = 167 

        if self.data in ["lj", "tip3p", "tip4p"]: 
            self.embed = torch.nn.Sequential(
                embedding.AddEdges(n_neighbors=n_neighbors, cutoff=args.cutoff), 
                embedding.AddEquivariantFeatures(n_features),
                embedding.NominalEmbedding("atom_number", n_features, n_types=n_types),
                embedding.PositionalEmbedding("t_phys", n_features, max_lag),
                embedding.AddInvariantFeatures("loc_vel"), 
                embedding.CombineInvariantFeatures(2 * n_features + 6, n_features),
                PaiNNBase(
                    n_features=n_features,
                    n_features_out=n_features,
                    n_layers=n_layers,
                    dist_encoding=dist_encoding,
                ),
            )
            
        else: 
            self.embed = torch.nn.Sequential(
                embedding.AddEdges(n_neighbors=n_neighbors, cutoff=args.cutoff),
                embedding.AddEquivariantFeatures(n_features),
                embedding.NominalEmbedding("atom_number", n_features, n_types=n_types),
                embedding.PositionalEmbedding("t_phys", n_features, max_lag),
                embedding.CombineInvariantFeatures(2 * n_features, n_features),
                PaiNNBase(
                    n_features=n_features,
                    n_features_out=n_features,
                    n_layers=n_layers,
                    dist_encoding=dist_encoding,
                ),
            )

        self.sequence_w = nn.Linear(args.out_feats, args.out_feats) # for encoder

        self.aggregate = args.aggregate
        self.use_vae = use_vae 
        if self.use_vae:
            self.out_w_encoder = nn.Linear(args.out_feats, args.final_dim * 2)
        else: 
            self.out_w_encoder = nn.Linear(args.out_feats, args.final_dim) 

        utils.init_network_weights(self.out_w_encoder)

        # Attention module 
        self.temporal_net = TemporalEncoding(args.out_feats)
        self.w_transfer = nn.Linear(args.out_feats + 1, args.out_feats, bias=True)
        utils.init_network_weights(self.w_transfer)


    def rewrite_batch(self, batch, batch_y):
        assert (torch.sum(batch_y).item() == list(batch.size())[0])
        batch_new = torch.zeros_like(batch)
        group_num = 0
        current_index = 0
        for ball_time in batch_y:
            batch_new[current_index:current_index + ball_time] = group_num
            group_num += 1
            current_index += ball_time.item()

        return batch_new

    def attention_expand(self,attention_ball, batch,batch_y):
        '''

        :param attention_ball: [num_ball, d]
        :param batch: [num_nodes,d]
        :param batch_new: [num_ball,d]
        :return:
        '''
        node_size = batch.size()[0]
        dim = attention_ball.size()[1]
        new_attention = torch.ones(node_size, dim)
        if attention_ball.device != torch.device("cpu"):
            new_attention = new_attention.cuda()

        group_num = 0
        current_index = 0
        for ball_time in batch_y:
            new_attention[current_index:current_index+ball_time] = attention_ball[group_num]
            group_num +=1
            current_index += ball_time.item()

        return new_attention

    def split_mean_mu(self,h):
        last_dim = h.size()[-1] //2
        res = h[:,:last_dim], h[:,last_dim:]
        return res

    # def forward(self, batch_0, x_time=None, batch=None, is_train=False): 
    def forward(self, batch_0): 
        # print("beware of the difference between x and x_coord.") 
        # print("beware, self.embed might be expecting a different meaning for batch_0.batch.")
        # TODO: inspect and get the form of batch that embedded wants.
        # TODO: add t_phys 

        batch_embedded = SimpleNamespace(
            x=batch_0.x_coord.reshape((batch_0.num_graphs, self.num_of_atoms, -1, 3)).permute(0, 2, 1, 3).reshape((-1, 3)), 
            atom_number=batch_0.atom_number.reshape((batch_0.num_graphs, self.num_of_atoms, -1)).permute(0, 2, 1).flatten().to(batch_0.x.device),   
            batch=torch.repeat_interleave(torch.arange(batch_0.x.shape[0]/self.num_of_atoms, dtype=torch.int64), self.num_of_atoms).to(batch_0.x.device), 
            # should it be normalized this way? Have anything to do with max_lag? 
            t_phys=batch_0.pos.reshape((batch_0.num_graphs, self.num_of_atoms, -1)).permute(0, 2, 1).flatten(), 
        )

        if self.data in ["lj", "tip3p", "tip4p"]: 
            batch_embedded.loc_vel = batch_0.x.reshape((batch_0.num_graphs, self.num_of_atoms, -1, 6)).permute(0, 2, 1, 3).reshape((-1, 6)) # [batch_size * seq_length, num_atoms, 6]

        assert batch_embedded.x.size(0) == batch_embedded.batch.numel()
        embedded = self.embed(batch_embedded)  
        invariant_embedding = embedded.invariant_node_features 
        # print("TODO: reshape and swap axis to appropriate form. ") 

        # batch_y = torch.bincount(batch_0.batch) 
        # print("to be verified. ")

        batch, batch_y, x_time = batch_0.batch, batch_0.y, batch_0.pos  
        # batch_y = batch_y / num_of_atoms # to be verified 
        batch_new = self.rewrite_batch(batch, batch_y) 
        seq_length = int(batch_0.x.shape[0] / self.num_of_atoms / batch_0.num_graphs) 

        encoding_size = invariant_embedding.shape[-1] 
        h_t = invariant_embedding.reshape((batch_0.num_graphs, seq_length, self.num_of_atoms, encoding_size)).permute(0, 2, 1, 3).reshape((-1, encoding_size)) # [batch_size * num_atoms * seq_length, encoding_size=64]

        # For encoder 
        if self.aggregate == "add":
            h_ball = global_mean_pool(invariant_embedding, batch_new) #[num_ball,d], without activation
        elif self.aggregate == "attention":
            #h_t = F.gelu(self.w_transfer(torch.cat((h_t, edges_temporal), dim=1))) + edges_temporal
            x_time = x_time.view(-1,1)
            h_t = F.gelu(self.w_transfer(torch.cat((h_t, x_time), dim=1))) + self.temporal_net(x_time)
            attention_vector = F.relu(self.sequence_w(global_mean_pool(h_t,batch_new))) #[num_ball,d] ,graph vector with activation Relu
            attention_vector_expanded = self.attention_expand(attention_vector,batch,batch_y) #[num_nodes,d]
            attention_nodes = torch.sigmoid(torch.squeeze(torch.bmm(torch.unsqueeze(attention_vector_expanded,1),torch.unsqueeze(h_t,2)))).view(-1,1) #[num_nodes]
            nodes_attention = attention_nodes * h_t #[num_nodes,d]
            h_ball = global_mean_pool(nodes_attention,batch_new) #[num_ball,d] without activation

            h_out = self.out_w_encoder(h_ball) #[num_ball,2d/d]
        
        x_coord_latest_observed = batch_0.x_coord.reshape((batch_0.num_graphs, self.num_of_atoms, -1, 3))[:,:,-1,:] # [batch_size, num_atoms, 3]
        
        if self.use_vae: 
            mean,mu = self.split_mean_mu(h_out)
            mu = mu.abs()
            return mean, mu, x_coord_latest_observed 
        else: 
            return h_out, x_coord_latest_observed 
             

class PaiNNODENet(torch.nn.Module):
    def __init__(
        self,
        args, 
        # n_features=32,
        n_layers=2,
        max_lag=1000,
        diff_steps=1000,
        n_neighbors=100,
        # n_types=167,
        dist_encoding="positional_encoding",
        num_of_atoms=22, 
        # use_vae=False, 
    ):
        super().__init__()

        self.ode_input_dim = args.in_feats 
        n_features = args.out_feats 

        self.data = args.data 

        if self.data == "ala2": 
            self.atom_numbers = get_ala2_atom_numbers(distinguish=(not args.indistinguishable))  
        elif self.data == "lj": 
            self.atom_numbers = get_lj_atom_numbers(distinguish=(not args.indistinguishable)) 
        elif self.data == "tip3p": 
            self.atom_numbers = get_tip3p_atom_numbers(distinguish=(not args.indistinguishable)) 
        elif self.data == "tip4p": 
            self.atom_numbers = get_tip4p_atom_numbers(distinguish=(not args.indistinguishable)) 
        self.num_of_atoms = num_of_atoms 

        # n_type is the maximum value in self.atom_numbers 
        n_types = max(self.atom_numbers) + 1 
        if self.data == "ala2": 
            n_types = 167 


        self.net = torch.nn.Sequential(
            embedding.AddEdges(n_neighbors=n_neighbors, cutoff=args.cutoff), 
            embedding.AddEquivariantFeatures(n_features), 
            embedding.NominalEmbedding("atom_number", self.ode_input_dim, n_types=n_types),
            # embedding.PositionalEmbedding("t_phys", self.ode_input_dim, max_lag),
            embedding.AddInvariantFeatures("latent_encoding"), # ode_input_dim  
            embedding.CombineInvariantFeatures(2 * self.ode_input_dim, n_features),
            PaiNNBase(
                n_features=n_features,
                n_features_out=n_features,
                n_layers=n_layers,
                dist_encoding=dist_encoding,
            ),
        )


        self.out_w_ode = nn.Linear(args.out_feats, args.final_dim)

        self.x_coord_latest_observed = None 
    
    def forward(self, batch_0): 
        assert (batch_0.shape[0] == 1) # assuming single batch processing 
        assert (self.x_coord_latest_observed.shape[0] == 1) # assuming single batch processing
        # print("to be certified. ")
        batch_embedded = SimpleNamespace(
            x=self.x_coord_latest_observed.reshape((-1, 3)), 
            atom_number=self.atom_numbers.to(self.x_coord_latest_observed.device),    
            batch=torch.repeat_interleave(torch.arange(batch_0.shape[0], dtype=torch.int64), self.num_of_atoms).to(self.x_coord_latest_observed.device), 
            latent_encoding=batch_0.reshape((-1, self.ode_input_dim)),     
            # should it be normalized this way? Have anything to do with max_lag? 
            # t_phys=batch_0.pos.reshape((batch_0.num_graphs, self.num_of_atoms, -1)).permute(0, 2, 1).flatten(), 
        )

        # make sure this lateset observed coordinate will not be re-used in the next iteration 
        # self.x_coord_latest_observed = None 

        assert batch_embedded.x.size(0) == batch_embedded.batch.numel()
        embedded = self.net(batch_embedded)  
        invariant_embedding = embedded.invariant_node_features 
        # print("TODO: reshape and swap axis to appropriate form. ") 

        # batch_y = torch.bincount(batch_0.batch) 
        # print("to be verified. ")

        # batch, batch_y, x_time = batch_0.batch, batch_0.y, batch_0.pos  
        # batch_y = batch_y / num_of_atoms # to be verified 
        # batch_new = self.rewrite_batch(batch, batch_y) 
        # seq_length = int(batch_0.x.shape[0] / self.num_of_atoms / batch_0.num_graphs) 

        # encoding_size = invariant_embedding.shape[-1] 
        # h_t = invariant_embedding.reshape((batch_0.num_graphs, seq_length, self.num_of_atoms, encoding_size)).permute(0, 2, 1, 3).reshape((-1, encoding_size)) # [batch_size * num_atoms * seq_length, encoding_size=64]

        # for decoder 
        h_out = self.out_w_ode(invariant_embedding) 
        h_out = h_out.reshape((-1, self.num_of_atoms, h_out.shape[-1]))
        # Can be accessed multiple times in ode after one initialization 
        # nfe is the counter. 
        # self.edge_idx_latest_observed = None 
        # self.x_coord_latest_observed = None 

        return h_out


class PaiNNTLScore(torch.nn.Module):
    def __init__(
        self, 
        n_features=32,
        n_layers=2,
        max_lag=1000,
        diff_steps=1000,
        n_neighbors=100,
        n_types=167,
        dist_encoding="positional_encoding",
    ):
        assert False, "Deprecated. "
        super().__init__()
        self.embed = torch.nn.Sequential(
            embedding.AddEdges(n_neighbors=n_neighbors),
            embedding.AddEquivariantFeatures(n_features),
            embedding.NominalEmbedding("atom_number", n_features, n_types=n_types),
            embedding.PositionalEmbedding("t_phys", n_features, max_lag),
            embedding.CombineInvariantFeatures(2 * n_features, n_features),
            PaiNNBase(
                n_features=n_features,
                n_features_out=n_features,
                n_layers=n_layers,
                dist_encoding=dist_encoding,
            ),
        )

        self.net = torch.nn.Sequential(
            embedding.AddEdges(should_generate_edge_index=False),
            embedding.PositionalEmbedding("t_diff", n_features, diff_steps),
            embedding.CombineInvariantFeatures(2 * n_features, n_features),
            PaiNNBase(n_features=n_features, dist_encoding=dist_encoding),
        )

    def forward(self, noise_batch, batch_0):
        batch_0 = batch_0.clone()
        noise_batch = noise_batch.clone()

        embedded = self.embed(batch_0)
        cond_inv_features = embedded.invariant_node_features
        cond_eqv_features = embedded.equivariant_node_features
        cond_edge_index = embedded.edge_index

        noise_batch.invariant_node_features = cond_inv_features
        noise_batch.equivariant_node_features = cond_eqv_features
        noise_batch.edge_index = cond_edge_index

        dx = self.net(noise_batch).equivariant_node_features.squeeze()
        noise_batch.x = noise_batch.x + dx

        return noise_batch


class PaiNNBase(torch.nn.Module):
    def __init__(
        self,
        n_features=128,
        n_layers=5,
        n_features_out=1,
        length_scale=10,
        dist_encoding="positional_encoding",
    ):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(
                Message(
                    n_features=n_features,
                    length_scale=length_scale,
                    dist_encoding=dist_encoding,
                )
            )
            layers.append(Update(n_features))

        layers.append(Readout(n_features, n_features_out))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, batch):
        return self.layers(batch)


class Message(torch.nn.Module):
    def __init__(
        self, n_features=128, length_scale=10, dist_encoding="positional_encoding"
    ):
        super().__init__()
        self.n_features = n_features

        assert dist_encoding in (
            a := ["positional_encoding", "soft_one_hot"]
        ), f"positional_encoder must be one of {a}"

        if dist_encoding in ["positional_encoding", None]:
            self.positional_encoder = embedding.PositionalEncoder(
                n_features, length=length_scale
            )
        elif dist_encoding == "soft_one_hot":
            self.positional_encoder = embedding.SoftOneHotEncoder(
                n_features, max_radius=length_scale
            )

        self.phi = embedding.MLP(n_features, n_features, 4 * n_features)
        self.W = embedding.MLP(n_features, n_features, 4 * n_features)

    def forward(self, batch):
        src_node = batch.edge_index[0]
        dst_node = batch.edge_index[1]

        positional_encoding = self.positional_encoder(batch.edge_dist)
        gates, cross_product_gates, scale_edge_dir, scale_features = torch.split(
            self.phi(batch.invariant_node_features[src_node])
            * self.W(positional_encoding),
            self.n_features,
            dim=-1,
        )
        gated_features = multiply_first_dim(
            gates, batch.equivariant_node_features[src_node]
        )
        scaled_edge_dir = multiply_first_dim(
            scale_edge_dir, batch.edge_dir.unsqueeze(1).repeat(1, self.n_features, 1)
        )

        dst_node_edges = batch.edge_dir.unsqueeze(1).repeat(1, self.n_features, 1)
        dst_equivariant_node_features = batch.equivariant_node_features[dst_node]
        cross_produts = torch.cross(
            dst_node_edges, dst_equivariant_node_features, dim=-1
        )

        gated_cross_products = multiply_first_dim(cross_product_gates, cross_produts)

        dv = scaled_edge_dir + gated_features + gated_cross_products
        ds = multiply_first_dim(scale_features, batch.invariant_node_features[src_node])

        dv = scatter(dv, dst_node, dim=0)
        ds = scatter(ds, dst_node, dim=0)

        batch.equivariant_node_features += dv
        batch.invariant_node_features += ds

        return batch


def multiply_first_dim(w, x):
    with warnings.catch_warnings(record=True):
        return (w.T * x.T).T


class Update(torch.nn.Module):
    def __init__(self, n_features=128):
        super().__init__()
        self.U = EquivariantLinear(n_features, n_features)
        self.V = EquivariantLinear(n_features, n_features)
        self.n_features = n_features
        self.mlp = embedding.MLP(2 * n_features, n_features, 3 * n_features)

    def forward(self, batch):
        v = batch.equivariant_node_features
        s = batch.invariant_node_features

        Vv = self.V(v)
        Uv = self.U(v)

        Vv_norm = Vv.norm(dim=-1)
        Vv_squared_norm = Vv_norm**2

        mlp_in = torch.cat([Vv_norm, s], dim=-1)

        gates, scale_squared_norm, add_invariant_features = torch.split(
            self.mlp(mlp_in), self.n_features, dim=-1
        )

        delta_v = multiply_first_dim(Uv, gates)
        delta_s = Vv_squared_norm * scale_squared_norm + add_invariant_features

        batch.invariant_node_features = batch.invariant_node_features + delta_s
        batch.equivariant_node_features = batch.equivariant_node_features + delta_v

        return batch


class EquivariantLinear(torch.nn.Module):
    def __init__(self, n_features_in, n_features_out):
        super().__init__()
        self.linear = torch.nn.Linear(n_features_in, n_features_out, bias=False)

    def forward(self, x):
        # torch.swapaxes is actually an alias for torch.transpose as of PyTorch 1.8.0 (current master branch) 
        return self.linear(x.transpose(-1, -2)).transpose(-1, -2)


class Readout(torch.nn.Module):
    def __init__(self, n_features=128, n_features_out=13):
        super().__init__()
        self.mlp = embedding.MLP(n_features, n_features, 2 * n_features_out)
        self.V = EquivariantLinear(n_features, n_features_out)
        self.n_features_out = n_features_out

    def forward(self, batch):
        invariant_node_features_out, gates = torch.split(
            self.mlp(batch.invariant_node_features), self.n_features_out, dim=-1
        )

        equivariant_node_features = self.V(batch.equivariant_node_features)
        equivariant_node_features_out = multiply_first_dim(
            equivariant_node_features, gates
        )

        batch.invariant_node_features = invariant_node_features_out
        batch.equivariant_node_features = equivariant_node_features_out
        return batch
