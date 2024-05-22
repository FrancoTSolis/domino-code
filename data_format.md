# Data Format for MD datasets 

```
<dataset>: lj, tip3p, tip4p 
```

## Data Preparation: `data/generate_<dataset>_data.py` 

Environment setup: 

```
conda install -c conda-forge openmm=7.7.0
conda install -c conda-forge openmmtools=0.21.0
```

File structure: (use `lj` dataset as an example) 

```
data/lj_data_10000 
├── force_test_lj.npy
├── force_train_lj.npy
├── loc_test_lj.npy
├── loc_train_lj.npy
├── times_test_lj.npy
├── times_train_lj.npy
├── vel_test_lj.npy
└── vel_train_lj.npy
```

where `force`, `loc`, and `vel` corresponds to the force, location, velocity of each atom at each time point, the raw data are of the same format. Each `.npy` mentioned above is a tensor of shape (`[n_data, n_atoms, seq_len, n_dim]`) (e.g. `[3, 258, 10000, 3]`), where: 

- `n_data`: pieces of training/testing data; 
- `n_atoms`: number of atoms in the system; 
- `seq_len`: sequence length; 
- `n_dim`: number of dimensions (for 3D space, the number is 3).  

## Data splitting: `lib/new_dataloader.py` 

Suppose the entire sequence length we are interested is `entire_num`, for a piece of training/testing data with shape [n_atoms, seq_len, n_dim], we split it into: (we now start looking from the front) (e.g.: `observed_scope=2000, extrap_scope=8000`) 

- The part observed: `data[:,:observed_scope,:]`; 
- The part from which to predict: `data[:,observed_scope:observed_scope+extrap_scope,:]`; 

## Evaluation metrics: `lib/base_model.py` 

Evaluation is performed on the predicted part of the sequence: 

- MSE (`(pred_y - real_y)**2`) 
- **RMSE (square root of MSE)** 
- MAE (`abs(pred_y - real_y)`) 
- MAPE (`abs((pred_y - real_y) / real_y)`) 
- **RMSE with PCA analysis** 

## Updates on tip3p and tip4p 

`NUM_ATOMS` for tip3p is `3*258`, for tip4p is `3*251`. 

The `3i`-th atoms are oxygen atoms, and the `3i+1`-th and `3i+2`-th atoms are the hydrogen atoms in the same molecule. 

There are chemical bonds between the oxygen atom and the hydrogen atoms in the same molecule. We can extract the bonding information like this (`gamd_code/tip4p/train_network_tip4p.py`): 

```
def create_water_bond(total_atom_num):
    bond = []
    for i in range(0, total_atom_num, 3):
        bond += [[i, i+1], [i, i+2]]
    return np.array(bond)
```

Remove the pseudo atom for the tip4p dataset: 

```
if (self.suffix == '_tip4p'): # remove the pseudo-atom 
    loc = loc[:, (np.mod(np.arange(loc.shape[1]), 4) < 3), :, :] 
    vel = vel[:, (np.mod(np.arange(vel.shape[1]), 4) < 3), :, :] 
    times = times[:, (np.mod(np.arange(times.shape[1]), 4) < 3), :] 
```


