import numpy as np

def moving_average(data, window_size):
    """
    Calculate the moving average of the given data.

    :param data: Array of data points.
    :param window_size: Size of the moving average window.
    :return: Array of moving averages.
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1")

    # return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    filter = np.ones(window_size) / window_size 
    return np.apply_along_axis(lambda m: np.convolve(m, filter, mode='valid'), axis=-2, arr=data) 

def decompose_trajectory(trajectory, n_layers, window_sizes):
    """
    Decompose the trajectory into multiple layers of moving averages and residuals.

    :param trajectory: Original trajectory data s.
    :param n_layers: Number of layers in the hierarchy.
    :param window_sizes: List of window sizes for moving averages at each layer.
    :return: Decomposed trajectory data.
    """
    if n_layers - 1 != len(window_sizes):
        raise ValueError("Number of layers and number of window sizes must match")
    
    if (len(trajectory.shape) != 4): 
        print("padding is hard wired for 4 dimensions! ")

    layers = []
    residual = trajectory

    for i in range(n_layers - 1):
        s = residual
        s_ma = moving_average(s, window_sizes[i])

        # Pad the moving average arrays to match the original trajectory length
        # pad_size = len(s) - len(s_ma)
        pad_size = s.shape[-2] - s_ma.shape[-2] 
        # s_ma_padded = np.pad(s_ma, (pad_size, 0), 'constant', constant_values=(0, 0))
        s_ma_padded = np.pad(s_ma, ((0, 0), (0, 0), (pad_size, 0), (0, 0)), 'constant', constant_values=(0, 0))


        layers.append(s_ma_padded)
        residual = s - s_ma_padded
    
    layers.append(residual) # The final layer is the residual itself 

    return layers


if __name__ == "__main__": 
    # Example usage
    '''
    s = np.random.random(100)  # position data
    v = np.random.random(100)  # velocity data
    trajectory = (s, v)
    '''
    np.random.seed(42) 
    trajectory = np.random.random((2, 7, 10, 3)) 
    n_layers = 3
    window_sizes = [5, 2]  # Example window sizes for each layer

    decomposed_trajectory = decompose_trajectory(trajectory, n_layers, window_sizes)
    # print(decomposed_trajectory.shape) 
    
    for x in decomposed_trajectory: 
        print(x.shape) 
        print(x)
    
    print("------------------------")
    print(trajectory.shape)
    print(trajectory) 

    print("------------------------") 
    check_sum = np.zeros_like(trajectory) 
    for x in decomposed_trajectory: 
        check_sum += x 
    
    check_sum /= trajectory 
    print(check_sum.shape)
    print(check_sum) 

    
