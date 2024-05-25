import torch
import numpy as np
from sklearn.model_selection import train_test_split


def trt_ctr(treatment):
    list1, list0 = [], []
    for index, i in enumerate(treatment):
        if i == 1:
            list1.append(index)
        elif i == 0:
            list0.append(index)
        else:
            raise TypeError('Invalid treatment value found')

    return list1, list0


def train_test_splitting(seed, device):

    ihdp_test = np.load('dataset/ibm/ibm_test.npz')
    ihdp_train = np.load('dataset/ibm/ibm_train.npz')

    # Convert back to PyTorch tensors if needed
    combine_x_train = torch.from_numpy(ihdp_train['x'][:, :, seed])

    T_train = torch.from_numpy(ihdp_train['t'][:, seed])
    list_1, _ = trt_ctr(T_train)
    treated_x_train = combine_x_train[list_1].cpu()
    # Define the range
    lower_bound = -3.00
    upper_bound = 0.00
    dist_check = treated_x_train[:, 7]
    # Find the indices where the elements fall within the specified range
    relative_indices = torch.nonzero((dist_check.cpu() >= lower_bound) & (dist_check.cpu() <= upper_bound)).squeeze()
    relative_indices = relative_indices.tolist()
    indices = np.array(list_1)[relative_indices]
    indices = torch.tensor(indices.tolist())
    # Get all indices
    all_indices = torch.arange(len(T_train))
    # Subtract the subset from the original list using list comprehension
    biased_train = [item for item in all_indices if item not in indices]

    valid_size = 0.25
    training_idx, valid_idx = train_test_split(list(range(ihdp_train['x'][biased_train][:, :, seed].shape[0])),
                                                test_size=valid_size,
                                                random_state=seed)

    combine_x_train = torch.from_numpy(ihdp_train['x'][biased_train][training_idx][:, :, seed])
    combine_x_valid = torch.from_numpy(ihdp_train['x'][biased_train][valid_idx][:, :, seed])
    combine_x_test = torch.from_numpy(ihdp_test['x'][:, :, seed])

    combined_y_train = torch.from_numpy(ihdp_train['yf'][biased_train][training_idx][:, seed])  # No y normalization
 
    y_mean, y_std = torch.mean(combined_y_train), torch.std(combined_y_train)  # Get the std before normalization

    combined_y_train = (combined_y_train - y_mean) / y_std  # Get y train normalization
#    print('MEAN AND STD of y:', torch.mean(combined_y_train), torch.std(combined_y_train))
    combined_y_valid = torch.from_numpy(ihdp_train['yf'][biased_train][valid_idx][:, seed])
    combined_y_valid = (combined_y_valid - y_mean) / y_std  # Get y valid normalization

    tau_test = torch.from_numpy(ihdp_test['mu1'][:, seed] - ihdp_test['mu0'][:, seed])  # No y normalization for ground truth

    T_train = torch.from_numpy(ihdp_train['t'][biased_train][training_idx][:, seed])
    T_valid = torch.from_numpy(ihdp_train['t'][biased_train][valid_idx][:, seed])
    T_test = torch.from_numpy(ihdp_test['t'])[:, seed]

    return combine_x_train.type(torch.float32).to(device),\
           combine_x_test.type(torch.float32).to(device),\
           combined_y_train.type(torch.float32).to(device),\
           combine_x_valid.type(torch.float32).to(device),\
           combined_y_valid.type(torch.float32).to(device),\
           tau_test.type(torch.float32).to(device),\
           T_train.to(device),\
           T_valid.to(device),\
           T_test.to(device),\
           y_std.numpy()

