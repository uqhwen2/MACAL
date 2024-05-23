import torch
from causal_bald.library.datasets import IHDP

def train_test_splitting(seed, device):

    ihdp_test = IHDP(root='assets', split='test', mode='mu', seed=seed)
    ihdp_train = IHDP(root='assets', split='train', mode='mu', seed=seed)
    ihdp_valid = IHDP(root='assets', split='valid', mode='mu', seed=seed)

#    y_mean, y_std = ihdp_train.y_mean[0], ihdp_train.y_std[0]
    y_mean, y_std = 0, 1  # No normalization

    # Convert back to PyTorch tensors if needed
    combine_x_train = torch.from_numpy(ihdp_train.x)
    combine_x_test = torch.from_numpy(ihdp_test.x)
    combine_x_valid = torch.from_numpy(ihdp_valid.x)

    combined_y_train = torch.from_numpy((ihdp_train.y - y_mean)/y_std)  # No y normalization for training
    combined_y_valid = torch.from_numpy((ihdp_valid.y - y_mean)/y_std)  # No y normalization for validation

    tau_test = torch.from_numpy(ihdp_test.y)  # No y normalization for the ground truth

    T_train = torch.from_numpy(ihdp_train.t)
    T_valid = torch.from_numpy(ihdp_valid.t)
    T_test = torch.from_numpy(ihdp_test.t)

    return combine_x_train.to(device),\
           combine_x_test.to(device),\
           combined_y_train.to(device),\
           combine_x_valid.to(device),\
           combined_y_valid.to(device),\
           tau_test.to(device),\
           T_train.to(device),\
           T_valid.to(device),\
           T_test.to(device),\
           y_std
