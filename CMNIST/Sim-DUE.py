#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import csv
from torch.nn.utils import spectral_norm
import torch.nn as nn
from models.nn_model import nnModel_1, nnModel_0
from pathlib import Path
from models.nn_model import train_deep_kernel_gp, predict_deep_kernel_gp
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

import json
# Read the configuration from the file
with open('experiments/config_sim.json', 'r') as file:
    config = json.load(file)

from torch.utils.data import Dataset, DataLoader
from causal_bald.library.datasets import HCMNIST

class toDataLoader(Dataset):
    def __init__(self, x_train, y_train, t_train):
        # Generate random data for input features (x) and target variable (y)
        self.x = x_train
        self.t = t_train
        self.y = y_train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Return a single sample as a dictionary containing input features and target variable
        inputs = torch.hstack([self.x[idx], self.t[idx]]).float()
        targets = self.y[idx].float()

        return inputs, targets


class MNISTDataLoader(Dataset):
    def __init__(self, x_train, y_train):
        # Generate random data for input features (x) and target variable (y)
        self.x = x_train
        self.y = y_train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Return a single sample as a dictionary containing input features and target variable
        #inputs = torch.hstack([self.x[idx], self.t[idx]]).float()
        inputs = self.x[idx].view(1, 28, 28)
        targets = self.y[idx]

        return inputs, targets


def generation(mean_1, std_1, func_1, mean_0, std_0, func_0, device):
    # Training data is 100 points in [0,1] inclusive regularly spaced
    train_x_1 = torch.linspace(-15, 10, 200)
    # True function is sin(2*pi*x) with Gaussian noise
    train_y_1 = func_1(train_x_1 * (2 * math.pi) / math.pi)

    # Generate Gaussian noise
    noise_x_1 = torch.normal(mean=mean_1, std=std_1, size=(1000,))  

    noise_y_1 = func_1(noise_x_1 * (2 * math.pi) / math.pi)
    
    train_x_1 = torch.cat([train_x_1, noise_x_1], dim=0)
    train_y_1 = torch.cat([train_y_1, noise_y_1], dim=0)
    random_idx = np.random.permutation(len(train_x_1))

    train_x_1 = train_x_1[random_idx]
    train_y_1 = train_y_1[random_idx]

    # Training data is 100 points in [0,1] inclusive regularly spaced
    train_x_0 = torch.linspace(-10, 15, 200)
    # True function is sin(2*pi*x) with Gaussian noise
    train_y_0 = func_0(train_x_0 * (2 * math.pi) / math.pi)

    # Generate Gaussian noise
    noise_x_0 = torch.normal(mean=mean_0, std=std_0, size=(1000,)) 

    noise_y_0 = func_0(noise_x_0 * (2 * math.pi) / math.pi) 

    train_x_0 = torch.cat([train_x_0, noise_x_0], dim=0)
    train_y_0 = torch.cat([train_y_0, noise_y_0], dim=0)
    random_idx = np.random.permutation(len(train_x_0))

    train_x_0 = train_x_0[random_idx]
    train_y_0 = train_y_0[random_idx]

    combine_x = torch.cat([train_x_1, train_x_0], dim=0)
    combined_y = torch.cat([train_y_1, train_y_0], dim=0)
    combine_y_1 = func_1(combine_x * (2 * math.pi) / math.pi) 
    combine_y_0 = func_0(combine_x * (2 * math.pi) / math.pi) 
    tau = combine_y_1 - combine_y_0
    
    treated_x = torch.ones_like(train_x_1)
    control_x = torch.zeros_like(train_x_0)
    T = torch.cat([treated_x, control_x], dim=0)

    return combine_x.to(device), combined_y.to(device), tau.to(device), T.to(device)


def train_test_splitting(combine_x, combined_y, tau, T, test_size, seed, device):
    # Convert tensors to numpy arrays
    combine_x_np = combine_x.cpu().numpy()
    combined_y_np = combined_y.cpu().numpy()
    tau_np = tau.cpu().numpy()
    T_np = T.cpu().numpy()
    
    combine_x_train, combine_x_test, \
    combined_y_train, combined_y_test, \
    tau_train, tau_test, \
    T_train, T_test = train_test_split(combine_x_np, combined_y_np, tau_np, T_np, test_size=test_size, random_state=seed)
    
    ihdp_train = HCMNIST(root='assets', split='train', mode='mu', seed=seed)
    ihdp_test = HCMNIST(root='assets', split='valid', mode='mu', seed=seed)

    valid_size = 0.25
    training_idx, valid_idx = train_test_split(list(range(ihdp_train.x.shape[0])),
                                               test_size=valid_size,
                                               random_state=seed)

    # Convert back to PyTorch tensors if needed
    combine_x_train = torch.from_numpy(ihdp_train.x[training_idx])
    combine_x_valid = torch.from_numpy(ihdp_train.x[valid_idx])
    combine_x_test = torch.from_numpy(ihdp_test.x)

    combined_y_train = torch.from_numpy(ihdp_train.y[training_idx])  # No y normalization
    combined_y_valid = torch.from_numpy(ihdp_train.y[valid_idx])

    tau_test = torch.from_numpy(ihdp_test.tau)

    T_train = torch.from_numpy(ihdp_train.t[training_idx])
    T_valid = torch.from_numpy(ihdp_train.t[valid_idx])
    T_test = torch.from_numpy(ihdp_test.t)

    combined_ylabel_train = torch.from_numpy(ihdp_train.targets[training_idx])  # No y normalization
    L_test = torch.from_numpy(ihdp_test.targets)

    return combine_x_train.to(device), combine_x_test.to(device), combined_y_train.to(device), combine_x_valid.to(
        device), combined_y_valid.to(device), tau_test.to(device), T_train.to(device), T_valid.to(device), T_test.to(
        device), combined_ylabel_train.to(device), L_test.to(device)


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


def training_nn(x_1, y_1, x_0, y_0, training_iter, combine_x_valid, combined_y_valid, T_valid):
    input_dim, latent_dim, output_dim = x_1.shape[1], 200, 1

    model_1 = nnModel_1(input_dim, latent_dim, output_dim).to(device)
    model_0 = nnModel_0(input_dim, latent_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    # Use the adam optimizer
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=1e-3,
                                   weight_decay=1e-4)  # Includes GaussianLikelihood parameters
    optimizer_0 = torch.optim.Adam(model_0.parameters(), lr=1e-3,
                                   weight_decay=1e-4)  # Includes GaussianLikelihood parameters

    list_1_valid, list_0_valid = trt_ctr(T_valid)
    last_value = float('inf')
    for i in range(training_iter):
        # Find optimal model hyperparameters
        model_1.train()
        model_0.train()

        # Zero gradients from previous iteration
        optimizer_1.zero_grad()
        # Output from model
        output_1 = model_1(x_1)
        # Calc loss and backprop gradients
        loss_1 = criterion(output_1, y_1)
        loss_1.backward()
        optimizer_1.step()

        # Zero gradients from previous iteration
        optimizer_0.zero_grad()
        # Output from model
        output_0 = model_0(x_0)
        # Calc loss and backprop gradients
        loss_0 = criterion(output_0, y_0)
        loss_0.backward()
        optimizer_0.step()

        model_1.eval()
        model_0.eval()

        with torch.no_grad():
            valid_pred_1 = model_1(combine_x_valid[list_1_valid])
            valid_pred_0 = model_0(combine_x_valid[list_0_valid])
        valid_loss = torch.mean((valid_pred_1 - combined_y_valid[list_1_valid]) ** 2) + torch.mean(
            (valid_pred_0 - combined_y_valid[list_0_valid]) ** 2)

        if last_value > valid_loss:
            # Save only the model state_dict (architecture and parameters)
            torch.save({
                'model_state_dict': model_1.state_dict(),
            }, 'model_selections/sim_nnmodel_1.pth')

            torch.save({
                'model_state_dict': model_0.state_dict(),
            }, 'model_selections/sim_nnmodel_0.pth')

            last_value = valid_loss

        if i % 500 == 0:
            print('Iter %d/%d - Loss: %.3f - Valid Loss: %.3f' % (
                i + 1,
                training_iter,
                loss_0.item(),
                valid_loss))

    # Load the model
    if True:
        model_1 = nnModel_1(input_dim, latent_dim, output_dim).to(device)
        model_0 = nnModel_0(input_dim, latent_dim, output_dim).to(device)

        checkpoint_1 = torch.load('model_selections/sim_nnmodel_1.pth')
        model_1.load_state_dict(checkpoint_1['model_state_dict'])

        checkpoint_0 = torch.load('model_selections/sim_nnmodel_0.pth')
        model_0.load_state_dict(checkpoint_0['model_state_dict'])

    return model_1, model_0


def evaluation_nn(pred_1, pred_0, test_tau, query_step):

    esti_tau = torch.from_numpy(pred_1 - pred_0).float()
    pehe_test = torch.sqrt(torch.mean((esti_tau - test_tau) ** 2))

    print('\n', 'PEHE at query step: {} is {}'.format(query_step, pehe_test), '\n')

    return pehe_test


def pool_updating(idx_remaining, idx_sub_training, querying_idx):
    # Update the training and pool set for the next AL stage
    idx_sub_training = np.concatenate((idx_sub_training, querying_idx), axis=0)  # Update the training pool
    # Update the remaining pool by deleting the selected data
    mask = np.isin(idx_remaining, querying_idx, invert=True)  # Create a mask that selects the elements to delete from array1
    idx_remaining = idx_remaining[mask]  # Update the remaining pool by subtracting the selected samples
    
    return idx_sub_training, idx_remaining


def one_side_uncertainty(combine_x_train, index, num_of_samples, model):
    model.eval()
    
    pred = model(combine_x_train[index])
    pred_variance = pred.variance.sqrt()
    
    uncertainty = pred_variance
    draw_dist = uncertainty.cpu().detach().numpy()
    #quantile_threshold = np.quantile(draw_dist, 1 - percentage)  # taking top 5% of the values
    
    top_k = num_of_samples
    threshold_top_k = np.partition(draw_dist, -top_k)[-top_k]  # Calculate the threshold for the top 5 values instead of top 5%
    print('Uncertainty threshold:', threshold_top_k)

    acquired_idx = []
    for idx, i in enumerate(draw_dist):
        # print(round(i.item(),2), round(uncertainty[idx].item(),2), round(uncertainty[idx].item()/uncertainty_mean.item(),2))
        if draw_dist[idx] >= threshold_top_k:
            acquired_idx.append(idx)
            
    #print('Top 5 uncertain:', draw_dist[acquired_idx])
    acquired_idx = index[acquired_idx]
    random_idx = np.random.permutation(len(acquired_idx))
    acquired_idx = acquired_idx[random_idx]
    
    num_elements_to_select = num_of_samples  # Selecting 5 values randomly as the step size
    
    return acquired_idx[:num_elements_to_select], threshold_top_k


# Define a simple CNN for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(4 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        latent = x.view(-1, 4 * 7 * 7)
        x = self.fc1(latent)
        x = self.relu3(x)
        x = self.fc2(x)
        return x, latent


def MNISTraining(train_dataloader, test_dataloader, device):

    # Initialize the simple CNN model and move it to GPU
    model = SimpleCNN().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model on GPU
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_dataloader:
            # Convert labels to torch.long
            #labels = labels.long()
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Print training loss after each epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Evaluate the model on the test set
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {100 * accuracy:.2f}%')

    return model

# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_size = 0.1
num_trial = 1
al_step = 50
epoch = 1000
warm_up = 1000
num_of_samples = 25
seed = args.seed

if num_trial == 1:
# for seed in range(num_trial):
    
    print('Trial:', seed+1)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # Set random seed for reproducibility in DataLoader
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Set to False for reproducibility

    combine_x, combined_y, tau, T = generation(mean_1=0, std_1=1, func_1=torch.sin, mean_0=3, std_0=1, func_0=torch.cos, device=device)
    combine_x_train, combine_x_test, combined_y_train, combine_x_valid, combined_y_valid, tau_test, T_train, T_valid, T_test, combined_ylabel_train, L_test = train_test_splitting(combine_x, combined_y, tau, T, test_size, seed, device=device)

    train_dataset = MNISTDataLoader(x_train=combine_x_train.cpu(),
                                     y_train=combined_ylabel_train.cpu())

    test_dataset = MNISTDataLoader(x_train=combine_x_test.cpu(),
                                    y_train=L_test.cpu())

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    MNISTExtractor = MNISTraining(train_dataloader, test_dataloader, device)
    #raise TypeError('Checking')

    MNISTExtractor.eval()
    with torch.no_grad():
        _, combine_x_train = MNISTExtractor(combine_x_train.view(-1, 1, 28, 28))
        _, combine_x_valid = MNISTExtractor(combine_x_valid.view(-1, 1, 28, 28))
        _, combine_x_test = MNISTExtractor(combine_x_test.view(-1, 1, 28, 28))
    idx_pool = np.random.permutation(len(combine_x_train))  # Global dataset index
    idx_sub_training = idx_pool[:warm_up]  # Global dataset index
    idx_remaining = idx_pool[warm_up:]  # Global dataset index
    
    sub_training_1, sub_training_0 = trt_ctr(T_train[idx_sub_training])
    remaining_1, remaining_0 = trt_ctr(T_train[idx_remaining])
    
    # Initialize the data-limited starting size as 20% of whole treated training set
    idx_sub_training_1 = idx_sub_training[sub_training_1]  # 20% as initial
    idx_remaining_1 = idx_remaining[remaining_1]  # 20% left for querying
    
    # Initialize the data-limited starting size as 20% of whole control training set
    idx_sub_training_0 = idx_sub_training[sub_training_0]  # 10% as initial
    idx_remaining_0 = idx_remaining[remaining_0]  # 90% left for querying
    
    acquired_treated, acquired_control = None, None
    error_list = []
    num_of_acquire = [len(idx_sub_training_1) + len(idx_sub_training_0)]
    
    for query_step in range(al_step):
        
        train_x_1, train_y_1 = combine_x_train[idx_sub_training_1], combined_y_train[idx_sub_training_1]
        train_x_0, train_y_0 = combine_x_train[idx_sub_training_0], combined_y_train[idx_sub_training_0]
        print("Number of data used for training in treated and control:", len(idx_sub_training_1), len(idx_sub_training_0))

        # Concatenate vertically and shuffle randomly for the sub training
        combine_train_idx = np.concatenate([idx_sub_training_1, idx_sub_training_0], axis=0)
        np.random.shuffle(combine_train_idx)
        train_dataset = toDataLoader(x_train=combine_x_train[combine_train_idx].cpu(),
                                     y_train=combined_y_train[combine_train_idx].cpu(),
                                     t_train=T_train[combine_train_idx].cpu())

        tune_dataset = toDataLoader(x_train=combine_x_valid.cpu(),
                                    y_train=combined_y_valid.cpu(),
                                    t_train=T_valid.cpu())

        job_dir_path = Path('experiments/method_{}/seed_{}/step_{}'.format(config.get("acquisition_function"), seed, query_step))

        train_deep_kernel_gp(ds_train=train_dataset,
                             ds_valid=tune_dataset,
                             job_dir=job_dir_path,
                             config=config,
                             dim_input=combine_x_train.shape[1],  # list input for the Conv Resnet
                             seed=seed)

        test_dataset = toDataLoader(x_train=combine_x_test.cpu(),
                                    y_train=tau_test.cpu(),
                                    t_train=T_test.cpu())

        (mu_0, mu_1), _ = predict_deep_kernel_gp(dataset=test_dataset,
                                                 job_dir=job_dir_path,
                                                 config=config,
                                                 dim_input=combine_x_train.shape[1],  # list input for the Conv Resnet
                                                 seed=seed
                                                )

        pehe_error = evaluation_nn(pred_1=mu_1,
                                   pred_0=mu_0,
                                   test_tau=tau_test.cpu(),
                                   query_step=query_step
                                   )

        error_list.append(np.round(pehe_error.cpu().numpy(), 4))

        acquired_treated, acquired_control = [], []
        while len(acquired_treated) < num_of_samples:        
            #cov_matrix_x = torch.cat([combine_x_train[idx_remaining_0], combine_x_train[idx_remaining_1]])
            # Calculate the pairwise squared Euclidean distances
            sim_matrix = torch.cdist(combine_x_train[idx_remaining_0], combine_x_train[idx_remaining_1], p=2).pow(2)
            # Calculatee RBF kernel values
#            rbf_kernel_values = torch.exp(-1 * distances_sq)
            #rbf_kernel_values = distances_sq
            #sim_matrix = distances_sq[:len(idx_remaining_0), len(idx_remaining_0):]

            #cov_matrix_idx = np.concatenate((idx_sub_training_1, idx_remaining_1))
            #cov_matrix_x = combine_x_train[cov_matrix_idx]
            # Calculate the pairwise squared Euclidean distances
            part_cov_matrix = torch.cdist(combine_x_train[idx_sub_training_1], combine_x_train[idx_remaining_1], p=2).pow(2)
            # Calculate the RBF kernel values
#            rbf_kernel_values = 1 - torch.exp(-1 * distances_sq)
            #rbf_kernel_values = distances_sq
            #part_cov_matrix = distances_sq[:len(idx_sub_training_1), len(idx_sub_training_1):]
            # Find the minimum value in each column direction
            max_values_remaining_1, _ = torch.min(part_cov_matrix, dim=0)

            #cov_matrix_idx = np.concatenate((idx_sub_training_0, idx_remaining_0))
            #cov_matrix_x = combine_x_train[cov_matrix_idx]
            # Calculate the pairwise squared Euclidean distances
            part_cov_matrix = torch.cdist(combine_x_train[idx_sub_training_0], combine_x_train[idx_remaining_0], p=2).pow(2)
            # Calculate the RBF kernel values
#            rbf_kernel_values = 1 - torch.exp(-1 * distances_sq)
            #rbf_kernel_values = distances_sq
            #part_cov_matrix = distances_sq[:len(idx_sub_training_0), len(idx_sub_training_0):]
            # Find the minimum value in each column direction
            max_values_remaining_0, _ = torch.min(part_cov_matrix, dim=0)

            diversity_matrix = max_values_remaining_0.view(-1, 1) + max_values_remaining_1
            criterion = - sim_matrix + diversity_matrix
#            criterion = sim_matrix * diversity_matrix
#            criterion = sim_matrix / (torch.exp(-1 * diversity_matrix) + 1e-300)

            # Find the largest value and its indices
            max_value, _ = torch.max(criterion.view(-1), dim=0)
            row_index, col_index = torch.where(criterion==max_value)

            acquired_control.append(idx_remaining_0[row_index.item()])
            acquired_treated.append(idx_remaining_1[col_index.item()])
            
            idx_sub_training_1, idx_remaining_1 = pool_updating(idx_remaining_1, idx_sub_training_1, [idx_remaining_1[col_index.item()]])
            idx_sub_training_0, idx_remaining_0 = pool_updating(idx_remaining_0, idx_sub_training_0, [idx_remaining_0[row_index.item()]])


        if len(acquired_treated) == 0 and len(acquired_control) == 0:
            raise  TypeError("Nothing acquired by AL")
        else:
            print('Acquiring the treated and control:', len(acquired_treated), len(acquired_control))
            if query_step != 0:
                num_of_acquire.append(num_of_acquire[query_step - 1] + len(acquired_treated) + len(acquired_control))
        
#        if len(acquired_treated) != 0:
#            idx_sub_training_1, idx_remaining_1 = pool_updating(idx_remaining_1, idx_sub_training_1, acquired_treated)
#            
#        idx_sub_training_0, idx_remaining_0 = pool_updating(idx_remaining_0, idx_sub_training_0, acquired_control)

    average_pehe = np.array(error_list)

# Specify the file path
file_path = 'text_results/truesim/pehe_truesim_{}.csv'.format(args.seed)

# Open the CSV file in write mode
with open(file_path, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    # Write the data for list_1
    csv_writer.writerow(['Number of Samples'] + list(map(str, num_of_acquire)))

    # Write the data for list_2
    csv_writer.writerow(['PEHE'] + list(map(str, average_pehe.tolist())))

print(f'The data has been successfully written to {file_path}')
