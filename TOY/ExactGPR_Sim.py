#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import random
from utils import wasserstein
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import csv

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# We will use the simplest form of GP model, exact inference
class ExactGPModel_1(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel_1, self).__init__(train_x, train_y, likelihood)
        self.kernel = kernel
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(self.kernel)

    def forward(self, x):
        # Set the desired lengthscale
#        lengthscale_value = 0.20  # You can change this value to your desired lengthscale
#        self.kernel.lengthscale = torch.tensor(lengthscale_value)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# We will use the simplest form of GP model, exact inference
class ExactGPModel_0(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel_0, self).__init__(train_x, train_y, likelihood)
        self.kernel = kernel
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(self.kernel)

    def forward(self, x):
        # Set the desired lengthscale
#        lengthscale_value = 0.20  # You can change this value to your desired lengthscale
#        self.kernel.lengthscale = torch.tensor(lengthscale_value)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
def generation(mean_1, std_1, func_1, mean_0, std_0, func_0, device):
    # Training data is 100 points in [0,1] inclusive regularly spaced
    train_x_1 = torch.linspace(-12, 10, 100)
    # True function is sin(2*pi*x) with Gaussian noise
    train_y_1 = func_1(train_x_1 * (2 * math.pi) / math.pi)

    # Generate Gaussian noise
    noise_x_1 = torch.normal(mean=mean_1, std=std_1, size=(400,))  

    noise_y_1 = func_1(noise_x_1 * (2 * math.pi) / math.pi)
    
    train_x_1 = torch.cat([train_x_1, noise_x_1], dim=0)
    train_y_1 = torch.cat([train_y_1, noise_y_1], dim=0)
    random_idx = np.random.permutation(len(train_x_1))

    train_x_1 = train_x_1[random_idx]
    train_y_1 = train_y_1[random_idx]

    # Training data is 100 points in [0,1] inclusive regularly spaced
    train_x_0 = torch.linspace(-10, 11, 500)
    # True function is sin(2*pi*x) with Gaussian noise
    train_y_0 = func_0(train_x_0 * (2 * math.pi) / math.pi)

    # Generate Gaussian noise
    noise_x_0 = torch.normal(mean=mean_0, std=std_0, size=(2000,)) 

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
    
    # Convert back to PyTorch tensors if needed
    combine_x_train = torch.from_numpy(combine_x_train)
    combine_x_test = torch.from_numpy(combine_x_test)
    combined_y_train = torch.from_numpy(combined_y_train)
    combined_y_test = torch.from_numpy(combined_y_test)
    tau_train = torch.from_numpy(tau_train)
    tau_test = torch.from_numpy(tau_test)
    T_train = torch.from_numpy(T_train)
    T_test = torch.from_numpy(T_test)
    
    return combine_x_train.to(device), combine_x_test.to(device), combined_y_train.to(device), combined_y_test.to(device), tau_train.to(device), tau_test.to(device), T_train.to(device), T_test.to(device)


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


def training(x_1, y_1, x_0, y_0, training_iter):
    kernel_1 = gpytorch.kernels.RBFKernel().to(device)
    kernel_0 = gpytorch.kernels.RBFKernel().to(device)
    
    likelihood_1 = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood_0 = gpytorch.likelihoods.GaussianLikelihood().to(device)
    
    model_1 = ExactGPModel_1(x_1, y_1, likelihood_1, kernel_1).to(device)
    model_0 = ExactGPModel_0(x_0, y_0, likelihood_0, kernel_0).to(device)
    
    # Find optimal model hyperparameters
    model_1.train()
    likelihood_1.train()
    model_0.train()
    likelihood_0.train()
    
    # Use the adam optimizer
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters
    optimizer_0 = torch.optim.Adam(model_0.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters
    
    # "Loss" for GPs - the marginal log likelihood
    mll_1 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_1, model_1)
    mll_0 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_0, model_0)
    
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer_1.zero_grad()
        # Output from model
        output_1 = model_1(x_1)
        # Calc loss and backprop gradients
        loss_1 = -mll_1(output_1, y_1)
        loss_1.backward()
        optimizer_1.step()
        
        # Zero gradients from previous iteration
        optimizer_0.zero_grad()
        # Output from model
        output_0 = model_0(x_0)
        # Calc loss and backprop gradients
        loss_0 = -mll_0(output_0, y_0)
        loss_0.backward()
        optimizer_0.step()
        
        if i % 100 == 0:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss_0.item(),
            model_0.covar_module.base_kernel.lengthscale.item(),
            model_0.likelihood.noise.item()))
    
    return model_1, model_0, likelihood_1, likelihood_0


def evaluation(test_x, test_tau, query_step):
    model_1.eval()
    likelihood_1.eval()
    model_0.eval()
    likelihood_0.eval()
    
    #mse_loss = torch.nn.MSELoss
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred_1 = likelihood_1(model_1(test_x))
        observed_pred_0 = likelihood_0(model_0(test_x))
        
    esti_tau = observed_pred_1.mean - observed_pred_0.mean
    pehe_test = torch.sqrt(torch.mean((esti_tau - test_tau) ** 2))
    
    print('\n', 'PEHE at query step: {} is {}'.format(query_step, pehe_test), '\n')
    
    return pehe_test


def plot(combine_x, combined_y, model_1, model_0, likelihood_1, likelihood_0, x_1, x_0, y_1, y_0, acquired_treated, acquired_control):
    # Get into evaluation (predictive posterior) mode
    model_1.eval()
    likelihood_1.eval()
    model_0.eval()
    likelihood_0.eval()
    
    model_1, model_0, likelihood_1, likelihood_0 = model_1.cpu(), model_0.cpu(), likelihood_1.cpu(), likelihood_0.cpu()
    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(-15, 15, 1000)
        observed_pred_1 = likelihood_1(model_1(test_x))
        observed_pred_0 = likelihood_0(model_0(test_x))
        
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(16, 6))
        
        ax.scatter(combine_x[list_1].cpu(), torch.zeros_like(combine_x[list_1]).cpu() + 2, c='r')
        ax.scatter(combine_x[list_0].cpu(), torch.zeros_like(combine_x[list_0]).cpu() - 2, c='b')
        ax.scatter(combine_x[list_1][acquired_treated].cpu(), combined_y[list_1][acquired_treated].cpu(), c='r', marker='*', s=200)
        ax.scatter(combine_x[list_0][acquired_control].cpu(), combined_y[list_0][acquired_control].cpu(), c='b', marker='*', s=200)
        
        # Get upper and lower confidence bounds
        lower_1, upper_1 = observed_pred_1.confidence_region()
        # Plot training data as black stars
        ax.scatter(x_1.cpu().numpy(), y_1.cpu().numpy(), c='r', s=20)
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred_1.mean.numpy(), 'r')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower_1.numpy(), upper_1.numpy(), alpha=0.1)
        
        # Get upper and lower confidence bounds
        lower_0, upper_0 = observed_pred_0.confidence_region()
        # Plot training data as black stars
        ax.scatter(x_0.cpu().numpy(), y_0.cpu().numpy(), c='b', s=20)
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred_0.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower_0.numpy(), upper_0.numpy(), alpha=0.1)

        ax.set_ylim([-3, 3])
        ax.legend(['Treated', 'Control', 'Acquired Treated', 'Acquired Control'])


def pool_updating(idx_remaining, idx_sub_training, querying_idx):
    # Update the training and pool set for the next AL stage
    idx_sub_training = np.concatenate((idx_sub_training, querying_idx), axis=0)  # Update the training pool
    # Update the remaining pool by deleting the selected data
    mask = np.isin(idx_remaining, querying_idx, invert=True)  # Create a mask that selects the elements to delete from array1
    idx_remaining = idx_remaining[mask]  # Update the remaining pool by subtracting the selected samples
    
    return idx_sub_training, idx_remaining


def uncertainty_calculation(combine_x, T):
    model_1.eval()
    model_0.eval()
    
    pred_1, pred_0 = model_1(combine_x), model_0(combine_x)
    pred_1_variance, pred_0_variance = pred_1.variance, pred_0.variance  # Leave out the .sqrt() to maintain the variance term.
    
    uncertainty_outputs = torch.concat([pred_1_variance.unsqueeze(-1), pred_0_variance.unsqueeze(-1)], dim=1)
    # Convert non-zero values to True (1) and zero values to False (0)
    T = T.bool()
    # Use torch.where to select values from uncertainty_outputs based on indicator_array
    uncertainty = torch.where(T.unsqueeze(-1), uncertainty_outputs[:, 0:1], uncertainty_outputs[:, 1:2])

    # Use both side of the uncertainty to calculate the total vairance, could be biased, single-sided is better.
    #uncertainty = (pred_1_variance + pred_0_variance) / 2  
    draw_dist = uncertainty.cpu().detach().numpy()
    mean_variance = np.mean(draw_dist.flatten())
    
    return mean_variance


# In[2]:


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
test_size = 0.2


# In[3]:


combine_x, combined_y, tau, T = generation(mean_1=-2.5, std_1=1, func_1=torch.sin, mean_0=2.5, std_0=1, func_0=torch.cos, device=device)
combine_x_train, combine_x_test, combined_y_train, combined_y_test, tau_train, tau_test, T_train, T_test = train_test_splitting(combine_x, combined_y, tau, T, test_size, seed, device=device)


# In[4]:


list_1, list_0 = trt_ctr(T_train)


# In[5]:


warm_up = 25
num_of_samples=10

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
variance_list = []
wasserstein_list = []
num_of_acquire = [len(idx_sub_training_1) + len(idx_sub_training_0)]


# In[6]:


for query_step in range(25):
    
    train_x_1, train_y_1 = combine_x_train[idx_sub_training_1], combined_y_train[idx_sub_training_1]
    train_x_0, train_y_0 = combine_x_train[idx_sub_training_0], combined_y_train[idx_sub_training_0]
    print("Number of data used for training in treated and control:", len(idx_sub_training_1), len(idx_sub_training_0))
    
    model_1, model_0, likelihood_1, likelihood_0 = training(train_x_1, train_y_1, train_x_0, train_y_0, 500)
    mean_variance = uncertainty_calculation(combine_x_train, T_train)
    wasserstein_distance = wasserstein(train_x_0, train_x_1)[0].detach().cpu().numpy()
    print('Mean variance at step {}:'.format(query_step), mean_variance)
    print('Wasserstein distance at step {}:'.format(query_step), wasserstein_distance)
    
    variance_list.append(np.round(mean_variance, 4))
    wasserstein_list.append(np.round(wasserstein_distance, 4))   
    # Concatenate vertically and shuffle randomly for the sub training
    combine_train_idx = np.concatenate([idx_sub_training_1, idx_sub_training_0], axis=0)
    np.random.shuffle(combine_train_idx)
            
    acquired_treated, acquired_control = [], []
    while len(acquired_treated) + len(acquired_control) < num_of_samples:

        if len(idx_remaining_1) == 0:
            part_cov_matrix = torch.cdist(combine_x_train[idx_sub_training_0].unsqueeze(1), combine_x_train[idx_remaining_0].unsqueeze(1), p=2).pow(2)
            max_values_remaining_0, _ = torch.min(part_cov_matrix, dim=0)
            max_index = torch.argmax(max_values_remaining_0)
            acquired_control.append(idx_remaining_0[max_index.item()])
            idx_sub_training_0, idx_remaining_0 = pool_updating(idx_remaining_0, idx_sub_training_0, [idx_remaining_0[max_index.item()]])

        else:
            sim_matrix = torch.cdist(combine_x_train[idx_remaining_0].unsqueeze(1), combine_x_train[idx_remaining_1].unsqueeze(1), p=2).pow(2)
            #print(sim_matrix)
            
            if len(idx_sub_training_1) == 0:
                max_values_remaining_1 = torch.zeros_like(combined_y_train[idx_remaining_1])
            else:
                part_cov_matrix = torch.cdist(combine_x_train[idx_sub_training_1].unsqueeze(1), combine_x_train[idx_remaining_1].unsqueeze(1), p=2).pow(2)
                max_values_remaining_1, _ = torch.min(part_cov_matrix, dim=0)

            part_cov_matrix = torch.cdist(combine_x_train[idx_sub_training_0].unsqueeze(1), combine_x_train[idx_remaining_0].unsqueeze(1), p=2).pow(2)
            max_values_remaining_0, _ = torch.min(part_cov_matrix, dim=0)

            diversity_matrix = max_values_remaining_0.view(-1, 1) + max_values_remaining_1
            criterion = - 1 * sim_matrix + 1 * diversity_matrix

            max_value, _ = torch.max(criterion.view(-1), dim=0)
            row_index, col_index = torch.where(criterion==max_value)
            
            #print(max_value)
            #print(combine_x[list_0][idx_remaining_0[row_index[0].item()]], combine_x[list_1][idx_remaining_1[col_index[0].item()]])
            
            acquired_control.append(idx_remaining_0[row_index[0].item()])
            acquired_treated.append(idx_remaining_1[col_index[0].item()])
            
            if len(acquired_treated) != 0:
                idx_sub_training_1, idx_remaining_1 = pool_updating(idx_remaining_1, idx_sub_training_1, [idx_remaining_1[col_index[0].item()]])
            if len(acquired_control) != 0:
                idx_sub_training_0, idx_remaining_0 = pool_updating(idx_remaining_0, idx_sub_training_0, [idx_remaining_0[row_index[0].item()]])

    if len(acquired_treated) == 0 and len(acquired_control) == 0:
        raise  TypeError("Nothing acquired by AL")
    else:
        pehe_error = evaluation(test_x=combine_x_test, 
                                test_tau=tau_test, 
                                query_step=query_step
                                )
        error_list.append(np.round(pehe_error.cpu().numpy(), 4))
        
        if query_step != 0:
            num_of_acquire.append(num_of_acquire[query_step - 1] + len(acquired_treated) + len(acquired_control))
        #plot(combine_x_train, combined_y_train, model_1, model_0, likelihood_1, likelihood_0, train_x_1, train_x_0, train_y_1, train_y_0, acquired_treated, acquired_control)
        
# Specify the file path
file_path = 'text_results/truesim/pehe_truesim_{}.csv'.format(args.seed)


# Open the CSV file in write mode
with open(file_path, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    # Write the data for list_1
    csv_writer.writerow(['Number of Samples'] + list(map(str, num_of_acquire)))

    # Write the data for list_2
    csv_writer.writerow(['PEHE'] + list(map(str, error_list)))
    # Write the data for list_2
    csv_writer.writerow(['Var'] + list(map(str, variance_list)))
    # Write the data for list_2
    csv_writer.writerow(['Wass'] + list(map(str, wasserstein_list)))
    
print(f'The data has been successfully written to {file_path}')


# In[ ]:




