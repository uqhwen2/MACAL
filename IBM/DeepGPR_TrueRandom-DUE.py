import math
import torch
import torch.nn as nn
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import csv
from torch.nn.utils import spectral_norm

from pathlib import Path
from models.nn_model import train_deep_kernel_gp, predict_deep_kernel_gp
from causal_bald.library import acquisitions
from models.nn_model import nnModel_1, nnModel_0

from sklearn.manifold import TSNE

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

import json
# Read the configuration from the file
with open('experiments/config_random.json', 'r') as file:
    config = json.load(file)

from torch.utils.data import Dataset

from models.utils import train_test_splitting

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
            }, 'model_selections/random_nnmodel_1.pth')

            torch.save({
                'model_state_dict': model_0.state_dict(),
            }, 'model_selections/random_nnmodel_0.pth')

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

        checkpoint_1 = torch.load('model_selections/random_nnmodel_1.pth')
        model_1.load_state_dict(checkpoint_1['model_state_dict'])

        checkpoint_0 = torch.load('model_selections/random_nnmodel_0.pth')
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


def random_sampling(pool, num_of_samples):

    # Use numpy.random.choice to randomly select elements
    acquired_idx = np.random.choice(pool, size=num_of_samples, replace=False)
    
    return acquired_idx


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_size = 0.1
num_trial = 1
al_step = 50
warm_up = 50
num_of_samples = 50
seed = args.seed

if num_trial == 1:
# for seed in range(num_trial):
    
    print('Trial:', seed)
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    combine_x_train,\
    combine_x_test,\
    combined_y_train,\
    combine_x_valid,\
    combined_y_valid,\
    tau_test, T_train,\
    T_valid,\
    T_test,\
    y_std = train_test_splitting(seed, device=device)

    print('Train size:', combine_x_train.shape[0], 'Test size:', combine_x_test.shape[0], 'Valid size:', combine_x_valid.shape[0])

    idx_pool = np.random.permutation(len(combine_x_train))
    idx_sub_training = idx_pool[:warm_up]
    idx_remaining = idx_pool[warm_up:]
    
    list_1, list_0 = trt_ctr(T_train[idx_sub_training])
    
    # Initialize the data-limited starting size as 20% of whole treated training set
    idx_sub_training_1 = idx_sub_training[list_1]  # 20% as initial
    idx_sub_training_0 = idx_sub_training[list_0]  # 10% as initial
    
    acquired_treated, acquired_control = None, None
    error_list = []
    num_of_acquire = [len(idx_sub_training_1) + len(idx_sub_training_0)]

    for query_step in range(al_step):
        
        train_x_1, train_y_1 = combine_x_train[idx_sub_training_1], combined_y_train[idx_sub_training_1]
        train_x_0, train_y_0 = combine_x_train[idx_sub_training_0], combined_y_train[idx_sub_training_0]
        print("Number of data used for training in treated and control:", len(idx_sub_training_1), len(idx_sub_training_0))

        # Do t-SNE visualization for the updataed dataset
        # Concatenate the data matrices

        X = np.vstack([train_x_1.cpu(), train_x_0.cpu()])

        # Create labels for the data points
        labels_ = np.ones(X.shape[0])
        labels_[train_x_1.shape[0]:] = 0  # Set labels for control x_0 as 0

        print("Ploting the t-SNE")
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=args.seed)
        print("Initializing the tsne function to call")
        X_embedded_ = tsne.fit_transform(X)
        print("Embedding complete")

        # Save the embedding and labels to a single file
        np.savez('embeddings/truerandom/embedding_and_labels_{}.npz'.format(query_step), X_embedded=X_embedded_, labels=labels_)
        print("Embedding and labels saved to embedding_and_labels.npz")

        # To reload the embedding and labels later
        print("Reloading the embedding and labels from file")
        data = np.load('embeddings/truerandom/embedding_and_labels_{}.npz'.format(query_step))
        X_embedded = data['X_embedded']
        labels = data['labels']

        # Concatenate vertically and shuffle randomly for the sub training
        combine_train_idx = np.concatenate([idx_sub_training_1, idx_sub_training_0], axis=0)
        np.random.shuffle(combine_train_idx)
        train_dataset = toDataLoader(x_train=combine_x_train[combine_train_idx].cpu(),
                                     y_train=combined_y_train[combine_train_idx].cpu(),
                                     t_train=T_train[combine_train_idx].cpu())

        tune_dataset = toDataLoader(x_train=combine_x_valid.cpu(),
                                    y_train=combined_y_valid.cpu(),
                                    t_train=T_valid.cpu())

        job_dir_path = Path('saved_models/IBM-DUE/method_{}/seed_{}/step_{}'.format(config.get("acquisition_function"), seed, query_step))

        train_deep_kernel_gp(ds_train=train_dataset,
                             ds_valid=tune_dataset,
                             job_dir=job_dir_path,
                             config=config,
                             dim_input=train_x_1.shape[1],
                             seed=seed)

        test_dataset = toDataLoader(x_train=combine_x_test.cpu(),
                                    y_train=tau_test.cpu(),
                                    t_train=T_test.cpu())

        (mu_0, mu_1), _ = predict_deep_kernel_gp(dataset=test_dataset,
                                                 job_dir=job_dir_path,
                                                 config=config,
                                                 dim_input=train_x_1.shape[1],
                                                 seed=seed
                                                )

        pehe_error = evaluation_nn(pred_1=mu_1.mean(0) * y_std,
                                   pred_0=mu_0.mean(0) * y_std,
                                   test_tau=tau_test.cpu(),
                                   query_step=query_step
                                   )

        error_list.append(np.round(pehe_error.cpu().numpy(), 4))

        acquired_index = random_sampling(pool=idx_remaining, num_of_samples=num_of_samples)
        
        treated_idx, control_idx = trt_ctr(T_train[acquired_index])
        
        acquired_treated, acquired_control = acquired_index[treated_idx], acquired_index[control_idx]
        
        if len(acquired_treated) == 0 and len(acquired_control) == 0:
            raise  TypeError("Nothing acquired by AL")
        else:
            if query_step != 0:
                num_of_acquire.append(num_of_acquire[query_step - 1] + len(acquired_treated) + len(acquired_control))
            #plot(combine_x_train, combined_y_train, model_1, model_0, likelihood_1, likelihood_0, train_x_1, train_x_0, train_y_1, train_y_0, acquired_treated, acquired_control)
            
        if len(acquired_treated) != 0:
            idx_sub_training_1, idx_remaining = pool_updating(idx_remaining, idx_sub_training_1, acquired_treated)
        if len(acquired_control) != 0: 
            idx_sub_training_0, idx_remaining = pool_updating(idx_remaining, idx_sub_training_0, acquired_control)

    average_pehe = np.array(error_list)
    
# Specify the file path
file_path = 'text_results/truerandom/pehe_truerandom_{}.csv'.format(args.seed)
    
# Open the CSV file in write mode
with open(file_path, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)
    
    # Write the data for list_1
    csv_writer.writerow(['Number of Samples'] + list(map(str, num_of_acquire)))
    
    # Write the data for list_2
    csv_writer.writerow(['PEHE'] + list(map(str, average_pehe.tolist())))
    
print(f'The data has been successfully written to {file_path}')

