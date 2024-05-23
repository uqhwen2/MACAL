import torch
import numpy as np
import math
from torch.utils import data

from causal_bald.library.datasets import utils

class Synthetic0(data.Dataset):
    def __init__(
        self,
        num_examples,
        mode,
        beta=0.75,
        sigma_y=1.0,
        bimodal=False,
        seed=1331,
        split=None,
    ):
        super(Synthetic0, self).__init__()
        rng = np.random.RandomState(seed=seed)
        self.num_examples = num_examples
        self.dim_input = 1
        self.dim_treatment = 1
        self.dim_output = 1
        if bimodal:
            self.x = np.vstack(
                [
                    rng.normal(loc=-2, scale=0.7, size=(num_examples // 2, 1)).astype(
                        "float32"
                    ),
                    rng.normal(loc=2, scale=0.7, size=(num_examples // 2, 1)).astype(
                        "float32"
                    ),
                ]
            )
        else:
            self.x = rng.normal(size=(num_examples, 1)).astype("float32")

        self.pi = (
            utils.complete_propensity(x=self.x, u=0, lambda_=1.0, beta=beta)
            .astype("float32")
            .ravel()
        )
        self.t = rng.binomial(1, self.pi).astype("float32")
        eps = (sigma_y * rng.normal(size=self.t.shape)).astype("float32")
        self.mu0 = utils.f_mu(x=self.x, t=0.0, u=0, gamma=0.0).astype("float32").ravel()
        self.mu1 = utils.f_mu(x=self.x, t=1.0, u=0, gamma=0.0).astype("float32").ravel()
        self.y0 = self.mu0 + eps
        self.y1 = self.mu1 + eps
        self.y = self.t * self.y1 + (1 - self.t) * self.y0
        self.tau = self.mu1 - self.mu0
        if mode == "pi":
            self.inputs = self.x
            self.targets = self.t
        elif mode == "mu":
            self.inputs = np.hstack([self.x, np.expand_dims(self.t, -1)])
            self.targets = self.y
        else:
            raise NotImplementedError(
                f"{mode} not supported. Choose from 'pi'  for propensity models or 'mu' for expected outcome models"
            )
        self.y_mean = np.array([0.0], dtype="float32")
        self.y_std = np.array([1.0], dtype="float32")
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

        #print(self.mu1.shape, self.y.shape, self.y0.shape, self.tau.shape, self.inputs.shape, self.targets.shape)
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index : index + 1]

    def tau_fn(self, x):
        return utils.f_mu(x=x, t=1.0, u=1.0, gamma=0.0) - utils.f_mu(
            x=x, t=0.0, u=1.0, gamma=0.0
        )


class Synthetic(data.Dataset):
    def __init__(
        self,
        num_examples,
        mode,
        beta=0.75,
        sigma_y=1.0,
        bimodal=False,
        seed=1331,
        split=None,
    ):
        super(Synthetic, self).__init__()
        rng = np.random.RandomState(seed=seed)
        self.num_examples = num_examples
        self.dim_input = 1
        self.dim_treatment = 1
        self.dim_output = 1
        if bimodal:
            self.x = np.vstack(
                [
                    rng.normal(loc=-2, scale=0.7, size=(num_examples // 2, 1)).astype(
                        "float32"
                    ),
                    rng.normal(loc=2, scale=0.7, size=(num_examples // 2, 1)).astype(
                        "float32"
                    ),
                ]
            )
        else:
            self.x = rng.normal(size=(num_examples, 1)).astype("float32")

        self.pi = (
            utils.complete_propensity(x=self.x, u=0, lambda_=1.0, beta=beta)
            .astype("float32")
            .ravel()
        )

        combine_x, _, _, T, combine_y_1, combine_y_0 = self.generation()
        # Create an array of indices (assuming you have an array of size 100)
        indices = np.arange(len(combine_x))
        # Shuffle the indices in-place
        np.random.shuffle(indices)

        self.x = combine_x[indices].reshape(-1, 1)
        self.t = T[indices]
        self.mu0 = combine_y_0[indices]
        self.mu1 = combine_y_1[indices]
        self.y0 = self.mu0
        self.y1 = self.mu1
        self.y = self.t * self.y1 + (1 - self.t) * self.y0
        self.tau = self.mu1 - self.mu0
        if mode == "pi":
            self.inputs = self.x
            self.targets = self.t
        elif mode == "mu":
            self.inputs = np.hstack([self.x, np.expand_dims(self.t, -1)])
            #self.inputs = np.hstack([self.x, self.t])
            self.targets = self.y
        else:
            raise NotImplementedError(
                f"{mode} not supported. Choose from 'pi'  for propensity models or 'mu' for expected outcome models"
            )
        self.y_mean = np.array([0.0], dtype="float32")
        self.y_std = np.array([1.0], dtype="float32")
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)
        #print(self.mu1.shape, self.y.shape, self.y0.shape, self.tau.shape, self.t.shape)

    def generation(self, mean_1=0, std_1=1, func_1=torch.sin, mean_0=3, std_0=1, func_0=torch.cos):
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

        combine_x = torch.cat([train_x_1, train_x_0], dim=0).float()
        combined_y = torch.cat([train_y_1, train_y_0], dim=0).float()
        combine_y_1 = func_1(combine_x * (2 * math.pi) / math.pi).float()
        combine_y_0 = func_0(combine_x * (2 * math.pi) / math.pi).float()
        tau = combine_y_1 - combine_y_0

        treated_x = torch.ones_like(train_x_1)
        control_x = torch.zeros_like(train_x_0)
        T = torch.cat([treated_x, control_x], dim=0).float()

        return combine_x.numpy(), combined_y.numpy(), tau.numpy(), T.numpy(), combine_y_1.numpy(), combine_y_0.numpy()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index : index + 1]

#    def tau_fn(self, x):
#        return utils.f_mu(x=x, t=1.0, u=1.0, gamma=0.0) - utils.f_mu(
#            x=x, t=0.0, u=1.0, gamma=0.0
#        )
