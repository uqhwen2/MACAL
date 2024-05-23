import torch.nn as nn
from causal_bald.library.models.deep_kernel import DeepKernelGP

class nnModel_1(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(nnModel_1, self).__init__()

        self.fc_y1_pred = nn.Sequential(
            # spectral_norm(nn.Linear(input_dim, latent_dim)), nn.ELU(),
            # spectral_norm(nn.Linear(latent_dim, latent_dim)), nn.ELU(),  # nn.Dropout(0.1),
            # spectral_norm(nn.Linear(latent_dim, latent_dim)), nn.ELU(),  # nn.Dropout(0.1),
            # spectral_norm(nn.Linear(latent_dim, output_dim))
            nn.Linear(input_dim, latent_dim), nn.ELU(),
            nn.Linear(latent_dim, latent_dim), nn.ELU(),
            nn.Linear(latent_dim, latent_dim), nn.ELU(),
            nn.Linear(latent_dim, output_dim)
        )

    def forward(self, x):
        latent_x = self.fc_y1_pred(x)

        return latent_x.reshape(-1)


# We will use the simplest form of GP model, exact inference
class nnModel_0(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(nnModel_0, self).__init__()

        self.fc_y0_pred = nn.Sequential(
            # spectral_norm(nn.Linear(input_dim, latent_dim)), nn.ELU(),
            # spectral_norm(nn.Linear(latent_dim, latent_dim)), nn.ELU(),  # nn.Dropout(0.1),
            # spectral_norm(nn.Linear(latent_dim, latent_dim)), nn.ELU(),  # nn.Dropout(0.1),
            # spectral_norm(nn.Linear(latent_dim, output_dim))
            nn.Linear(input_dim, latent_dim), nn.ELU(),
            nn.Linear(latent_dim, latent_dim), nn.ELU(),
            nn.Linear(latent_dim, latent_dim), nn.ELU(),
            nn.Linear(latent_dim, output_dim)
        )

    def forward(self, x):
        latent_x = self.fc_y0_pred(x)

        return latent_x.reshape(-1)


class TARNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):  # Define all the neural nets

        super(TARNet, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Feature extractor
        self.encoder = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ELU(),  # nn.Dropout(0.1),
                                     nn.Linear(self.hidden_dim, self.hidden_dim), nn.ELU(),
                                     nn.Linear(self.hidden_dim, self.latent_dim), nn.ELU()
                                     )

        # Prediction nerual nets
        self.fc_y1_pred = nn.Sequential(nn.Linear(self.latent_dim, self.hidden_dim), nn.ELU(),  # nn.Dropout(0.1),
                                        nn.Linear(self.hidden_dim, self.hidden_dim), nn.ELU(),  # nn.Dropout(0.1),
                                        nn.Linear(self.hidden_dim, 1))  # model the mean of y1

        self.fc_y0_pred = nn.Sequential(nn.Linear(self.latent_dim, self.hidden_dim), nn.ELU(),
                                        nn.Linear(self.hidden_dim, self.hidden_dim), nn.ELU(),  # nn.Dropout(0.1),
                                        nn.Linear(self.hidden_dim, 1))  # model the mean of y0

    def forward(self, x, t):  # Sample the latent distribution and forward prediction

        hidden = self.encoder(x)

        y1_pred = self.fc_y1_pred(hidden)
        y0_pred = self.fc_y0_pred(hidden)

        y_factual = torch.where(t, y1_pred, y0_pred)

        return y1_pred, y0_pred, y_factual


def train_deep_kernel_gp(ds_train, ds_valid, job_dir, config, dim_input, seed):
    if not (job_dir / "best_checkpoint.pt").exists():
        # Get model parameters from config
        kernel = config.get("kernel")
        num_inducing_points = config.get("num_inducing_points")
        dim_hidden = config.get("dim_hidden")
        dim_output = config.get("dim_output")
        depth = config.get("depth")
        negative_slope = config.get("negative_slope")
        dropout_rate = config.get("dropout_rate")
        spectral_norm = config.get("spectral_norm")
        learning_rate = config.get("learning_rate")
        batch_size = config.get("batch_size")
        epochs = config.get("epochs")
        model = DeepKernelGP(
            job_dir=job_dir,
            kernel=kernel,
            num_inducing_points=num_inducing_points,
            inducing_point_dataset=ds_train,
            architecture="resnet",
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            dim_output=dim_output,
            depth=depth,
            negative_slope=negative_slope,
            batch_norm=False,
            spectral_norm=spectral_norm,
            dropout_rate=dropout_rate,
            weight_decay=(0.5 * (1 - config.get("dropout_rate"))) / len(ds_train),
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=5,
            num_workers=0,
            seed=seed
        )
        _ = model.fit(ds_train, ds_valid)


def predict_deep_kernel_gp(dataset, job_dir, config, dim_input, seed):
    # Get model parameters from config
    kernel = config.get("kernel")
    num_inducing_points = config.get("num_inducing_points")
    dim_hidden = config.get("dim_hidden")
    dim_output = config.get("dim_output")
    depth = config.get("depth")
    negative_slope = config.get("negative_slope")
    dropout_rate = config.get("dropout_rate")
    spectral_norm = config.get("spectral_norm")
    learning_rate = config.get("learning_rate")
    batch_size = config.get("batch_size")
    epochs = config.get("epochs")
    model = DeepKernelGP(
        job_dir=job_dir,
        kernel=kernel,
        num_inducing_points=num_inducing_points,
        inducing_point_dataset=dataset,
        architecture="resnet",
        dim_input=dim_input,
        dim_hidden=dim_hidden,
        dim_output=dim_output,
        depth=depth,
        negative_slope=negative_slope,
        batch_norm=False,
        spectral_norm=spectral_norm,
        dropout_rate=dropout_rate,
        weight_decay=(0.5 * (1 - config.get("dropout_rate"))) / len(dataset),
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=5,
        num_workers=0,
        seed=seed
    )
    model.load()
    return model.predict_mus(dataset), model
