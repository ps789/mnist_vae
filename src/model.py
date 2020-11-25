import torch
import torch.nn as nn
import torch.nn.functional as F
features = 20
class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=784, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=features*2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x).view(-1, 2, features)
        mu = x[:, 0, :]
        log_var = x[:, 1, :]
        std = torch.exp(0.5*log_var) # standard deviation
        sample_normal = torch.randn_like(log_var)
        latent_sample = mu + (sample_normal * std)

        reconstruction = self.decoder(latent_sample)
        return reconstruction, mu, log_var
