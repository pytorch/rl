import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, depth=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LazyConv2d( depth, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth*2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth*2, depth*4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth*4, depth*8, 4, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, observation):
        if observation.dim() > 4:
            *batch_sizes, C, H, W = observation.shape
            observation = observation.view(-1, C, H, W)
        obs_encoded = self.encoder(observation)
        latent = obs_encoded.view(obs_encoded.size(0), -1)
        if observation.dim() > 4:
            latent = latent.view(*batch_sizes, -1)
        return latent



class ConvDecoder(nn.Module):
    def __init__(self, depth=32):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(depth*8, depth*4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(depth*4, depth*2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(depth*2, depth, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(depth, 3, 4, stride=2, padding=1),
        )
        self._depth = depth

    def forward(self, latent):
        if latent.dim() > 2:
            *batch_sizes, D = latent.shape
            latent = latent.view(-1, D)
        w = h = latent.size(1) // (8*self._depth)
        latent_reshaped = latent.view(latent.size(0), 8*self._depth, w, h)
        obs_decoded = self.decoder(latent_reshaped)
        if latent.dim() > 2:
            _, C, H, W = obs_decoded.shape
            obs_decoded = obs_decoded.view(*batch_sizes, C, H, W)
        return obs_decoded

