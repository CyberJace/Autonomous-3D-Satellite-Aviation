import torch
import torch.nn as nn
import torch.nn.functional as F

class SimSREncoder(nn.Module):
    """
    SimSR (Simple Distance-Based State Representation) Encoder.
    Maps high-dimensional sensor/spatial observations to a compact latent vector 
    while preserving distance geometry via optimal transport / Wasserstein alignment.
    """
    def __init__(self, input_dim=16, latent_dim=8):
        super(SimSREncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.latent_head = nn.Linear(32, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        z = self.latent_head(h)
        # L2 normalize latent space for stable distance-based reward & Wasserstein loss
        return F.normalize(z, p=2, dim=-1)

def wasserstein_distance_loss(z_batch1, z_batch2, p=2):
    """
    Approximates the p-Wasserstein distance between two latent state distributions 
    to regularize representation drift across known and unknown obstacle domains.
    """
    # Pairwise distance matrix in latent space
    cost_matrix = torch.cdist(z_batch1, z_batch2, p=p)
    # Return mean optimal transport cost approximation
    return torch.mean(torch.min(cost_matrix, dim=1)[0]) + torch.mean(torch.min(cost_matrix, dim=0)[0])
