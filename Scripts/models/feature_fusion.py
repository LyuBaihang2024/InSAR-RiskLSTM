import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    def __init__(self, spatial_dim, temporal_dim, fusion_dim, output_dim):
        super(FeatureFusion, self).__init__()
        self.spatial_fc = nn.Linear(spatial_dim, fusion_dim)
        self.temporal_fc = nn.Linear(temporal_dim, fusion_dim)
        self.fusion_fc = nn.Linear(fusion_dim * 2, output_dim)
        self.bn1 = nn.BatchNorm1d(fusion_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, spatial_features, temporal_features):
        spatial_out = torch.relu(self.bn1(self.spatial_fc(spatial_features)))
        temporal_out = torch.relu(self.bn1(self.temporal_fc(temporal_features)))
        combined = torch.cat([spatial_out, temporal_out], dim=-1)
        return torch.relu(self.bn2(self.fusion_fc(combined)))

class ResidualFusion(nn.Module):
    def __init__(self, input_dim, fusion_dim, output_dim):
        super(ResidualFusion, self).__init__()
        self.fc1 = nn.Linear(input_dim, fusion_dim)
        self.fc2 = nn.Linear(fusion_dim, output_dim)
        self.shortcut = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        residual = self.shortcut(x)
        out = torch.relu(self.fc1(x))
        return torch.relu(self.fc2(out) + residual)
