import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super(SpatialAttention, self).__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.out = nn.Linear(output_dim, input_dim)

    def split_heads(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, x.size(-1) // self.num_heads).transpose(1, 2)

    def combine_heads(self, x):
        batch_size = x.size(0)
        return x.transpose(1, 2).contiguous().view(batch_size, -1, x.size(-1) * self.num_heads)

    def forward(self, x):
        Q = self.split_heads(self.query(x))
        K = self.split_heads(self.key(x))
        V = self.split_heads(self.value(x))
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        attended_combined = self.combine_heads(attended)
        return self.out(attended_combined)

class MultiScaleAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_scales=3):
        super(MultiScaleAttention, self).__init__()
        self.scales = nn.ModuleList([SpatialAttention(input_dim, output_dim) for _ in range(num_scales)])
        self.scale_weights = nn.Parameter(torch.ones(num_scales))

    def forward(self, x):
        outputs = [scale(x) for scale in self.scales]
        weighted_output = torch.stack(outputs, dim=0) * self.scale_weights.view(-1, 1, 1, 1)
        return weighted_output.sum(dim=0)
