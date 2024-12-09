import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AFMM']

class middleLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        H_hat_init = torch.ones([1, out_dim, in_dim], device='cuda')*0.1
        b_init = torch.zeros([1, out_dim, 1], device='cuda')*0.1
        a_hat_init = torch.zeros([1, out_dim, 1], device='cuda')*0.1
        
        self.H_hat = nn.Parameter(H_hat_init)
        self.b = nn.Parameter(b_init)
        self.a_hat = nn.Parameter(a_hat_init)
        
    def forward(self, x):
        H = F.softplus(self.H_hat)
        a = F.tanh(self.a_hat)
        
        x = torch.matmul(H, x) + self.b
        x = x + a * F.tanh(x)
        
        return x
        

class finalLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        H_hat_init = torch.ones([1, out_dim, in_dim], device='cuda')*0.1
        b_init = torch.zeros([1, out_dim, 1], device='cuda')*0.1
        
        self.H_hat = nn.Parameter(H_hat_init)
        self.b = nn.Parameter(b_init)
        
    def forward(self, x):
        H = F.softplus(self.H_hat)
        
        x = torch.matmul(H, x) + self.b
        x = F.sigmoid(x)
        
        return x

# AFMM : Activation Function Mimicing Model
class AFMM(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        layers = []
        if num_layers > 1:
            layers += [middleLayer(1, hidden_dim)]
            layers += [middleLayer(hidden_dim, hidden_dim)] * (num_layers - 2)
        layers += [finalLayer(hidden_dim, 1)]
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.layers(x)
        return x
    