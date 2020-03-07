import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
class FCLayer(torch.nn.Module):
    # this is an implementation of fully connected layer with arbitrary activation function. 
    # The implementation is meant to demonstrate technical sk
    def __init__(self, input_size, output_size, activation=F.relu):
        super().__init__()
        self.weight = Parameter(torch.Tensor(output_size, input_size))
        self.bias   = Parameter(torch.Tensor(output_size))
        self.activation = activation
        self.reset_parameters()
    def reset_parameters(self):
        init.normal_(self.weight)
        init.uniform_(self.bias)
    def forward(self, x):
        output = x.matmul(self.weight.t()) + self.bias
        output = self.activation(output)
        return output