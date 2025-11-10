import torch
from torch.autograd import Function


class QuantizableGRUCUDAModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # 创建权重和偏置
        self.weight_ih = torch.nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.empty(3 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / (self.hidden_size ** 0.5)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, h0=None):
        if h0 is None:
            h0 = input.new_zeros(self.num_layers, input.shape[0], self.hidden_size)
        return CudaQuantizedGRU.apply(input, h0, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
