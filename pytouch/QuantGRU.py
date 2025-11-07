import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

# 这里假设你已经通过 cpp_extension 编译了 kernel
# import quant_gru_cpp as _quant_gru

class QuantGRU(nn.Module):
    """
    PyTorch 封装的纯定点 GRU
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bias: bool = True,
            batch_first: bool = False,
            int_type: str = "int8",  # "int8" / "int16"
            quant_mode: str = "dynamic"  # "dynamic" / "static"
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.int_type = int_type
        self.quant_mode = quant_mode

        # 权重
        self.Wx = nn.Parameter(torch.randn(input_size, 3*hidden_size))
        self.Rh = nn.Parameter(torch.randn(hidden_size, 3*hidden_size))
        if bias:
            self.bx = nn.Parameter(torch.zeros(3*hidden_size))
            self.br = nn.Parameter(torch.zeros(3*hidden_size))
        else:
            self.register_parameter("bx", None)
            self.register_parameter("br", None)

    def forward(self, x: Tensor, h0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        x: [seq_len, batch, input_size] 或 [batch, seq_len, input_size] (batch_first)
        h0: [num_layers, batch, hidden_size] 或 None
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        device = x.device

        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        # 自动处理数据到 GPU
        Wx = self.Wx.to(device)
        Rh = self.Rh.to(device)
        bx = self.bx.to(device) if self.bias else None
        br = self.br.to(device) if self.bias else None

        # 调用 C++/CUDA kernel
        # 这里 kernel 会封装你的 GruTrain
        # kernel 应该返回 output, hn
        # 支持 int8 / int16 / float
        # output, hn = _quant_gru.forward(
        #     x, h0, Wx, Rh, bx, br, self.int_type, self.quant_mode
        # )
        raise NotImplementedError("请用你的 CUDA kernel 实现")

        return output, hn
