import torch
from torch.autograd import Function
# import cpp_cuda_module  # 你编译好的 C++/CUDA 扩展
from pytouch.optimized_quantizable_gru import OptimizedQuantizableGRU

def test_cuda_gru_vs_python():
    batch_size = 4
    seq_len = 10
    input_size = 32
    hidden_size = 64
    num_layers = 2

    x = torch.randn(batch_size, seq_len, input_size, device='cuda')

    # Python 版 OptimizedQuantizableGRU
    gru_py = OptimizedQuantizableGRU(input_size, hidden_size, num_layers).cuda()
    gru_py.eval()

    # CUDA kernel GRU
    gru_cuda = OptimizedQuantizableGRUCUDAModule(input_size, hidden_size, num_layers).cuda()
    # 迁移权重
    with torch.no_grad():
        for l in range(num_layers):
            cell_py = gru_py.cells[l]
            gru_cuda.weight_ih.data.copy_(cell_py.weight_ih.weight.data)
            gru_cuda.weight_hh.data.copy_(cell_py.weight_hh.weight.data)
            gru_cuda.bias_ih.data.copy_(cell_py.weight_ih.bias.data)
            gru_cuda.bias_hh.data.copy_(cell_py.weight_hh.bias.data)

    with torch.no_grad():
        out_py, h_py = gru_py(x)
        out_cuda, h_cuda = gru_cuda(x)

    max_diff_out = (out_py - out_cuda).abs().max().item()
    max_diff_h = (h_py - h_cuda).abs().max().item()

    print(f"最大输出差异: {max_diff_out:.6e}")
    print(f"最大隐藏状态差异: {max_diff_h:.6e}")
    assert max_diff_out < 1e-5 and max_diff_h < 1e-5
    print("✅ CUDA 前向 GRU 与 Python 版本一致")

if __name__ == "__main__":
    test_cuda_gru_vs_python()