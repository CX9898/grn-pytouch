#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cmath>
#include <iostream>

#include "gru_interface.h"
#include "gru_quantization_ranges.h"
#include "histogram_collector.h"
#include "calibration_gpu.cuh"
#include "quantize_ops_helper.h"

// 全局 cublas handle
static cublasHandle_t g_blas_handle = nullptr;

// 初始化 cublas handle 的包装函数
void init_gru_cublas_wrapper() { init_gru_cublas(g_blas_handle); }

// CalibrationMethod 字符串转换
// Python 端只需要传字符串，C++ wrapper 内部转换
inline CalibrationMethod stringToCalibMethod(const std::string &method) {
    if (method.empty() || method == "none") {
        return CalibrationMethod::NONE;
    } else if (method == "minmax") {
        return CalibrationMethod::MINMAX;
    } else if (method == "sqnr") {
        return CalibrationMethod::SQNR;
    } else if (method == "percentile") {
        return CalibrationMethod::PERCENTILE;
    } else {
        throw std::invalid_argument("Invalid calibration method: '" + method + 
            "'. Valid options: 'none', 'minmax', 'sqnr', 'percentile'");
    }
}

// ============================================================================
//                    位宽转换辅助函数（前置定义）
// ============================================================================
//
// 设计说明：
// - Python 端只配置位宽数量（1-32），不关心实际类型（INT/UINT）
// - C++ 端在 to_cpp() 时决定实际类型
// - 这些辅助函数需要在 OperatorQuantConfigPy 之前定义
//
// ============================================================================

/// 位宽转换：Python 端的位宽数值 → C++ QuantBitWidth 结构体
/// @param bitwidth 位宽 (1-32)
/// @param is_unsigned false=INT有符号(默认), true=UINT无符号
inline QuantBitWidth toBitwidth(int8_t bitwidth, bool is_unsigned = false) {
    if (bitwidth < 1 || bitwidth > 32) {
        throw std::invalid_argument("Invalid bitwidth: " + std::to_string(bitwidth) +
                                    ". Must be 1-32.");
    }
    return QuantBitWidth(bitwidth, is_unsigned);
}

// ============================================================================
//                    OperatorQuantConfig Python 绑定（前置定义）
// ============================================================================
//
// 注意：这个结构体需要在 GRUQuantParamsPy 之前定义，
// 因为 GRUQuantParamsPy 中包含 OperatorQuantConfigPy 成员
//
// ============================================================================

struct OperatorQuantConfigPy {
    // 位宽配置（存储位宽数量 1-32）
    // 命名与 C++ OperatorQuantConfig 保持一致
    int8_t x_, h_;
    int8_t W_, R_, bw_, br_;
    int8_t weight_ih_linear_, weight_hh_linear_;  // GEMM+bias 融合输出
    int8_t update_gate_input_, update_gate_output_;
    int8_t reset_gate_input_, reset_gate_output_;
    int8_t new_gate_input_, new_gate_output_;
    int8_t mul_reset_hidden_;
    int8_t mul_old_contribution_, mul_new_contribution_;

    // 对称量化配置
    bool x_symmetric_, h_symmetric_;
    bool W_symmetric_, R_symmetric_, bw_symmetric_, br_symmetric_;
    bool weight_ih_linear_symmetric_, weight_hh_linear_symmetric_;
    bool update_gate_input_symmetric_, update_gate_output_symmetric_;
    bool reset_gate_input_symmetric_, reset_gate_output_symmetric_;
    bool new_gate_input_symmetric_, new_gate_output_symmetric_;
    bool mul_reset_hidden_symmetric_;
    bool mul_old_contribution_symmetric_, mul_new_contribution_symmetric_;

    // 无符号量化配置（与 C++ is_unsigned_ 一致，只标记例外情况）
    bool x_unsigned_, h_unsigned_;
    bool W_unsigned_, R_unsigned_, bw_unsigned_, br_unsigned_;
    bool weight_ih_linear_unsigned_, weight_hh_linear_unsigned_;
    bool update_gate_input_unsigned_, update_gate_output_unsigned_;
    bool reset_gate_input_unsigned_, reset_gate_output_unsigned_;
    bool new_gate_input_unsigned_, new_gate_output_unsigned_;
    bool mul_reset_hidden_unsigned_;
    bool mul_old_contribution_unsigned_, mul_new_contribution_unsigned_;

    // 量化粒度配置（仅对 W, R, bw, br 有效）
    // 0=PER_TENSOR, 1=PER_GATE, 2=PER_CHANNEL
    int W_granularity_ = 2;  // 默认 PER_CHANNEL
    int R_granularity_ = 2;
    int bw_granularity_ = 2;
    int br_granularity_ = 2;
    bool usePOT2_ = false;
    // 0=Round, 1=CoverRange(default), 2=Floor —— 与 C++ PotScaleMethod 一致
    int pot_scale_method_ = 1;
    float pot_scale_tolerance_ = 0.02f;

    // 方法声明（实现在文件末尾）
    OperatorQuantConfigPy();                           // 默认构造函数：从 C++ 默认值初始化
    OperatorQuantConfig to_cpp() const;                // 转换为 C++ 结构体
    void from_cpp(const OperatorQuantConfig &cfg);     // 从 C++ 结构体读取
};

// GRUQuantizationRanges 的 Python 绑定（直接包装 C++ 对象）
struct GRUQuantizationRangesPy {
    GRUQuantizationRanges cpp_ranges;  // 直接包装 C++ 对象

    // 默认构造函数
    GRUQuantizationRangesPy() = default;

    // 带 hidden 参数的构造函数
    explicit GRUQuantizationRangesPy(int hidden) : cpp_ranges(hidden) {}

    // 重置所有范围
    void reset(int hidden = -1) { cpp_ranges.reset(hidden); }

    // 直接访问内部 C++ 对象（用于 C++ 层调用）
    GRUQuantizationRanges& get_cpp() { return cpp_ranges; }
    const GRUQuantizationRanges& get_cpp() const { return cpp_ranges; }
};

// GRUQuantParams 的 Python 绑定
// 命名与 C++ GRUQuantParams 保持一致
struct GRUQuantParamsPy {
    int hidden_;
    // 基础参数
    float scale_x_;
    int32_t zp_x_;
    float scale_h_;
    int32_t zp_h_;
    // 权重/偏置参数：唯一 per-channel 权威数组（始终 3*hidden，按粒度广播）。
    // tensor 视图 = scale_W_[0]，gate 视图 = scale_W_[g*hidden]，纯索引还原，不再单独存储。
    std::vector<float> scale_W_;
    std::vector<float> scale_R_;
    std::vector<float> scale_bw_;
    std::vector<float> scale_br_;
    // Linear 输出参数 (GEMM+bias)
    float scale_weight_ih_linear_;
    int32_t zp_weight_ih_linear_;
    float scale_weight_hh_linear_;
    int32_t zp_weight_hh_linear_;
    // 门激活函数输入参数（pre-activation）
    float scale_update_gate_input_;
    int32_t zp_update_gate_input_;
    float scale_reset_gate_input_;
    int32_t zp_reset_gate_input_;
    float scale_new_gate_input_;
    int32_t zp_new_gate_input_;
    // 门激活函数输出参数（post-activation）
    float scale_update_gate_output_;
    int32_t zp_update_gate_output_;
    float scale_reset_gate_output_;
    int32_t zp_reset_gate_output_;
    float scale_new_gate_output_;
    int32_t zp_new_gate_output_;
    // 中间计算参数
    float scale_mul_reset_hidden_;
    int32_t zp_mul_reset_hidden_;
    // 隐状态更新参数
    float scale_mul_new_contribution_;
    int32_t zp_mul_new_contribution_;
    float scale_mul_old_contribution_;
    int32_t zp_mul_old_contribution_;

    // ⚠️ 关键字段：位宽配置必须在 Python 和 C++ 之间正确传递
    // 否则会使用默认的 8 位配置
    OperatorQuantConfigPy bitwidth_config_;

    // 方法声明（实现在文件末尾）
    void from_cpp(const GRUQuantParams &cpp_params);  // 从 C++ 结构体转换
    GRUQuantParams to_cpp() const;                     // 转换为 C++ 结构体
};

// 根据量化范围计算量化参数的包装函数（支持自定义位宽配置）
GRUQuantParamsPy calculate_gru_quantitative_parameters_wrapper(
    const GRUQuantizationRangesPy &quant_ranges,
    const OperatorQuantConfigPy &bitwidth_config = OperatorQuantConfigPy()) {
    // 直接使用包装的 C++ 对象
    OperatorQuantConfig cpp_bitwidth = bitwidth_config.to_cpp();

    // 调用 C++ 函数
    GRUQuantParams quant_params =
        calculateGRUQuantitativeParameters(quant_ranges.cpp_ranges, cpp_bitwidth);

    GRUQuantParamsPy py_params;
    py_params.from_cpp(quant_params);
    return py_params;
}

// =====================================================================
// AIMET 风格直方图校准接口（默认使用 GPU 加速）
// =====================================================================

// GRUHistogramCollectors 的 Python 绑定包装
// 内部使用 GPU 版本实现，Python 接口保持不变
struct GRUHistogramCollectorsPy {
    GRUGPUHistogramCollectors gpu_collectors;  // 使用 GPU 版本

    GRUHistogramCollectorsPy() = default;
    explicit GRUHistogramCollectorsPy(int hidden, int num_bins = 2048) : gpu_collectors(hidden, num_bins) {}

    void reset(int hidden = -1, int num_bins = -1) { gpu_collectors.reset(hidden, num_bins); }
    bool is_valid() const { return gpu_collectors.is_valid(); }
    int hidden() const { return gpu_collectors.hidden_; }
    int num_bins() const { return gpu_collectors.num_bins_; }
    
    // 转换为 CPU 版本（内部使用）
    GRUHistogramCollectors to_cpu() const {
        return convertGPUHistogramsToCPU(gpu_collectors);
    }
    
    // 打印统计信息
    void print() const {
        std::cout << "GRUHistogramCollectors (GPU):" << std::endl;
        std::cout << "  hidden: " << gpu_collectors.hidden_ << std::endl;
        std::cout << "  num_bins: " << gpu_collectors.num_bins_ << std::endl;
        std::cout << "  is_valid: " << (is_valid() ? "true" : "false") << std::endl;
    }
};

// 从直方图计算量化参数的包装函数
// 内部自动选择 GPU/CPU 实现：SQNR 用 GPU，Percentile 用 CPU
GRUQuantParamsPy calculate_gru_quantitative_parameters_from_histograms_wrapper(
    GRUHistogramCollectorsPy &hist_collectors,
    const OperatorQuantConfigPy &bitwidth_config = OperatorQuantConfigPy(),
    bool use_percentile = false,
    float percentile_value = 99.99f) {

    OperatorQuantConfig cpp_bitwidth = bitwidth_config.to_cpp();
    GRUQuantParams quant_params;
    
    if (use_percentile) {
        // Percentile 方案：转换为 CPU 直方图后计算
        GRUHistogramCollectors cpu_collectors = hist_collectors.to_cpu();
        quant_params = calculateGRUQuantitativeParametersFromHistograms(
            cpu_collectors, cpp_bitwidth, true, percentile_value);
    } else {
        // SQNR 方案：直接使用 GPU 计算
        quant_params = calculateGRUQuantitativeParametersFromGPUHistograms(
            hist_collectors.gpu_collectors, cpp_bitwidth);
    }

    GRUQuantParamsPy py_params;
    py_params.from_cpp(quant_params);
    return py_params;
}

// =====================================================================
// forward_quant_wrapper: 量化前向传播（训练/推理）
// =====================================================================
// 返回: (h, v, W_q, R_q, bw_q, br_q, x_q, x_mask, h0_mask, W_mask, R_mask, bw_mask, br_mask,
//        weight_ih_linear_mask, weight_hh_linear_mask, gate_input_mask, gate_output_mask, h_mask)
//   h: [T+1, B, H] 隐藏状态序列（包含初始状态）
//   v: [T, B, H*4] 中间值（训练时需要，推理时可忽略）
//   W_q, R_q, bw_q, br_q, x_q: 量化后的值（仅在训练模式时有效，推理模式为空张量）
//   输入量化 mask:
//   x_mask: [T, B, I] 输入序列量化 mask
//   h0_mask: [B, H] 初始隐状态量化 mask
//   W_mask: [I, H*3] 输入权重量化 mask
//   R_mask: [H, H*3] 循环权重量化 mask
//   bw_mask: [H*3] 输入偏置量化 mask
//   br_mask: [H*3] 循环偏置量化 mask
//   计算过程 mask:
//   weight_ih_linear_mask: [T, B, H*3] QAT mask
//   weight_hh_linear_mask: [T, B, H*3] QAT mask
//   gate_input_mask: [T, B, H*3] 门输入 clamp mask
//   gate_output_mask: [T, B, H*3] 门输出 clamp mask
//   h_mask: [T, B, H] QAT mask
std::tuple<torch::Tensor, torch::Tensor, 
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
forward_quant_float_storage_wrapper(
    bool is_training,  // 是否开启训练模式
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W, const torch::Tensor &R, const torch::Tensor &bw,
    const torch::Tensor &br, const torch::Tensor &x,
    const torch::Tensor &h0,  // 初始隐藏状态，可以为空张量
    const GRUQuantParamsPy &quant_params) {
    
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bw.is_cuda() && bw.dtype() == torch::kFloat32, "bw must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32 tensor");

    // h0 可以为空张量（未提供初始状态）
    const float *h0_ptr = nullptr;
    if (h0.defined() && h0.numel() > 0) {
        TORCH_CHECK(h0.is_cuda() && h0.dtype() == torch::kFloat32,
                    "h0 must be CUDA float32 tensor");
        TORCH_CHECK(h0.sizes() == torch::IntArrayRef({batch_size, hidden_size}),
                    "h0 must have shape [batch_size, hidden_size]");
        h0_ptr = h0.data_ptr<float>();
    }

    // 确保 cublas handle 已初始化
    if (g_blas_handle == nullptr) {
        init_gru_cublas(g_blas_handle);
    }

    // 创建输出张量
    auto h = torch::empty({time_steps + 1, batch_size, hidden_size},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto v = torch::empty({time_steps, batch_size, hidden_size * 4},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // 创建量化值输出张量（训练和推理模式都需要分配，因为 quantGRUForwardFP 要求这些指针非空）
    const int hidden3 = hidden_size * 3;
    torch::Tensor W_q, R_q, bw_q, br_q, x_q;
    float *W_q_ptr = nullptr, *R_q_ptr = nullptr, *bw_q_ptr = nullptr, *br_q_ptr = nullptr, *x_q_ptr = nullptr;
    
    // 无论训练还是推理模式，都需要分配内存（quantGRUForwardFP 要求这些指针非空）
    W_q = torch::empty({input_size, hidden3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    R_q = torch::empty({hidden_size, hidden3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    bw_q = torch::empty({hidden3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    br_q = torch::empty({hidden3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    x_q = torch::empty({time_steps, batch_size, input_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    W_q_ptr = W_q.data_ptr<float>();
    R_q_ptr = R_q.data_ptr<float>();
    bw_q_ptr = bw_q.data_ptr<float>();
    br_q_ptr = br_q.data_ptr<float>();
    x_q_ptr = x_q.data_ptr<float>();

    // 创建 QAT mask 张量（训练模式时分配，否则为空张量）
    // 输入量化 mask
    torch::Tensor x_mask_tensor, h0_mask_tensor, W_mask_tensor, R_mask_tensor, bw_mask_tensor, br_mask_tensor;
    // 计算过程 mask
    torch::Tensor weight_ih_linear_mask, weight_hh_linear_mask, gate_input_mask, gate_output_mask, h_mask;
    
    // mask 指针
    uint8_t *x_mask_ptr = nullptr, *h0_mask_ptr = nullptr;
    uint8_t *W_mask_ptr = nullptr, *R_mask_ptr = nullptr;
    uint8_t *bw_mask_ptr = nullptr, *br_mask_ptr = nullptr;
    uint8_t *weight_ih_mask_ptr = nullptr, *weight_hh_mask_ptr = nullptr;
    uint8_t *gate_input_mask_ptr = nullptr, *gate_output_mask_ptr = nullptr, *h_mask_ptr = nullptr;
    
    if (is_training) {
        // 输入量化 mask
        x_mask_tensor = torch::empty({time_steps, batch_size, input_size},
                                     torch::dtype(torch::kUInt8).device(torch::kCUDA));
        h0_mask_tensor = (h0_ptr != nullptr) 
            ? torch::empty({batch_size, hidden_size}, torch::dtype(torch::kUInt8).device(torch::kCUDA))
            : torch::empty({0}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
        W_mask_tensor = torch::empty({input_size, hidden3},
                                     torch::dtype(torch::kUInt8).device(torch::kCUDA));
        R_mask_tensor = torch::empty({hidden_size, hidden3},
                                     torch::dtype(torch::kUInt8).device(torch::kCUDA));
        bw_mask_tensor = torch::empty({hidden3},
                                      torch::dtype(torch::kUInt8).device(torch::kCUDA));
        br_mask_tensor = torch::empty({hidden3},
                                      torch::dtype(torch::kUInt8).device(torch::kCUDA));
        
        // 计算过程 mask
        weight_ih_linear_mask = torch::empty({time_steps, batch_size, hidden3},
                                             torch::dtype(torch::kUInt8).device(torch::kCUDA));
        weight_hh_linear_mask = torch::empty({time_steps, batch_size, hidden3},
                                             torch::dtype(torch::kUInt8).device(torch::kCUDA));
        gate_input_mask = torch::empty({time_steps, batch_size, hidden3},
                                       torch::dtype(torch::kUInt8).device(torch::kCUDA));
        gate_output_mask = torch::empty({time_steps, batch_size, hidden3},
                                        torch::dtype(torch::kUInt8).device(torch::kCUDA));
        h_mask = torch::empty({time_steps, batch_size, hidden_size},
                              torch::dtype(torch::kUInt8).device(torch::kCUDA));
        
        // 获取指针
        x_mask_ptr = x_mask_tensor.data_ptr<uint8_t>();
        h0_mask_ptr = (h0_ptr != nullptr) ? h0_mask_tensor.data_ptr<uint8_t>() : nullptr;
        W_mask_ptr = W_mask_tensor.data_ptr<uint8_t>();
        R_mask_ptr = R_mask_tensor.data_ptr<uint8_t>();
        bw_mask_ptr = bw_mask_tensor.data_ptr<uint8_t>();
        br_mask_ptr = br_mask_tensor.data_ptr<uint8_t>();
        weight_ih_mask_ptr = weight_ih_linear_mask.data_ptr<uint8_t>();
        weight_hh_mask_ptr = weight_hh_linear_mask.data_ptr<uint8_t>();
        gate_input_mask_ptr = gate_input_mask.data_ptr<uint8_t>();
        gate_output_mask_ptr = gate_output_mask.data_ptr<uint8_t>();
        h_mask_ptr = h_mask.data_ptr<uint8_t>();
    } else {
        // 返回空张量（numel() == 0）
        auto empty_mask = torch::empty({0}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
        x_mask_tensor = empty_mask.clone();
        h0_mask_tensor = empty_mask.clone();
        W_mask_tensor = empty_mask.clone();
        R_mask_tensor = empty_mask.clone();
        bw_mask_tensor = empty_mask.clone();
        br_mask_tensor = empty_mask.clone();
        weight_ih_linear_mask = empty_mask.clone();
        weight_hh_linear_mask = empty_mask.clone();
        gate_input_mask = empty_mask.clone();
        gate_output_mask = empty_mask.clone();
        h_mask = empty_mask.clone();
    }

    // 转换量化参数
    GRUQuantParams cpp_params = quant_params.to_cpp();
    
    // 调用量化前向传播，直接写入量化值输出指针（零拷贝）
    quantGRUForwardFP(is_training, time_steps, batch_size, input_size, hidden_size,
                      W.data_ptr<float>(), R.data_ptr<float>(), 
                      bw.data_ptr<float>(), br.data_ptr<float>(), 
                      x.data_ptr<float>(), h0_ptr, cpp_params, g_blas_handle,
                      h.data_ptr<float>(), v.data_ptr<float>(),
                      W_q_ptr, R_q_ptr, bw_q_ptr, br_q_ptr, x_q_ptr,
                      x_mask_ptr, h0_mask_ptr, W_mask_ptr, R_mask_ptr, bw_mask_ptr, br_mask_ptr,
                      weight_ih_mask_ptr, weight_hh_mask_ptr, gate_input_mask_ptr, gate_output_mask_ptr, h_mask_ptr);

    return std::make_tuple(h, v, W_q, R_q, bw_q, br_q, x_q,
                           x_mask_tensor, h0_mask_tensor, W_mask_tensor, R_mask_tensor, bw_mask_tensor, br_mask_tensor,
                           weight_ih_linear_mask, weight_hh_linear_mask, gate_input_mask, gate_output_mask, h_mask);
}

// =====================================================================
// forward_quant_int_storage_wrapper: 量化前向传播（int32 存储）
// =====================================================================
// 与 forward_quant_float_storage_wrapper 的区别：
//   - 量化值 W_q/R_q/bw_q/br_q/x_q 使用 torch.int32 存储（内容为定点整数）
//   - 内部走 quantGRUForwardInt -> quantGRUForwardInt32 纯定点计算
//   - h/v 仍反量化为 float32 返回，供 autograd 使用
// 返回协议与 float 版本一致：
//   (h, v, W_q, R_q, bw_q, br_q, x_q, ...masks)
std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward_quant_int_storage_wrapper(
    bool is_training,  // 是否开启训练模式
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W, const torch::Tensor &R, const torch::Tensor &bw,
    const torch::Tensor &br, const torch::Tensor &x,
    const torch::Tensor &h0,  // 初始隐藏状态，可以为空张量
    const GRUQuantParamsPy &quant_params) {

    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bw.is_cuda() && bw.dtype() == torch::kFloat32, "bw must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32 tensor");

    // h0 可以为空张量（未提供初始状态）
    const float *h0_ptr = nullptr;
    if (h0.defined() && h0.numel() > 0) {
        TORCH_CHECK(h0.is_cuda() && h0.dtype() == torch::kFloat32,
                    "h0 must be CUDA float32 tensor");
        TORCH_CHECK(h0.sizes() == torch::IntArrayRef({batch_size, hidden_size}),
                    "h0 must have shape [batch_size, hidden_size]");
        h0_ptr = h0.data_ptr<float>();
    }

    // 确保 cublas handle 已初始化
    if (g_blas_handle == nullptr) {
        init_gru_cublas(g_blas_handle);
    }

    // 创建输出张量（h/v 仍是 float32，反量化后返回给 autograd）
    auto h = torch::empty({time_steps + 1, batch_size, hidden_size},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto v = torch::empty({time_steps, batch_size, hidden_size * 4},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // 创建 int32 量化值输出张量（训练和推理都分配，由 quantGRUForwardInt 写入）
    const int hidden3 = hidden_size * 3;
    auto W_q = torch::empty({input_size, hidden3}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto R_q = torch::empty({hidden_size, hidden3}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto bw_q = torch::empty({hidden3}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto br_q = torch::empty({hidden3}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto x_q = torch::empty({time_steps, batch_size, input_size}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    // 创建 QAT mask 张量（训练模式时分配，否则为空张量）
    torch::Tensor x_mask_tensor, h0_mask_tensor, W_mask_tensor, R_mask_tensor, bw_mask_tensor, br_mask_tensor;
    torch::Tensor weight_ih_linear_mask, weight_hh_linear_mask, gate_input_mask, gate_output_mask, h_mask;

    uint8_t *x_mask_ptr = nullptr, *h0_mask_ptr = nullptr;
    uint8_t *W_mask_ptr = nullptr, *R_mask_ptr = nullptr;
    uint8_t *bw_mask_ptr = nullptr, *br_mask_ptr = nullptr;
    uint8_t *weight_ih_mask_ptr = nullptr, *weight_hh_mask_ptr = nullptr;
    uint8_t *gate_input_mask_ptr = nullptr, *gate_output_mask_ptr = nullptr, *h_mask_ptr = nullptr;

    if (is_training) {
        x_mask_tensor = torch::empty({time_steps, batch_size, input_size},
                                     torch::dtype(torch::kUInt8).device(torch::kCUDA));
        h0_mask_tensor = (h0_ptr != nullptr)
            ? torch::empty({batch_size, hidden_size}, torch::dtype(torch::kUInt8).device(torch::kCUDA))
            : torch::empty({0}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
        W_mask_tensor = torch::empty({input_size, hidden3},
                                     torch::dtype(torch::kUInt8).device(torch::kCUDA));
        R_mask_tensor = torch::empty({hidden_size, hidden3},
                                     torch::dtype(torch::kUInt8).device(torch::kCUDA));
        bw_mask_tensor = torch::empty({hidden3},
                                      torch::dtype(torch::kUInt8).device(torch::kCUDA));
        br_mask_tensor = torch::empty({hidden3},
                                      torch::dtype(torch::kUInt8).device(torch::kCUDA));

        weight_ih_linear_mask = torch::empty({time_steps, batch_size, hidden3},
                                             torch::dtype(torch::kUInt8).device(torch::kCUDA));
        weight_hh_linear_mask = torch::empty({time_steps, batch_size, hidden3},
                                             torch::dtype(torch::kUInt8).device(torch::kCUDA));
        gate_input_mask = torch::empty({time_steps, batch_size, hidden3},
                                       torch::dtype(torch::kUInt8).device(torch::kCUDA));
        gate_output_mask = torch::empty({time_steps, batch_size, hidden3},
                                        torch::dtype(torch::kUInt8).device(torch::kCUDA));
        h_mask = torch::empty({time_steps, batch_size, hidden_size},
                              torch::dtype(torch::kUInt8).device(torch::kCUDA));

        x_mask_ptr = x_mask_tensor.data_ptr<uint8_t>();
        h0_mask_ptr = (h0_ptr != nullptr) ? h0_mask_tensor.data_ptr<uint8_t>() : nullptr;
        W_mask_ptr = W_mask_tensor.data_ptr<uint8_t>();
        R_mask_ptr = R_mask_tensor.data_ptr<uint8_t>();
        bw_mask_ptr = bw_mask_tensor.data_ptr<uint8_t>();
        br_mask_ptr = br_mask_tensor.data_ptr<uint8_t>();
        weight_ih_mask_ptr = weight_ih_linear_mask.data_ptr<uint8_t>();
        weight_hh_mask_ptr = weight_hh_linear_mask.data_ptr<uint8_t>();
        gate_input_mask_ptr = gate_input_mask.data_ptr<uint8_t>();
        gate_output_mask_ptr = gate_output_mask.data_ptr<uint8_t>();
        h_mask_ptr = h_mask.data_ptr<uint8_t>();
    } else {
        auto empty_mask = torch::empty({0}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
        x_mask_tensor = empty_mask.clone();
        h0_mask_tensor = empty_mask.clone();
        W_mask_tensor = empty_mask.clone();
        R_mask_tensor = empty_mask.clone();
        bw_mask_tensor = empty_mask.clone();
        br_mask_tensor = empty_mask.clone();
        weight_ih_linear_mask = empty_mask.clone();
        weight_hh_linear_mask = empty_mask.clone();
        gate_input_mask = empty_mask.clone();
        gate_output_mask = empty_mask.clone();
        h_mask = empty_mask.clone();
    }

    // 转换量化参数
    GRUQuantParams cpp_params = quant_params.to_cpp();

    // 调用整数存储量化前向传播：内部量化并写入 int32 q 输出，h/v 反量化为 float
    quantGRUForwardInt(is_training, time_steps, batch_size, input_size, hidden_size,
                       W.data_ptr<float>(), R.data_ptr<float>(),
                       bw.data_ptr<float>(), br.data_ptr<float>(),
                       x.data_ptr<float>(), h0_ptr, cpp_params, g_blas_handle,
                       h.data_ptr<float>(), v.data_ptr<float>(),
                       x_mask_ptr, h0_mask_ptr, W_mask_ptr, R_mask_ptr, bw_mask_ptr, br_mask_ptr,
                       weight_ih_mask_ptr, weight_hh_mask_ptr, gate_input_mask_ptr, gate_output_mask_ptr, h_mask_ptr,
                       W_q.data_ptr<int32_t>(), R_q.data_ptr<int32_t>(),
                       bw_q.data_ptr<int32_t>(), br_q.data_ptr<int32_t>(), x_q.data_ptr<int32_t>());

    return std::make_tuple(h, v, W_q, R_q, bw_q, br_q, x_q,
                           x_mask_tensor, h0_mask_tensor, W_mask_tensor, R_mask_tensor, bw_mask_tensor, br_mask_tensor,
                           weight_ih_linear_mask, weight_hh_linear_mask, gate_input_mask, gate_output_mask, h_mask);
}

// =====================================================================
// forward_quantized_int_io_wrapper: 纯定点 int 进 int 出推理前向
// =====================================================================
// 与 forward_quantized_wrapper 区别：输入 x_q / h0_q 已是 int32 量化值
// （上游层在同一 scale_x/zp_x、scale_h/zp_h 网格产生），内部不再量化输入，
// 仅量化权重，直出 int32 h_q。用于打通 AIMET INT16_FIXED_EVAL 纯定点链路。
// 输入:
//   W, R, bw, br: float master 权重/偏置
//   x_q: [T, B, I] int32（已量化输入）
//   h0_q: [B, H] int32（已量化初始状态），可为空张量
// 返回:
//   (output_q[T,B,H] int32, h_n_q[1,B,H] int32)
std::tuple<torch::Tensor, torch::Tensor>
forward_quantized_int_io_wrapper(
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W, const torch::Tensor &R, const torch::Tensor &bw,
    const torch::Tensor &br, const torch::Tensor &x_q,
    const torch::Tensor &h0_q,  // 已量化初始隐藏状态，可以为空张量
    const GRUQuantParamsPy &quant_params) {

    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bw.is_cuda() && bw.dtype() == torch::kFloat32, "bw must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(x_q.is_cuda() && x_q.dtype() == torch::kInt32, "x_q must be CUDA int32 tensor");
    TORCH_CHECK(x_q.sizes() == torch::IntArrayRef({time_steps, batch_size, input_size}),
                "x_q must have shape [time_steps, batch_size, input_size]");

    // h0_q 可以为空张量（未提供初始状态）
    const int32_t *h0_q_ptr = nullptr;
    if (h0_q.defined() && h0_q.numel() > 0) {
        TORCH_CHECK(h0_q.is_cuda() && h0_q.dtype() == torch::kInt32,
                    "h0_q must be CUDA int32 tensor");
        TORCH_CHECK(h0_q.sizes() == torch::IntArrayRef({batch_size, hidden_size}),
                    "h0_q must have shape [batch_size, hidden_size]");
        h0_q_ptr = h0_q.data_ptr<int32_t>();
    }

    // 确保 cublas handle 已初始化
    if (g_blas_handle == nullptr) {
        init_gru_cublas(g_blas_handle);
    }

    // int32 隐藏状态量化值输出 [(T+1), B, H]
    auto h_q = torch::empty({time_steps + 1, batch_size, hidden_size},
                            torch::dtype(torch::kInt32).device(torch::kCUDA));

    GRUQuantParams cpp_params = quant_params.to_cpp();

    quantGRUForwardIntIO(/*is_training=*/false, time_steps, batch_size, input_size, hidden_size,
                         W.data_ptr<float>(), R.data_ptr<float>(),
                         bw.data_ptr<float>(), br.data_ptr<float>(),
                         x_q.data_ptr<int32_t>(), h0_q_ptr,
                         cpp_params, g_blas_handle,
                         h_q.data_ptr<int32_t>(), /*v_q=*/nullptr);

    auto output_q = h_q.slice(0, 1, time_steps + 1).contiguous();      // [T, B, H]
    auto h_n_q = h_q.slice(0, time_steps, time_steps + 1).contiguous(); // [1, B, H]

    return std::make_tuple(output_q, h_n_q);
}

// =====================================================================
// forward_fp_wrapper: 浮点前向传播（训练/推理）
// =====================================================================
// 返回: (h, v)
//   h: [T+1, B, H] 隐藏状态序列（包含初始状态）
//   v: [T, B, H*4] 中间值（训练时需要，推理时可忽略）
std::tuple<torch::Tensor, torch::Tensor> 
forward_fp_wrapper(
    bool is_training,  // 是否开启训练模式
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W, const torch::Tensor &R, const torch::Tensor &bw,
    const torch::Tensor &br, const torch::Tensor &x,
    const torch::Tensor &h0) {  // 初始隐藏状态，可以为空张量
    
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bw.is_cuda() && bw.dtype() == torch::kFloat32, "bw must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32 tensor");

    // h0 可以为空张量（未提供初始状态）
    const float *h0_ptr = nullptr;
    if (h0.defined() && h0.numel() > 0) {
        TORCH_CHECK(h0.is_cuda() && h0.dtype() == torch::kFloat32,
                    "h0 must be CUDA float32 tensor");
        TORCH_CHECK(h0.sizes() == torch::IntArrayRef({batch_size, hidden_size}),
                    "h0 must have shape [batch_size, hidden_size]");
        h0_ptr = h0.data_ptr<float>();
    }

    // 确保 cublas handle 已初始化
    if (g_blas_handle == nullptr) {
        init_gru_cublas(g_blas_handle);
    }

    // 创建输出张量
    auto h = torch::empty({time_steps + 1, batch_size, hidden_size},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto v = torch::empty({time_steps, batch_size, hidden_size * 4},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // 调用浮点前向传播（直接调用 hasteGRUForward）
    hasteGRUForward(is_training, time_steps, batch_size, input_size, hidden_size,
                    W.data_ptr<float>(), R.data_ptr<float>(), 
                    bw.data_ptr<float>(), br.data_ptr<float>(), 
                    x.data_ptr<float>(), h0_ptr, g_blas_handle,
                    h.data_ptr<float>(), v.data_ptr<float>());

    return std::make_tuple(h, v);
}

// =====================================================================
// forward_calibrate: 校准前向传播（统一接口）
// =====================================================================
// 返回: (h, v)
//   h: [T+1, B, H] 隐藏状态序列
//   v: [T, B, H*4] 中间值
//
// 校准数据通过指针参数原地累积更新：
// - MINMAX: quant_ranges 被原地更新
// - SQNR/Percentile: hist_collectors 被原地更新
std::tuple<torch::Tensor, torch::Tensor> forward_calibrate_wrapper(
    bool is_training,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W, const torch::Tensor &R, const torch::Tensor &bw,
    const torch::Tensor &br, const torch::Tensor &x,
    const torch::Tensor &h0,
    const std::string &calib_method_str,  // 校准方法: 'minmax', 'sqnr', 'percentile'
    const OperatorQuantConfigPy &bitwidth_config,  // 位宽配置（必须）
    GRUQuantizationRangesPy *quant_ranges = nullptr,      // MINMAX 需要
    GRUHistogramCollectorsPy *hist_collectors = nullptr) {  // SQNR/Percentile 需要
    
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bw.is_cuda() && bw.dtype() == torch::kFloat32, "bw must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32 tensor");

    // 字符串转枚举
    CalibrationMethod calib_method = stringToCalibMethod(calib_method_str);
    
    // 校准模式不能是 NONE
    TORCH_CHECK(calib_method != CalibrationMethod::NONE,
                "forward_calibrate requires a calibration method ('minmax', 'sqnr', 'percentile')");
    
    // 验证校准参数
    if (calib_method == CalibrationMethod::MINMAX) {
        TORCH_CHECK(quant_ranges != nullptr, 
                    "quant_ranges is required for MINMAX calibration");
    } else {
        TORCH_CHECK(hist_collectors != nullptr, 
                    "hist_collectors is required for SQNR/Percentile calibration");
    }

    // h0 可以为空张量
    const float *h0_ptr = nullptr;
    if (h0.defined() && h0.numel() > 0) {
        TORCH_CHECK(h0.is_cuda() && h0.dtype() == torch::kFloat32,
                    "h0 must be CUDA float32 tensor");
        TORCH_CHECK(h0.sizes() == torch::IntArrayRef({batch_size, hidden_size}),
                    "h0 must have shape [batch_size, hidden_size]");
        h0_ptr = h0.data_ptr<float>();
    }

    // 确保 cublas handle 已初始化
    if (g_blas_handle == nullptr) {
        init_gru_cublas(g_blas_handle);
    }

    // 创建输出张量
    auto h = torch::empty({time_steps + 1, batch_size, hidden_size},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto v = torch::empty({time_steps, batch_size, hidden_size * 4},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // 转换位宽配置为 C++ 对象
    OperatorQuantConfig cpp_bitwidth_config = bitwidth_config.to_cpp();
    
    // 统一调用校准前向传播
    forwardWithCalibrationGPU(
        is_training, time_steps, batch_size, input_size, hidden_size,
        W.data_ptr<float>(), R.data_ptr<float>(), 
        bw.data_ptr<float>(), br.data_ptr<float>(), 
        x.data_ptr<float>(), h0_ptr, g_blas_handle,
        calib_method,
        quant_ranges ? &(quant_ranges->cpp_ranges) : nullptr,
        hist_collectors ? &(hist_collectors->gpu_collectors) : nullptr,
        cpp_bitwidth_config,
        h.data_ptr<float>(), v.data_ptr<float>());

    return std::make_tuple(h, v);
}



// =====================================================================
// backward_quant_wrapper: 量化反向传播（使用保存的量化值）
// =====================================================================
// 输入: W_q, R_q, bw_q, br_q, x_q 是前向传播保存的量化值
// 函数会直接反量化这些值，然后进行反向传播
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
backward_quant_float_storage_wrapper(
    int time_steps, int batch_size, int input_size, int hidden_size,
    torch::Tensor &W_q, torch::Tensor &R_q, torch::Tensor &bw_q,
    torch::Tensor &br_q, torch::Tensor &x_q,
    const torch::Tensor &dh_new,
    const torch::Tensor &h,
    const torch::Tensor &v,
    const GRUQuantParamsPy &quant_params,
    // QAT masks（仅在训练模式时有效）
    const torch::Tensor &x_mask,
    const torch::Tensor &h0_mask,
    const torch::Tensor &W_mask,
    const torch::Tensor &R_mask,
    const torch::Tensor &bw_mask,
    const torch::Tensor &br_mask,
    const torch::Tensor &weight_ih_linear_mask,
    const torch::Tensor &weight_hh_linear_mask,
    const torch::Tensor &gate_input_mask,
    const torch::Tensor &gate_output_mask,
    const torch::Tensor &h_mask) {

    // 检查输入张量的类型和设备
    TORCH_CHECK(W_q.is_cuda() && W_q.dtype() == torch::kFloat32, "W_q must be CUDA float32 tensor");
    TORCH_CHECK(R_q.is_cuda() && R_q.dtype() == torch::kFloat32, "R_q must be CUDA float32 tensor");
    TORCH_CHECK(bw_q.is_cuda() && bw_q.dtype() == torch::kFloat32, "bw_q must be CUDA float32 tensor");
    TORCH_CHECK(br_q.is_cuda() && br_q.dtype() == torch::kFloat32, "br_q must be CUDA float32 tensor");
    TORCH_CHECK(x_q.is_cuda() && x_q.dtype() == torch::kFloat32, "x_q must be CUDA float32 tensor");
    TORCH_CHECK(dh_new.is_cuda() && dh_new.dtype() == torch::kFloat32,
                "dh_new must be CUDA float32 tensor");
    TORCH_CHECK(h.is_cuda() && h.dtype() == torch::kFloat32, "h must be CUDA float32 tensor");
    TORCH_CHECK(v.is_cuda() && v.dtype() == torch::kFloat32, "v must be CUDA float32 tensor");

    // 检查张量形状
    const int hidden3 = hidden_size * 3;
    TORCH_CHECK(W_q.sizes() == torch::IntArrayRef({input_size, hidden3}),
                "W_q must have shape [input_size, hidden_size * 3]");
    TORCH_CHECK(R_q.sizes() == torch::IntArrayRef({hidden_size, hidden3}),
                "R_q must have shape [hidden_size, hidden_size * 3]");
    TORCH_CHECK(bw_q.sizes() == torch::IntArrayRef({hidden3}),
                "bw_q must have shape [hidden_size * 3]");
    TORCH_CHECK(br_q.sizes() == torch::IntArrayRef({hidden3}),
                "br_q must have shape [hidden_size * 3]");
    TORCH_CHECK(x_q.sizes() == torch::IntArrayRef({time_steps, batch_size, input_size}),
                "x_q must have shape [time_steps, batch_size, input_size]");
    TORCH_CHECK(dh_new.sizes() == torch::IntArrayRef({time_steps + 1, batch_size, hidden_size}),
                "dh_new must have shape [time_steps + 1, batch_size, hidden_size]");
    TORCH_CHECK(h.sizes() == torch::IntArrayRef({time_steps + 1, batch_size, hidden_size}),
                "h must have shape [time_steps + 1, batch_size, hidden_size]");
    TORCH_CHECK(v.sizes() == torch::IntArrayRef({time_steps, batch_size, hidden_size * 4}),
                "v must have shape [time_steps, batch_size, hidden_size * 4]");

    // 转换量化参数
    GRUQuantParams cpp_params = quant_params.to_cpp();

    // 原地反量化（直接修改保存的量化值）
    // 注意：这些量化值是从 forward 保存的中间变量，在 backward 中只使用一次
    // backward 完成后这些 Tensor 会被 PyTorch 自动释放，不需要保护原始数据
    // 使用统一接口，内部根据 granularity 自动选择 per-tensor、per-gate 或 per-channel 反量化
    dequantizeGRUWeights(
        W_q.data_ptr<float>(), R_q.data_ptr<float>(),
        bw_q.data_ptr<float>(), br_q.data_ptr<float>(),
        input_size, hidden_size, cpp_params);
    
    const std::size_t x_size = time_steps * batch_size * input_size;
    dev::dequantificationFPInplace(x_q.data_ptr<float>(), x_size,
                                   toFixedScale(cpp_params.x_, cpp_params.bitwidth_config_.usePOT2_),
                                   cpp_params.x_.zero_point);

    // 同步 CUDA 操作
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in backward_quant_wrapper dequant: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in backward_quant_wrapper dequant: ") + err_str);
    }

    // 转置操作（使用反量化后的值）
    torch::Tensor x_t = x_q.permute({2, 0, 1}).contiguous();  // [T,B,I] -> [I,T,B]
    torch::Tensor W_t = W_q.t().contiguous();  // [C, H*3] -> [H*3, C]
    torch::Tensor R_t = R_q.t().contiguous();  // [H, H*3] -> [H*3, H]

    // 确保 cublas handle 已初始化
    if (g_blas_handle == nullptr) {
        init_gru_cublas(g_blas_handle);
    }

    // 创建输出张量
    auto dx = torch::empty({time_steps, batch_size, input_size},
                           torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dW = torch::zeros({input_size, hidden_size * 3},
                           torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dR = torch::zeros({hidden_size, hidden_size * 3},
                           torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dbw = torch::zeros({hidden_size * 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dbr = torch::zeros({hidden_size * 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dh = torch::zeros({batch_size, hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // 获取 mask 指针（空 tensor 时为 nullptr）
    auto get_mask_ptr = [](const torch::Tensor &t) -> const uint8_t* {
        return t.numel() > 0 ? t.data_ptr<uint8_t>() : nullptr;
    };

    // 调用 C++ 量化反向接口（使用反量化后的转置值）
    quantGRUBackward(time_steps, batch_size, input_size, hidden_size,
                     W_t.data_ptr<float>(),  // [H*3, C] - 转置后的 W（已反量化）
                     R_t.data_ptr<float>(),  // [H*3, H] - 转置后的 R（已反量化）
                     bw_q.data_ptr<float>(), br_q.data_ptr<float>(),  // 已反量化
                     x_t.data_ptr<float>(),  // [I, T, B] - 转置后的 x（已反量化）
                     dh_new.data_ptr<float>(), h.data_ptr<float>(), v.data_ptr<float>(),
                     g_blas_handle,
                     dx.data_ptr<float>(), dW.data_ptr<float>(),
                     dR.data_ptr<float>(), dbw.data_ptr<float>(), dbr.data_ptr<float>(),
                     dh.data_ptr<float>(),
                     // 量化相关参数（用于 mask 生成，不用于 rescale 补偿）
                     &cpp_params,
                     // QAT masks
                     get_mask_ptr(x_mask),
                     get_mask_ptr(h0_mask),
                     get_mask_ptr(W_mask),
                     get_mask_ptr(R_mask),
                     get_mask_ptr(bw_mask),
                     get_mask_ptr(br_mask),
                     get_mask_ptr(weight_ih_linear_mask),
                     get_mask_ptr(weight_hh_linear_mask),
                     get_mask_ptr(gate_input_mask),
                     get_mask_ptr(gate_output_mask),
                     get_mask_ptr(h_mask));

    return std::make_tuple(dx, dW, dR, dbw, dbr, dh);
}

// =====================================================================
// backward_quant_int_storage_wrapper: 量化反向传播（int32 存储）
// =====================================================================
// 输入 W_q/R_q/bw_q/br_q/x_q 为 forward 保存的 int32 量化值。
// 先转换为 float32 q 值，然后复用 float 存储版的反量化 + 反向逻辑。
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
backward_quant_int_storage_wrapper(
    int time_steps, int batch_size, int input_size, int hidden_size,
    torch::Tensor &W_q, torch::Tensor &R_q, torch::Tensor &bw_q,
    torch::Tensor &br_q, torch::Tensor &x_q,
    const torch::Tensor &dh_new,
    const torch::Tensor &h,
    const torch::Tensor &v,
    const GRUQuantParamsPy &quant_params,
    // QAT masks（仅在训练模式时有效）
    const torch::Tensor &x_mask,
    const torch::Tensor &h0_mask,
    const torch::Tensor &W_mask,
    const torch::Tensor &R_mask,
    const torch::Tensor &bw_mask,
    const torch::Tensor &br_mask,
    const torch::Tensor &weight_ih_linear_mask,
    const torch::Tensor &weight_hh_linear_mask,
    const torch::Tensor &gate_input_mask,
    const torch::Tensor &gate_output_mask,
    const torch::Tensor &h_mask) {

    TORCH_CHECK(W_q.is_cuda() && W_q.dtype() == torch::kInt32, "W_q must be CUDA int32 tensor");
    TORCH_CHECK(R_q.is_cuda() && R_q.dtype() == torch::kInt32, "R_q must be CUDA int32 tensor");
    TORCH_CHECK(bw_q.is_cuda() && bw_q.dtype() == torch::kInt32, "bw_q must be CUDA int32 tensor");
    TORCH_CHECK(br_q.is_cuda() && br_q.dtype() == torch::kInt32, "br_q must be CUDA int32 tensor");
    TORCH_CHECK(x_q.is_cuda() && x_q.dtype() == torch::kInt32, "x_q must be CUDA int32 tensor");

    // int32 量化值 -> float32 量化值（内容仍是定点整数，dtype 转 float）
    // 后续 float 存储版会对这些 buffer 原地反量化，因此必须是独立可写副本。
    torch::Tensor W_q_f = W_q.to(torch::kFloat32);
    torch::Tensor R_q_f = R_q.to(torch::kFloat32);
    torch::Tensor bw_q_f = bw_q.to(torch::kFloat32);
    torch::Tensor br_q_f = br_q.to(torch::kFloat32);
    torch::Tensor x_q_f = x_q.to(torch::kFloat32);

    return backward_quant_float_storage_wrapper(
        time_steps, batch_size, input_size, hidden_size,
        W_q_f, R_q_f, bw_q_f, br_q_f, x_q_f,
        dh_new, h, v, quant_params,
        x_mask, h0_mask, W_mask, R_mask, bw_mask, br_mask,
        weight_ih_linear_mask, weight_hh_linear_mask, gate_input_mask, gate_output_mask, h_mask);
}

// =====================================================================
// backward_fp_wrapper: 浮点反向传播
// =====================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
backward_fp_wrapper(
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W, const torch::Tensor &R, const torch::Tensor &bw,
    const torch::Tensor &br, const torch::Tensor &x,
    const torch::Tensor &dh_new,
    const torch::Tensor &h,
    const torch::Tensor &v) {

    // 检查输入张量的类型和设备
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bw.is_cuda() && bw.dtype() == torch::kFloat32, "bw must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32 tensor");
    TORCH_CHECK(dh_new.is_cuda() && dh_new.dtype() == torch::kFloat32,
                "dh_new must be CUDA float32 tensor");
    TORCH_CHECK(h.is_cuda() && h.dtype() == torch::kFloat32, "h must be CUDA float32 tensor");
    TORCH_CHECK(v.is_cuda() && v.dtype() == torch::kFloat32, "v must be CUDA float32 tensor");

    // 检查张量形状
    TORCH_CHECK(x.sizes() == torch::IntArrayRef({time_steps, batch_size, input_size}),
                "x must have shape [time_steps, batch_size, input_size]");
    TORCH_CHECK(W.sizes() == torch::IntArrayRef({input_size, hidden_size * 3}),
                "W must have shape [input_size, hidden_size * 3]");
    TORCH_CHECK(R.sizes() == torch::IntArrayRef({hidden_size, hidden_size * 3}),
                "R must have shape [hidden_size, hidden_size * 3]");
    TORCH_CHECK(bw.sizes() == torch::IntArrayRef({hidden_size * 3}),
                "bw must have shape [hidden_size * 3]");
    TORCH_CHECK(br.sizes() == torch::IntArrayRef({hidden_size * 3}),
                "br must have shape [hidden_size * 3]");
    TORCH_CHECK(dh_new.sizes() == torch::IntArrayRef({time_steps + 1, batch_size, hidden_size}),
                "dh_new must have shape [time_steps + 1, batch_size, hidden_size]");
    TORCH_CHECK(h.sizes() == torch::IntArrayRef({time_steps + 1, batch_size, hidden_size}),
                "h must have shape [time_steps + 1, batch_size, hidden_size]");
    TORCH_CHECK(v.sizes() == torch::IntArrayRef({time_steps, batch_size, hidden_size * 4}),
                "v must have shape [time_steps, batch_size, hidden_size * 4]");

    // 转置操作
    torch::Tensor x_t = x.permute({2, 0, 1}).contiguous();  // [T,B,I] -> [I,T,B]
    torch::Tensor W_t = W.t().contiguous();  // [C, H*3] -> [H*3, C]
    torch::Tensor R_t = R.t().contiguous();  // [H, H*3] -> [H*3, H]

    // 确保 cublas handle 已初始化
    if (g_blas_handle == nullptr) {
        init_gru_cublas(g_blas_handle);
    }

    // 创建输出张量
    auto dx = torch::empty({time_steps, batch_size, input_size},
                           torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dW = torch::zeros({input_size, hidden_size * 3},
                           torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dR = torch::zeros({hidden_size, hidden_size * 3},
                           torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dbw = torch::zeros({hidden_size * 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dbr = torch::zeros({hidden_size * 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dh = torch::zeros({batch_size, hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // 调用 C++ 浮点反向接口（直接调用 hasteGRUBackward）
    hasteGRUBackward(time_steps, batch_size, input_size, hidden_size,
                     W_t.data_ptr<float>(),  // [H*3, C] - 转置后的 W
                     R_t.data_ptr<float>(),  // [H*3, H] - 转置后的 R
                     bw.data_ptr<float>(), br.data_ptr<float>(),
                     x_t.data_ptr<float>(),  // [I, T, B] - 转置后的 x
                     dh_new.data_ptr<float>(), h.data_ptr<float>(), v.data_ptr<float>(),
                     g_blas_handle,
                     dx.data_ptr<float>(), dW.data_ptr<float>(),
                     dR.data_ptr<float>(), dbw.data_ptr<float>(), dbr.data_ptr<float>(),
                     dh.data_ptr<float>());

    return std::make_tuple(dx, dW, dR, dbw, dbr, dh);
}

// ============================================================================
//                    OperatorQuantConfigPy 方法实现
// ============================================================================

// 默认构造函数：从 C++ OperatorQuantConfig 获取默认值
// 这样保证 Python 端和 C++ 端的默认值始终一致
OperatorQuantConfigPy::OperatorQuantConfigPy() {
    OperatorQuantConfig cpp_defaults;  // 使用 C++ 的默认值
    from_cpp(cpp_defaults);
}

// 转换为 C++ 结构体
OperatorQuantConfig OperatorQuantConfigPy::to_cpp() const {
    OperatorQuantConfig cfg;
    // 位宽配置（直接使用 is_unsigned 标志，与 C++ 语义一致）
    cfg.x_ = toBitwidth(x_, x_unsigned_);
    cfg.h_ = toBitwidth(h_, h_unsigned_);
    cfg.W_ = toBitwidth(W_, W_unsigned_);
    cfg.R_ = toBitwidth(R_, R_unsigned_);
    cfg.bw_ = toBitwidth(bw_, bw_unsigned_);
    cfg.br_ = toBitwidth(br_, br_unsigned_);
    cfg.weight_ih_linear_ = toBitwidth(weight_ih_linear_, weight_ih_linear_unsigned_);
    cfg.weight_hh_linear_ = toBitwidth(weight_hh_linear_, weight_hh_linear_unsigned_);
    cfg.update_gate_input_ = toBitwidth(update_gate_input_, update_gate_input_unsigned_);
    cfg.update_gate_output_ = toBitwidth(update_gate_output_, update_gate_output_unsigned_);
    cfg.reset_gate_input_ = toBitwidth(reset_gate_input_, reset_gate_input_unsigned_);
    cfg.reset_gate_output_ = toBitwidth(reset_gate_output_, reset_gate_output_unsigned_);
    cfg.new_gate_input_ = toBitwidth(new_gate_input_, new_gate_input_unsigned_);
    cfg.new_gate_output_ = toBitwidth(new_gate_output_, new_gate_output_unsigned_);
    cfg.mul_reset_hidden_ = toBitwidth(mul_reset_hidden_, mul_reset_hidden_unsigned_);
    cfg.mul_old_contribution_ = toBitwidth(mul_old_contribution_, mul_old_contribution_unsigned_);
    cfg.mul_new_contribution_ = toBitwidth(mul_new_contribution_, mul_new_contribution_unsigned_);
    // 对称量化配置
    cfg.x_symmetric_ = x_symmetric_;
    cfg.h_symmetric_ = h_symmetric_;
    cfg.W_symmetric_ = W_symmetric_;
    cfg.R_symmetric_ = R_symmetric_;
    cfg.bw_symmetric_ = bw_symmetric_;
    cfg.br_symmetric_ = br_symmetric_;
    cfg.weight_ih_linear_symmetric_ = weight_ih_linear_symmetric_;
    cfg.weight_hh_linear_symmetric_ = weight_hh_linear_symmetric_;
    cfg.update_gate_input_symmetric_ = update_gate_input_symmetric_;
    cfg.update_gate_output_symmetric_ = update_gate_output_symmetric_;
    cfg.reset_gate_input_symmetric_ = reset_gate_input_symmetric_;
    cfg.reset_gate_output_symmetric_ = reset_gate_output_symmetric_;
    cfg.new_gate_input_symmetric_ = new_gate_input_symmetric_;
    cfg.new_gate_output_symmetric_ = new_gate_output_symmetric_;
    cfg.mul_reset_hidden_symmetric_ = mul_reset_hidden_symmetric_;
    cfg.mul_old_contribution_symmetric_ = mul_old_contribution_symmetric_;
    cfg.mul_new_contribution_symmetric_ = mul_new_contribution_symmetric_;
    // 量化粒度配置
    cfg.W_granularity_ = static_cast<OperatorQuantConfig::QuantizationGranularity>(W_granularity_);
    cfg.R_granularity_ = static_cast<OperatorQuantConfig::QuantizationGranularity>(R_granularity_);
    cfg.bw_granularity_ = static_cast<OperatorQuantConfig::QuantizationGranularity>(bw_granularity_);
    cfg.br_granularity_ = static_cast<OperatorQuantConfig::QuantizationGranularity>(br_granularity_);
    cfg.usePOT2_ = usePOT2_;
    cfg.pot_scale_method_ = static_cast<PotScaleMethod>(pot_scale_method_);
    cfg.pot_scale_tolerance_ = pot_scale_tolerance_;
    return cfg;
}

// 从 C++ 结构体读取
void OperatorQuantConfigPy::from_cpp(const OperatorQuantConfig &cfg) {
    // 位宽配置
    x_ = cfg.x_.bits_;
    h_ = cfg.h_.bits_;
    W_ = cfg.W_.bits_;
    R_ = cfg.R_.bits_;
    bw_ = cfg.bw_.bits_;
    br_ = cfg.br_.bits_;
    weight_ih_linear_ = cfg.weight_ih_linear_.bits_;
    weight_hh_linear_ = cfg.weight_hh_linear_.bits_;
    update_gate_input_ = cfg.update_gate_input_.bits_;
    update_gate_output_ = cfg.update_gate_output_.bits_;
    reset_gate_input_ = cfg.reset_gate_input_.bits_;
    reset_gate_output_ = cfg.reset_gate_output_.bits_;
    new_gate_input_ = cfg.new_gate_input_.bits_;
    new_gate_output_ = cfg.new_gate_output_.bits_;
    mul_reset_hidden_ = cfg.mul_reset_hidden_.bits_;
    mul_old_contribution_ = cfg.mul_old_contribution_.bits_;
    mul_new_contribution_ = cfg.mul_new_contribution_.bits_;
    // 对称量化配置
    x_symmetric_ = cfg.x_symmetric_;
    h_symmetric_ = cfg.h_symmetric_;
    W_symmetric_ = cfg.W_symmetric_;
    R_symmetric_ = cfg.R_symmetric_;
    bw_symmetric_ = cfg.bw_symmetric_;
    br_symmetric_ = cfg.br_symmetric_;
    weight_ih_linear_symmetric_ = cfg.weight_ih_linear_symmetric_;
    weight_hh_linear_symmetric_ = cfg.weight_hh_linear_symmetric_;
    update_gate_input_symmetric_ = cfg.update_gate_input_symmetric_;
    update_gate_output_symmetric_ = cfg.update_gate_output_symmetric_;
    reset_gate_input_symmetric_ = cfg.reset_gate_input_symmetric_;
    reset_gate_output_symmetric_ = cfg.reset_gate_output_symmetric_;
    new_gate_input_symmetric_ = cfg.new_gate_input_symmetric_;
    new_gate_output_symmetric_ = cfg.new_gate_output_symmetric_;
    mul_reset_hidden_symmetric_ = cfg.mul_reset_hidden_symmetric_;
    mul_old_contribution_symmetric_ = cfg.mul_old_contribution_symmetric_;
    mul_new_contribution_symmetric_ = cfg.mul_new_contribution_symmetric_;
    // 无符号配置（直接从 C++ is_unsigned_ 读取，无需转换）
    x_unsigned_ = cfg.x_.is_unsigned_;
    h_unsigned_ = cfg.h_.is_unsigned_;
    W_unsigned_ = cfg.W_.is_unsigned_;
    R_unsigned_ = cfg.R_.is_unsigned_;
    bw_unsigned_ = cfg.bw_.is_unsigned_;
    br_unsigned_ = cfg.br_.is_unsigned_;
    weight_ih_linear_unsigned_ = cfg.weight_ih_linear_.is_unsigned_;
    weight_hh_linear_unsigned_ = cfg.weight_hh_linear_.is_unsigned_;
    update_gate_input_unsigned_ = cfg.update_gate_input_.is_unsigned_;
    update_gate_output_unsigned_ = cfg.update_gate_output_.is_unsigned_;
    reset_gate_input_unsigned_ = cfg.reset_gate_input_.is_unsigned_;
    reset_gate_output_unsigned_ = cfg.reset_gate_output_.is_unsigned_;
    new_gate_input_unsigned_ = cfg.new_gate_input_.is_unsigned_;
    new_gate_output_unsigned_ = cfg.new_gate_output_.is_unsigned_;
    mul_reset_hidden_unsigned_ = cfg.mul_reset_hidden_.is_unsigned_;
    mul_old_contribution_unsigned_ = cfg.mul_old_contribution_.is_unsigned_;
    mul_new_contribution_unsigned_ = cfg.mul_new_contribution_.is_unsigned_;
    // 量化粒度配置
    W_granularity_ = static_cast<int>(cfg.W_granularity_);
    R_granularity_ = static_cast<int>(cfg.R_granularity_);
    bw_granularity_ = static_cast<int>(cfg.bw_granularity_);
    br_granularity_ = static_cast<int>(cfg.br_granularity_);
    usePOT2_ = cfg.usePOT2_;
    pot_scale_method_ = static_cast<int>(cfg.pot_scale_method_);
    pot_scale_tolerance_ = cfg.pot_scale_tolerance_;
}

// ============================================================================
//                    GRUQuantParamsPy 方法实现
// ============================================================================

namespace {
inline float decode_scale_for_python(const FixedPointScale &fixed, int8_t shift) {
    if (fixed.multiplier != 0) return decode_scale(fixed);
    return exp2_scale(shift);
}

inline float prefer_raw_scale(float raw_scale, const FixedPointScale &fixed, int8_t shift) {
    if (raw_scale > 0.0f) {
        return raw_scale;
    }
    return decode_scale_for_python(fixed, shift);
}

inline int8_t encode_shift_from_scale(float scale) {
    if (!(scale > 0.0f)) {
        throw std::invalid_argument("scale must be > 0");
    }
    return static_cast<int8_t>(round_f(-std::log2(scale)));
}

inline FixedPointScale encode_fixed_from_scale(float scale, bool usePOT2) {
    if (usePOT2) {
        return FixedPointScale{1, encode_shift_from_scale(scale)};
    }
    return encodeMShift(scale);
}

}  // namespace

// 从 C++ 结构体转换
void GRUQuantParamsPy::from_cpp(const GRUQuantParams &cpp_params) {
    hidden_ = cpp_params.hidden_;
    // 基础参数：直读唯一权威 QuantParam
    scale_x_ = cpp_params.x_.scale;
    zp_x_ = cpp_params.x_.zero_point;
    scale_h_ = cpp_params.h_.scale;
    zp_h_ = cpp_params.h_.zero_point;

    // 权重/偏置：唯一 per-channel 权威数组（已按粒度广播，tensor/gate 视图由索引还原）
    auto read_channel = [](const ChannelQuantParam &c, std::vector<float> &out) {
        out.resize(c.channels.size());
        for (size_t i = 0; i < c.channels.size(); ++i) out[i] = c.channels[i].scale;
    };
    read_channel(cpp_params.W_, scale_W_);
    read_channel(cpp_params.R_, scale_R_);
    read_channel(cpp_params.bw_, scale_bw_);
    read_channel(cpp_params.br_, scale_br_);

    // Linear 输出参数
    scale_weight_ih_linear_ = cpp_params.weight_ih_linear_.scale;
    zp_weight_ih_linear_ = cpp_params.weight_ih_linear_.zero_point;
    scale_weight_hh_linear_ = cpp_params.weight_hh_linear_.scale;
    zp_weight_hh_linear_ = cpp_params.weight_hh_linear_.zero_point;
    // 门激活函数输入参数
    scale_update_gate_input_ = cpp_params.update_gate_input_.scale;
    zp_update_gate_input_ = cpp_params.update_gate_input_.zero_point;
    scale_reset_gate_input_ = cpp_params.reset_gate_input_.scale;
    zp_reset_gate_input_ = cpp_params.reset_gate_input_.zero_point;
    scale_new_gate_input_ = cpp_params.new_gate_input_.scale;
    zp_new_gate_input_ = cpp_params.new_gate_input_.zero_point;
    // 门激活函数输出参数
    scale_update_gate_output_ = cpp_params.update_gate_output_.scale;
    zp_update_gate_output_ = cpp_params.update_gate_output_.zero_point;
    scale_reset_gate_output_ = cpp_params.reset_gate_output_.scale;
    zp_reset_gate_output_ = cpp_params.reset_gate_output_.zero_point;
    scale_new_gate_output_ = cpp_params.new_gate_output_.scale;
    zp_new_gate_output_ = cpp_params.new_gate_output_.zero_point;
    // 中间计算参数
    scale_mul_reset_hidden_ = cpp_params.mul_reset_hidden_.scale;
    zp_mul_reset_hidden_ = cpp_params.mul_reset_hidden_.zero_point;
    // 隐状态更新参数
    scale_mul_new_contribution_ = cpp_params.mul_new_contribution_.scale;
    zp_mul_new_contribution_ = cpp_params.mul_new_contribution_.zero_point;
    scale_mul_old_contribution_ = cpp_params.mul_old_contribution_.scale;
    zp_mul_old_contribution_ = cpp_params.mul_old_contribution_.zero_point;

    // ⚠️ 关键：复制位宽配置
    bitwidth_config_.from_cpp(cpp_params.bitwidth_config_);
}

// 转换为 C++ 结构体
GRUQuantParams GRUQuantParamsPy::to_cpp() const {
    GRUQuantParams cpp_params;
    cpp_params.hidden_ = hidden_;
    // 基础参数：写入唯一权威 QuantParam
    cpp_params.x_ = QuantParam{scale_x_, zp_x_};
    cpp_params.h_ = QuantParam{scale_h_, zp_h_};

    // 权重/偏置：一律从唯一 per-channel 权威数组展开（已按粒度广播，权重零点取 0）。
    // granularity 仅作为元数据写回；scale 值不再依赖 tensor/gate 副本。
    auto fill_channel = [this](ChannelQuantParam &out, int gran,
                               const std::vector<float> &channel_scale, const char *name) {
        if (hidden_ <= 0) throw std::invalid_argument(std::string(name) + " hidden size must be > 0");
        const int channel_count = hidden_ * 3;
        if (static_cast<int>(channel_scale.size()) != channel_count) {
            throw std::invalid_argument(std::string(name) + " per-channel scale size mismatch");
        }
        out.hidden = hidden_;
        out.resize(channel_count);
        switch (gran) {
            case 0:  out.granularity = OperatorQuantConfig::PER_TENSOR;  break;
            case 1:  out.granularity = OperatorQuantConfig::PER_GATE;    break;
            default: out.granularity = OperatorQuantConfig::PER_CHANNEL; break;
        }
        for (int c = 0; c < channel_count; ++c) out.channel(c) = QuantParam{channel_scale[c], 0};
    };
    fill_channel(cpp_params.W_, bitwidth_config_.W_granularity_, scale_W_, "W");
    fill_channel(cpp_params.R_, bitwidth_config_.R_granularity_, scale_R_, "R");
    fill_channel(cpp_params.bw_, bitwidth_config_.bw_granularity_, scale_bw_, "bw");
    fill_channel(cpp_params.br_, bitwidth_config_.br_granularity_, scale_br_, "br");

    // Linear 输出参数
    cpp_params.weight_ih_linear_ = QuantParam{scale_weight_ih_linear_, zp_weight_ih_linear_};
    cpp_params.weight_hh_linear_ = QuantParam{scale_weight_hh_linear_, zp_weight_hh_linear_};
    // 门激活函数输入参数
    cpp_params.update_gate_input_ = QuantParam{scale_update_gate_input_, zp_update_gate_input_};
    cpp_params.reset_gate_input_ = QuantParam{scale_reset_gate_input_, zp_reset_gate_input_};
    cpp_params.new_gate_input_ = QuantParam{scale_new_gate_input_, zp_new_gate_input_};
    // 门激活函数输出参数
    cpp_params.update_gate_output_ = QuantParam{scale_update_gate_output_, zp_update_gate_output_};
    cpp_params.reset_gate_output_ = QuantParam{scale_reset_gate_output_, zp_reset_gate_output_};
    cpp_params.new_gate_output_ = QuantParam{scale_new_gate_output_, zp_new_gate_output_};
    // 中间计算参数
    cpp_params.mul_reset_hidden_ = QuantParam{scale_mul_reset_hidden_, zp_mul_reset_hidden_};
    // 隐状态更新参数
    cpp_params.mul_new_contribution_ = QuantParam{scale_mul_new_contribution_, zp_mul_new_contribution_};
    cpp_params.mul_old_contribution_ = QuantParam{scale_mul_old_contribution_, zp_mul_old_contribution_};

    // ⚠️ 关键：复制位宽配置
    cpp_params.bitwidth_config_ = bitwidth_config_.to_cpp();

    // 重新生成 LUT（因为 LUT 不能直接序列化到 Python）
    // 这会在每次 to_cpp() 时重新生成，但开销可以接受（只在 forward 前调用一次）
    generate_piecewise_linear_lut_to_params(cpp_params);

    return cpp_params;
}

// 返回 forward 实际使用的有效定点 scale。
// POT2 模式下 quant_params 中落盘的 scale 已是 po2_scale，此处 encode 近似恒等；
// Affine 模式下由连续 scale encodeMShift 得到 effective scale。
float decode_effective_scale_wrapper(float scale, bool usePOT2) {
    if (!(scale > 0.0f)) {
        throw std::invalid_argument("scale must be > 0");
    }
    FixedPointScale fixed = encode_fixed_from_scale(scale, usePOT2);
    int8_t shift = encode_shift_from_scale(scale);
    return decode_scale_for_python(fixed, shift);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GRU Interface Python Bindings";

    // 有效定点 scale 解码（供 get_io_quant_meta 返回 bit-exact 一致的 scale）
    m.def("decode_effective_scale", &decode_effective_scale_wrapper,
          "Decode the effective fixed-point scale actually used by the quantized "
          "kernel (POT2 or M16 encoded), given the raw continuous scale and "
          "usePOT2 flag. Ensures (q - zp) * scale matches forward_quantized output.",
          py::arg("scale"), py::arg("usePOT2"));

    // 初始化 cublas handle
    m.def("init_gru_cublas", &init_gru_cublas_wrapper, "Initialize cuBLAS handle for GRU");

    // GRUQuantizationRanges 绑定（通过 lambda 访问内部 C++ 对象的字段）
    // 宏定义简化属性绑定
    #define DEF_PROP(name) \
        .def_property(#name, \
            [](const GRUQuantizationRangesPy &self) { return self.cpp_ranges.name; }, \
            [](GRUQuantizationRangesPy &self, decltype(GRUQuantizationRanges::name) v) { self.cpp_ranges.name = v; })

    py::class_<GRUQuantizationRangesPy>(m, "GRUQuantizationRanges")
        .def(py::init<>())
        .def(py::init<int>(), py::arg("hidden"))
        DEF_PROP(hidden_)
        DEF_PROP(min_x_)
        DEF_PROP(max_x_)
        DEF_PROP(min_h_)
        DEF_PROP(max_h_)
        DEF_PROP(min_W_)
        DEF_PROP(max_W_)
        DEF_PROP(min_R_)
        DEF_PROP(max_R_)
        DEF_PROP(min_Wx_)
        DEF_PROP(max_Wx_)
        DEF_PROP(min_Rh_)
        DEF_PROP(max_Rh_)
        DEF_PROP(min_bw_)
        DEF_PROP(max_bw_)
        DEF_PROP(min_br_)
        DEF_PROP(max_br_)
        DEF_PROP(min_update_gate_input_)
        DEF_PROP(max_update_gate_input_)
        DEF_PROP(min_reset_gate_input_)
        DEF_PROP(max_reset_gate_input_)
        DEF_PROP(min_new_gate_input_)
        DEF_PROP(max_new_gate_input_)
        DEF_PROP(min_update_gate_output_)
        DEF_PROP(max_update_gate_output_)
        DEF_PROP(min_reset_gate_output_)
        DEF_PROP(max_reset_gate_output_)
        DEF_PROP(min_new_gate_output_)
        DEF_PROP(max_new_gate_output_)
        DEF_PROP(min_mul_reset_hidden_)
        DEF_PROP(max_mul_reset_hidden_)
        DEF_PROP(min_mul_new_contribution_)
        DEF_PROP(max_mul_new_contribution_)
        DEF_PROP(min_mul_old_contribution_)
        DEF_PROP(max_mul_old_contribution_)
        .def("reset", &GRUQuantizationRangesPy::reset,
             "Reset all ranges to invalid values. If hidden > 0, also update hidden_ and resize "
             "per-channel vectors.",
             py::arg("hidden") = -1);

    #undef DEF_PROP

    // OperatorQuantConfig 绑定（位宽配置 + 对称量化配置）
    // 位宽值: 1-32（Python 端只看到位宽数量，C++ 端决定实际类型）
    // 对称量化: is_symmetric=true 对称量化(zp=0), is_symmetric=false 非对称量化(zp≠0)
    py::class_<OperatorQuantConfigPy>(m, "OperatorQuantConfig")
        .def(py::init<>())
        // 位宽配置
        .def_readwrite("x_", &OperatorQuantConfigPy::x_)
        .def_readwrite("h_", &OperatorQuantConfigPy::h_)
        .def_readwrite("W_", &OperatorQuantConfigPy::W_)
        .def_readwrite("R_", &OperatorQuantConfigPy::R_)
        .def_readwrite("bw_", &OperatorQuantConfigPy::bw_)
        .def_readwrite("br_", &OperatorQuantConfigPy::br_)
        .def_readwrite("weight_ih_linear_", &OperatorQuantConfigPy::weight_ih_linear_)
        .def_readwrite("weight_hh_linear_", &OperatorQuantConfigPy::weight_hh_linear_)
        .def_readwrite("update_gate_input_", &OperatorQuantConfigPy::update_gate_input_)
        .def_readwrite("update_gate_output_", &OperatorQuantConfigPy::update_gate_output_)
        .def_readwrite("reset_gate_input_", &OperatorQuantConfigPy::reset_gate_input_)
        .def_readwrite("reset_gate_output_", &OperatorQuantConfigPy::reset_gate_output_)
        .def_readwrite("new_gate_input_", &OperatorQuantConfigPy::new_gate_input_)
        .def_readwrite("new_gate_output_", &OperatorQuantConfigPy::new_gate_output_)
        .def_readwrite("mul_reset_hidden_", &OperatorQuantConfigPy::mul_reset_hidden_)
        .def_readwrite("mul_old_contribution_", &OperatorQuantConfigPy::mul_old_contribution_)
        .def_readwrite("mul_new_contribution_", &OperatorQuantConfigPy::mul_new_contribution_)
        // 对称量化配置
        .def_readwrite("x_symmetric_", &OperatorQuantConfigPy::x_symmetric_)
        .def_readwrite("h_symmetric_", &OperatorQuantConfigPy::h_symmetric_)
        .def_readwrite("W_symmetric_", &OperatorQuantConfigPy::W_symmetric_)
        .def_readwrite("R_symmetric_", &OperatorQuantConfigPy::R_symmetric_)
        .def_readwrite("bw_symmetric_", &OperatorQuantConfigPy::bw_symmetric_)
        .def_readwrite("br_symmetric_", &OperatorQuantConfigPy::br_symmetric_)
        .def_readwrite("weight_ih_linear_symmetric_", &OperatorQuantConfigPy::weight_ih_linear_symmetric_)
        .def_readwrite("weight_hh_linear_symmetric_", &OperatorQuantConfigPy::weight_hh_linear_symmetric_)
        .def_readwrite("update_gate_input_symmetric_", &OperatorQuantConfigPy::update_gate_input_symmetric_)
        .def_readwrite("update_gate_output_symmetric_", &OperatorQuantConfigPy::update_gate_output_symmetric_)
        .def_readwrite("reset_gate_input_symmetric_", &OperatorQuantConfigPy::reset_gate_input_symmetric_)
        .def_readwrite("reset_gate_output_symmetric_", &OperatorQuantConfigPy::reset_gate_output_symmetric_)
        .def_readwrite("new_gate_input_symmetric_", &OperatorQuantConfigPy::new_gate_input_symmetric_)
        .def_readwrite("new_gate_output_symmetric_", &OperatorQuantConfigPy::new_gate_output_symmetric_)
        .def_readwrite("mul_reset_hidden_symmetric_", &OperatorQuantConfigPy::mul_reset_hidden_symmetric_)
        .def_readwrite("mul_old_contribution_symmetric_", &OperatorQuantConfigPy::mul_old_contribution_symmetric_)
        .def_readwrite("mul_new_contribution_symmetric_", &OperatorQuantConfigPy::mul_new_contribution_symmetric_)
        // 无符号配置（与 C++ is_unsigned_ 语义一致，只标记例外）
        .def_readwrite("x_unsigned_", &OperatorQuantConfigPy::x_unsigned_)
        .def_readwrite("h_unsigned_", &OperatorQuantConfigPy::h_unsigned_)
        .def_readwrite("W_unsigned_", &OperatorQuantConfigPy::W_unsigned_)
        .def_readwrite("R_unsigned_", &OperatorQuantConfigPy::R_unsigned_)
        .def_readwrite("bw_unsigned_", &OperatorQuantConfigPy::bw_unsigned_)
        .def_readwrite("br_unsigned_", &OperatorQuantConfigPy::br_unsigned_)
        .def_readwrite("weight_ih_linear_unsigned_", &OperatorQuantConfigPy::weight_ih_linear_unsigned_)
        .def_readwrite("weight_hh_linear_unsigned_", &OperatorQuantConfigPy::weight_hh_linear_unsigned_)
        .def_readwrite("update_gate_input_unsigned_", &OperatorQuantConfigPy::update_gate_input_unsigned_)
        .def_readwrite("update_gate_output_unsigned_", &OperatorQuantConfigPy::update_gate_output_unsigned_)
        .def_readwrite("reset_gate_input_unsigned_", &OperatorQuantConfigPy::reset_gate_input_unsigned_)
        .def_readwrite("reset_gate_output_unsigned_", &OperatorQuantConfigPy::reset_gate_output_unsigned_)
        .def_readwrite("new_gate_input_unsigned_", &OperatorQuantConfigPy::new_gate_input_unsigned_)
        .def_readwrite("new_gate_output_unsigned_", &OperatorQuantConfigPy::new_gate_output_unsigned_)
        .def_readwrite("mul_reset_hidden_unsigned_", &OperatorQuantConfigPy::mul_reset_hidden_unsigned_)
        .def_readwrite("mul_old_contribution_unsigned_", &OperatorQuantConfigPy::mul_old_contribution_unsigned_)
        .def_readwrite("mul_new_contribution_unsigned_", &OperatorQuantConfigPy::mul_new_contribution_unsigned_)
        // 量化粒度配置（仅对 W, R, bw, br 有效）
        .def_readwrite("W_granularity_", &OperatorQuantConfigPy::W_granularity_)
        .def_readwrite("R_granularity_", &OperatorQuantConfigPy::R_granularity_)
        .def_readwrite("bw_granularity_", &OperatorQuantConfigPy::bw_granularity_)
        .def_readwrite("br_granularity_", &OperatorQuantConfigPy::br_granularity_)
        .def_readwrite("usePOT2_", &OperatorQuantConfigPy::usePOT2_)
        .def_readwrite("pot_scale_method_", &OperatorQuantConfigPy::pot_scale_method_)
        .def_readwrite("pot_scale_tolerance_", &OperatorQuantConfigPy::pot_scale_tolerance_);

    // GRUQuantParams 绑定
    py::class_<GRUQuantParamsPy>(m, "GRUQuantParams")
        .def(py::init<>())
        .def_readwrite("hidden_", &GRUQuantParamsPy::hidden_)
        // 基础参数
        .def_readwrite("scale_x_", &GRUQuantParamsPy::scale_x_)
        .def_readwrite("zp_x_", &GRUQuantParamsPy::zp_x_)
        .def_readwrite("scale_h_", &GRUQuantParamsPy::scale_h_)
        .def_readwrite("zp_h_", &GRUQuantParamsPy::zp_h_)
    // 权重/偏置参数（唯一 per-channel 权威数组）
    .def_readwrite("scale_W_", &GRUQuantParamsPy::scale_W_)
    .def_readwrite("scale_R_", &GRUQuantParamsPy::scale_R_)
    .def_readwrite("scale_bw_", &GRUQuantParamsPy::scale_bw_)
    .def_readwrite("scale_br_", &GRUQuantParamsPy::scale_br_)
        // Linear 输出参数 (GEMM+bias)
        .def_readwrite("scale_weight_ih_linear_", &GRUQuantParamsPy::scale_weight_ih_linear_)
        .def_readwrite("zp_weight_ih_linear_", &GRUQuantParamsPy::zp_weight_ih_linear_)
        .def_readwrite("scale_weight_hh_linear_", &GRUQuantParamsPy::scale_weight_hh_linear_)
        .def_readwrite("zp_weight_hh_linear_", &GRUQuantParamsPy::zp_weight_hh_linear_)
        // 门激活函数输入参数（pre-activation）
        .def_readwrite("scale_update_gate_input_", &GRUQuantParamsPy::scale_update_gate_input_)
        .def_readwrite("zp_update_gate_input_", &GRUQuantParamsPy::zp_update_gate_input_)
        .def_readwrite("scale_reset_gate_input_", &GRUQuantParamsPy::scale_reset_gate_input_)
        .def_readwrite("zp_reset_gate_input_", &GRUQuantParamsPy::zp_reset_gate_input_)
        .def_readwrite("scale_new_gate_input_", &GRUQuantParamsPy::scale_new_gate_input_)
        .def_readwrite("zp_new_gate_input_", &GRUQuantParamsPy::zp_new_gate_input_)
        // 门激活函数输出参数（post-activation）
        .def_readwrite("scale_update_gate_output_", &GRUQuantParamsPy::scale_update_gate_output_)
        .def_readwrite("zp_update_gate_output_", &GRUQuantParamsPy::zp_update_gate_output_)
        .def_readwrite("scale_reset_gate_output_", &GRUQuantParamsPy::scale_reset_gate_output_)
        .def_readwrite("zp_reset_gate_output_", &GRUQuantParamsPy::zp_reset_gate_output_)
        .def_readwrite("scale_new_gate_output_", &GRUQuantParamsPy::scale_new_gate_output_)
        .def_readwrite("zp_new_gate_output_", &GRUQuantParamsPy::zp_new_gate_output_)
        // 中间计算参数
        .def_readwrite("scale_mul_reset_hidden_", &GRUQuantParamsPy::scale_mul_reset_hidden_)
        .def_readwrite("zp_mul_reset_hidden_", &GRUQuantParamsPy::zp_mul_reset_hidden_)
        // 隐状态更新参数
        .def_readwrite("scale_mul_new_contribution_", &GRUQuantParamsPy::scale_mul_new_contribution_)
        .def_readwrite("zp_mul_new_contribution_", &GRUQuantParamsPy::zp_mul_new_contribution_)
        .def_readwrite("scale_mul_old_contribution_", &GRUQuantParamsPy::scale_mul_old_contribution_)
        .def_readwrite("zp_mul_old_contribution_", &GRUQuantParamsPy::zp_mul_old_contribution_)
        // ⚠️ 关键字段：位宽配置，决定量化函数使用 int8 还是 int16
        .def_readwrite("bitwidth_config_", &GRUQuantParamsPy::bitwidth_config_);

    // 根据量化范围计算量化参数（支持自定义位宽配置）
    m.def("calculate_gru_quantitative_parameters", &calculate_gru_quantitative_parameters_wrapper,
          "Calculate GRU quantitative parameters from quantization ranges", py::arg("quant_ranges"),
          py::arg("bitwidth_config") = OperatorQuantConfigPy());

    // =====================================================================
    // AIMET 风格直方图校准接口（支持多批次累积，最高精度）
    // =====================================================================

    // GRUHistogramCollectors 绑定
    py::class_<GRUHistogramCollectorsPy>(m, "GRUHistogramCollectors")
        .def(py::init<>())
        .def(py::init<int, int>(), py::arg("hidden"), py::arg("num_bins") = 2048)
        .def("reset", &GRUHistogramCollectorsPy::reset,
             "Reset histogram collectors", py::arg("hidden") = -1, py::arg("num_bins") = -1)
        .def("is_valid", &GRUHistogramCollectorsPy::is_valid, "Check if histograms have valid data")
        .def("hidden", &GRUHistogramCollectorsPy::hidden, "Get hidden size")
        .def("num_bins", &GRUHistogramCollectorsPy::num_bins, "Get number of histogram bins")
        .def("print", &GRUHistogramCollectorsPy::print, "Print histogram statistics");

    // 从直方图计算量化参数（支持 SQNR 和 Percentile 两种方案）
    m.def("calculate_gru_quantitative_parameters_from_histograms",
          &calculate_gru_quantitative_parameters_from_histograms_wrapper,
          "Calculate GRU quantitative parameters from accumulated histograms.\n"
          "Supports two calibration schemes:\n"
          "  - SQNR (default): AIMET tf_enhanced style, searches optimal scale to minimize quantization noise\n"
          "  - Percentile: AIMET percentile style, uses percentile range for clipping",
          py::arg("hist_collectors"), py::arg("bitwidth_config") = OperatorQuantConfigPy(),
          py::arg("use_percentile") = false,
          py::arg("percentile_value") = 99.99f);

    // =====================================================================
    // forward_quant_float_storage: 量化前向传播（量化值 float32 存储）
    // =====================================================================
    m.def("forward_quant_float_storage", &forward_quant_float_storage_wrapper,
          "GRU quantized forward pass with float32-storage quantized values.\n"
          "\n"
          "Args:\n"
          "  is_training: Enable training mode (saves quantized values)\n"
          "  time_steps, batch_size, input_size, hidden_size: Dimension parameters\n"
          "  W, R, bw, br: Weight matrices and biases\n"
          "  x: Input tensor [T, B, I]\n"
          "  h0: Initial hidden state [B, H], optional\n"
          "  quant_params: Quantization parameters\n"
          "\n"
          "Returns:\n"
          "  tuple(h, v, W_q, R_q, bw_q, br_q, x_q, ...masks)\n"
          "  - h: Hidden states [T+1, B, H]\n"
          "  - v: Intermediate values [T, B, H*4]\n"
          "  - W_q, R_q, bw_q, br_q, x_q: Quantized values (float32 storage)\n"
          "  - ...masks: QAT masks (empty if not training)\n",
          py::arg("is_training"),
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W"), py::arg("R"), py::arg("bw"), py::arg("br"), py::arg("x"),
          py::arg("h0") = torch::Tensor(),
          py::arg("quant_params"));

    // =====================================================================
    // forward_quant_int_storage: 量化前向传播（量化值 int32 存储）
    // =====================================================================
    m.def("forward_quant_int_storage", &forward_quant_int_storage_wrapper,
          "GRU quantized forward pass with int32-storage quantized values.\n"
          "Internally uses quantGRUForwardInt -> quantGRUForwardInt32 fixed-point path.\n"
          "\n"
          "Args:\n"
          "  is_training: Enable training mode (saves quantized values)\n"
          "  time_steps, batch_size, input_size, hidden_size: Dimension parameters\n"
          "  W, R, bw, br: Weight matrices and biases (float32)\n"
          "  x: Input tensor [T, B, I] (float32)\n"
          "  h0: Initial hidden state [B, H], optional\n"
          "  quant_params: Quantization parameters\n"
          "\n"
          "Returns:\n"
          "  tuple(h, v, W_q, R_q, bw_q, br_q, x_q, ...masks)\n"
          "  - h: Hidden states [T+1, B, H] (float32, dequantized)\n"
          "  - v: Intermediate values [T, B, H*4] (float32, dequantized)\n"
          "  - W_q, R_q, bw_q, br_q, x_q: Quantized values (int32 storage)\n"
          "  - ...masks: QAT masks (empty if not training)\n",
          py::arg("is_training"),
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W"), py::arg("R"), py::arg("bw"), py::arg("br"), py::arg("x"),
          py::arg("h0") = torch::Tensor(),
          py::arg("quant_params"));

    // =====================================================================
    // forward_quantized_int_io: 纯定点 int 进 int 出推理前向（AIMET 打通链路）
    // =====================================================================
    m.def("forward_quantized_int_io", &forward_quantized_int_io_wrapper,
          "GRU pure fixed-point inference forward with INT input and INT output.\n"
          "x_q / h0_q are already-quantized int32 values (on scale_x/zp_x, scale_h/zp_h\n"
          "grids). Only weights are quantized internally; input quantization is skipped.\n"
          "Outputs int32 fixed-point hidden states WITHOUT dequantization.\n"
          "Intended for AIMET INT16_FIXED_EVAL fully-integer pipeline.\n"
          "\n"
          "Args:\n"
          "  time_steps, batch_size, input_size, hidden_size: Dimension parameters\n"
          "  W, R, bw, br: Weight matrices and biases (float32 master weights)\n"
          "  x_q: Input tensor [T, B, I] (int32, already quantized)\n"
          "  h0_q: Initial hidden state [B, H], optional (int32, already quantized)\n"
          "  quant_params: Quantization parameters\n"
          "\n"
          "Returns:\n"
          "  tuple(output_q, h_n_q)\n"
          "  - output_q: [T, B, H] int32 quantized hidden states (h_q[1:])\n"
          "  - h_n_q:    [1, B, H] int32 last hidden state (h_q[-1:])\n",
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W"), py::arg("R"), py::arg("bw"), py::arg("br"), py::arg("x_q"),
          py::arg("h0_q") = torch::Tensor(),
          py::arg("quant_params"));

    // =====================================================================
    // forward_fp: 浮点前向传播
    // =====================================================================
    m.def("forward_fp", &forward_fp_wrapper,
          "GRU floating-point forward pass.\n"
          "\n"
          "Args:\n"
          "  is_training: Enable training mode\n"
          "  time_steps, batch_size, input_size, hidden_size: Dimension parameters\n"
          "  W, R, bw, br: Weight matrices and biases\n"
          "  x: Input tensor [T, B, I]\n"
          "  h0: Initial hidden state [B, H], optional\n"
          "\n"
          "Returns:\n"
          "  tuple(h, v)\n"
          "  - h: Hidden states [T+1, B, H]\n"
          "  - v: Intermediate values [T, B, H*4]\n",
          py::arg("is_training"),
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W"), py::arg("R"), py::arg("bw"), py::arg("br"), py::arg("x"),
          py::arg("h0") = torch::Tensor());

    // =====================================================================
    // forward_calibrate: 校准前向传播（统一接口，原地累积）
    // =====================================================================
    m.def("forward_calibrate", &forward_calibrate_wrapper,
          "GRU forward pass for quantization calibration.\n"
          "Calibration data is accumulated in-place via pointer parameters.\n"
          "\n"
          "Args:\n"
          "  is_training: Enable training mode\n"
          "  time_steps, batch_size, input_size, hidden_size: Dimension parameters\n"
          "  W, R, bw, br: Weight matrices and biases (float)\n"
          "  x: Input tensor [T, B, I]\n"
          "  h0: Initial hidden state [B, H], optional\n"
          "  calib_method: Calibration method ('minmax', 'sqnr', 'percentile')\n"
          "  bitwidth_config: Quantization bitwidth configuration (required)\n"
          "  quant_ranges: Required for 'minmax' mode (updated in-place)\n"
          "  hist_collectors: Required for 'sqnr'/'percentile' mode (updated in-place)\n"
          "\n"
          "Returns:\n"
          "  tuple(h, v)\n"
          "  - h: Hidden states [T+1, B, H]\n"
          "  - v: Intermediate values [T, B, H*4]\n"
          "\n"
          "Usage:\n"
          "  # MINMAX calibration:\n"
          "  ranges = GRUQuantizationRanges(hidden_size)\n"
          "  for batch in batches:\n"
          "      h, v = forward_calibrate(..., 'minmax', bitwidth_config, ranges)\n"
          "  params = calculate_gru_quantitative_parameters(ranges, bitwidth_config)\n"
          "  \n"
          "  # SQNR/Percentile calibration:\n"
          "  hist = GRUHistogramCollectors(hidden_size)\n"
          "  for batch in batches:\n"
          "      h, v = forward_calibrate(..., 'sqnr', bitwidth_config, None, hist)\n"
          "  params = calculate_gru_quantitative_parameters_from_histograms(hist, bitwidth_config)",
          py::arg("is_training"),
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W"), py::arg("R"), py::arg("bw"), py::arg("br"), py::arg("x"),
          py::arg("h0") = torch::Tensor(),
          py::arg("calib_method"),
          py::arg("bitwidth_config"),
          py::arg("quant_ranges") = nullptr,
          py::arg("hist_collectors") = nullptr);


    // =====================================================================
    // backward_quant_float_storage: 量化反向传播（量化值 float32 存储）
    // =====================================================================
    m.def("backward_quant_float_storage", &backward_quant_float_storage_wrapper,
          "GRU quantized backward pass with float32-storage quantized values.\n"
          "\n"
          "Args:\n"
          "  time_steps, batch_size, input_size, hidden_size: Dimension parameters\n"
          "  W_q, R_q, bw_q, br_q, x_q: Quantized values from forward pass (float32 storage)\n"
          "  dh_new: Upstream gradient [T+1, B, H]\n"
          "  h: Hidden states from forward [T+1, B, H]\n"
          "  v: Intermediate values from forward [T, B, H*4]\n"
          "  quant_params: Quantization parameters\n"
          "  ...masks: QAT masks (empty if not training)\n"
          "\n"
          "Returns:\n"
          "  (dx, dW, dR, dbw, dbr, dh) gradient tuple\n",
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W_q"), py::arg("R_q"), py::arg("bw_q"), py::arg("br_q"), py::arg("x_q"),
          py::arg("dh_new"), py::arg("h"), py::arg("v"),
          py::arg("quant_params"),
          py::arg("x_mask") = torch::Tensor(),
          py::arg("h0_mask") = torch::Tensor(),
          py::arg("W_mask") = torch::Tensor(),
          py::arg("R_mask") = torch::Tensor(),
          py::arg("bw_mask") = torch::Tensor(),
          py::arg("br_mask") = torch::Tensor(),
          py::arg("weight_ih_linear_mask") = torch::Tensor(),
          py::arg("weight_hh_linear_mask") = torch::Tensor(),
          py::arg("gate_input_mask") = torch::Tensor(),
          py::arg("gate_output_mask") = torch::Tensor(),
          py::arg("h_mask") = torch::Tensor());

    // =====================================================================
    // backward_quant_int_storage: 量化反向传播（量化值 int32 存储）
    // =====================================================================
    m.def("backward_quant_int_storage", &backward_quant_int_storage_wrapper,
          "GRU quantized backward pass with int32-storage quantized values.\n"
          "Converts int32 quantized values to float32 then reuses float-storage backward.\n"
          "\n"
          "Args:\n"
          "  time_steps, batch_size, input_size, hidden_size: Dimension parameters\n"
          "  W_q, R_q, bw_q, br_q, x_q: Quantized values from forward pass (int32 storage)\n"
          "  dh_new: Upstream gradient [T+1, B, H]\n"
          "  h: Hidden states from forward [T+1, B, H]\n"
          "  v: Intermediate values from forward [T, B, H*4]\n"
          "  quant_params: Quantization parameters\n"
          "  ...masks: QAT masks (empty if not training)\n"
          "\n"
          "Returns:\n"
          "  (dx, dW, dR, dbw, dbr, dh) gradient tuple\n",
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W_q"), py::arg("R_q"), py::arg("bw_q"), py::arg("br_q"), py::arg("x_q"),
          py::arg("dh_new"), py::arg("h"), py::arg("v"),
          py::arg("quant_params"),
          py::arg("x_mask") = torch::Tensor(),
          py::arg("h0_mask") = torch::Tensor(),
          py::arg("W_mask") = torch::Tensor(),
          py::arg("R_mask") = torch::Tensor(),
          py::arg("bw_mask") = torch::Tensor(),
          py::arg("br_mask") = torch::Tensor(),
          py::arg("weight_ih_linear_mask") = torch::Tensor(),
          py::arg("weight_hh_linear_mask") = torch::Tensor(),
          py::arg("gate_input_mask") = torch::Tensor(),
          py::arg("gate_output_mask") = torch::Tensor(),
          py::arg("h_mask") = torch::Tensor());

    // =====================================================================
    // backward_fp: 浮点反向传播
    // =====================================================================
    m.def("backward_fp", &backward_fp_wrapper,
          "GRU floating-point backward pass.\n"
          "\n"
          "Args:\n"
          "  time_steps, batch_size, input_size, hidden_size: Dimension parameters\n"
          "  W, R, bw, br: Weight tensors\n"
          "  x: Input tensor [T, B, I]\n"
          "  dh_new: Upstream gradient [T+1, B, H]\n"
          "  h: Hidden states from forward [T+1, B, H]\n"
          "  v: Intermediate values from forward [T, B, H*4]\n"
          "\n"
          "Returns:\n"
          "  (dx, dW, dR, dbw, dbr, dh) gradient tuple\n",
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W"), py::arg("R"), py::arg("bw"), py::arg("br"), py::arg("x"),
          py::arg("dh_new"), py::arg("h"), py::arg("v"));
}
