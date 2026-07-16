// =====================================================================
// GRU 接口层实现 (gru_interface.cpp)
// =====================================================================

#include "gru_interface.h"

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <limits>
#include <cstdio>
#include <iostream>
#include <stdexcept>

#include "gru_quant_cpu.h"
#include "histogram_calibration_utils.h"
#include "histogram_collector.h"
#include "calibration_gpu.cuh"
#include "parallel_algorithm.h"
#include "pot_sqnr_calibrator.h"
#include "quantize_ops_helper.h"

namespace {

inline void ensureSymmetricWeightBiasConfig(const OperatorQuantConfig &cfg) {
    if (!cfg.W_symmetric_ || !cfg.R_symmetric_ || !cfg.bw_symmetric_ || !cfg.br_symmetric_) {
        throw std::runtime_error(
            "W/R/bw/br must remain symmetric (zp=0). "
            "Non-symmetric weight/bias quantization is not supported in this stage.");
    }
}

inline EncodedScaleResult encodeFromRange(
    float min_val,
    float max_val,
    QuantBitWidth bw,
    bool is_symmetric,
    bool use_pot2,
    PotScaleMethod method = PotScaleMethod::CoverRange,
    float tolerance = 0.02f) {
    const ContinuousScaleResult cont = calibrateContinuousScaleFromRange(min_val, max_val, bw, is_symmetric);
    // MinMax 与直方图共用 encodeScaleResult；策略由 method 决定（默认 CoverRange）
    return encodeScaleResult(cont, bw, is_symmetric, use_pot2, method, tolerance);
}

}  // namespace

// =====================================================================
// 量化参数计算
// =====================================================================

GRUQuantParams calculateGRUQuantitativeParameters(
    const GRUQuantizationRanges &quant_ranges, const OperatorQuantConfig &bitwidth_config) {
    ensureSymmetricWeightBiasConfig(bitwidth_config);
    GRUQuantParams quant_params;
    quant_params.hidden_ = quant_ranges.hidden_;
    quant_params.bitwidth_config_ = bitwidth_config;
    const bool usePOT2 = bitwidth_config.usePOT2_;
    const PotScaleMethod potMethod = bitwidth_config.pot_scale_method_;
    const float potTol = bitwidth_config.pot_scale_tolerance_;

    // 辅助 lambda：单值校准 -> 唯一权威 QuantParam（scale + zp）
    auto calibrateScalar = [usePOT2, potMethod, potTol](QuantBitWidth bw, bool symmetric,
                                     float min_val, float max_val) -> QuantParam {
        const EncodedScaleResult encoded =
            encodeFromRange(min_val, max_val, bw, symmetric, usePOT2, potMethod, potTol);
        return QuantParam{storedScaleForMode(encoded, usePOT2), encoded.zero_point};
    };

    // 输入 x 和隐藏状态 h
    quant_params.x_ = calibrateScalar(bitwidth_config.x_, bitwidth_config.x_symmetric_,
                                      quant_ranges.min_x_, quant_ranges.max_x_);
    quant_params.h_ = calibrateScalar(bitwidth_config.h_, bitwidth_config.h_symmetric_,
                                      quant_ranges.min_h_, quant_ranges.max_h_);

    // 权重 W/R 和偏置 bw/br：唯一 per-channel 权威数组，按粒度广播
    const int channel_size = quant_ranges.hidden_ * 3;
    const int hidden_size = quant_ranges.hidden_;

    auto calibrateChannel = [&](ChannelQuantParam& out,
                                OperatorQuantConfig::QuantizationGranularity gran,
                                QuantBitWidth bw, bool symmetric,
                                const std::vector<float>& mins, const std::vector<float>& maxs) {
        out.granularity = gran;
        out.hidden = hidden_size;
        out.resize(channel_size);
        if (gran == OperatorQuantConfig::PER_TENSOR) {
            float mn = *std::min_element(mins.begin(), mins.end());
            float mx = *std::max_element(maxs.begin(), maxs.end());
            const QuantParam qp = calibrateScalar(bw, symmetric, mn, mx);
            std::fill(out.channels.begin(), out.channels.end(), qp);
        } else if (gran == OperatorQuantConfig::PER_GATE) {
            for (int gate = 0; gate < 3; ++gate) {
                float mn = std::numeric_limits<float>::max();
                float mx = std::numeric_limits<float>::lowest();
                for (int c = gate * hidden_size; c < (gate + 1) * hidden_size; ++c) {
                    mn = std::min(mn, mins[c]);
                    mx = std::max(mx, maxs[c]);
                }
                const QuantParam qp = calibrateScalar(bw, symmetric, mn, mx);
                for (int c = gate * hidden_size; c < (gate + 1) * hidden_size; ++c) out.channel(c) = qp;
            }
        } else {  // PER_CHANNEL
            for (int c = 0; c < channel_size; ++c) {
                out.channel(c) = calibrateScalar(bw, symmetric, mins[c], maxs[c]);
            }
        }
    };

    calibrateChannel(quant_params.W_, bitwidth_config.W_granularity_, bitwidth_config.W_,
                     bitwidth_config.W_symmetric_, quant_ranges.min_W_, quant_ranges.max_W_);
    calibrateChannel(quant_params.R_, bitwidth_config.R_granularity_, bitwidth_config.R_,
                     bitwidth_config.R_symmetric_, quant_ranges.min_R_, quant_ranges.max_R_);
    calibrateChannel(quant_params.bw_, bitwidth_config.bw_granularity_, bitwidth_config.bw_,
                     bitwidth_config.bw_symmetric_, quant_ranges.min_bw_, quant_ranges.max_bw_);
    calibrateChannel(quant_params.br_, bitwidth_config.br_granularity_, bitwidth_config.br_,
                     bitwidth_config.br_symmetric_, quant_ranges.min_br_, quant_ranges.max_br_);

    // Linear 输出 (GEMM+bias)
    quant_params.weight_ih_linear_ = calibrateScalar(bitwidth_config.weight_ih_linear_,
        bitwidth_config.weight_ih_linear_symmetric_, quant_ranges.min_Wx_, quant_ranges.max_Wx_);
    quant_params.weight_hh_linear_ = calibrateScalar(bitwidth_config.weight_hh_linear_,
        bitwidth_config.weight_hh_linear_symmetric_, quant_ranges.min_Rh_, quant_ranges.max_Rh_);

    // 门激活输入
    quant_params.update_gate_input_ = calibrateScalar(bitwidth_config.update_gate_input_,
        bitwidth_config.update_gate_input_symmetric_, quant_ranges.min_update_gate_input_, quant_ranges.max_update_gate_input_);
    quant_params.reset_gate_input_ = calibrateScalar(bitwidth_config.reset_gate_input_,
        bitwidth_config.reset_gate_input_symmetric_, quant_ranges.min_reset_gate_input_, quant_ranges.max_reset_gate_input_);
    quant_params.new_gate_input_ = calibrateScalar(bitwidth_config.new_gate_input_,
        bitwidth_config.new_gate_input_symmetric_, quant_ranges.min_new_gate_input_, quant_ranges.max_new_gate_input_);

    // 激活函数输出（确保最小范围）
    constexpr float MIN_ACTIVATION_RANGE = 0.5f;
    auto calibrateWithMinRange = [&](const char* name, QuantBitWidth bw, bool symmetric,
                                     float min_val, float max_val) -> QuantParam {
        ensureMinRange(min_val, max_val, MIN_ACTIVATION_RANGE, name);
        return calibrateScalar(bw, symmetric, min_val, max_val);
    };

    quant_params.update_gate_output_ = calibrateWithMinRange("scale_update_gate_output",
        bitwidth_config.update_gate_output_, bitwidth_config.update_gate_output_symmetric_,
        quant_ranges.min_update_gate_output_, quant_ranges.max_update_gate_output_);
    quant_params.reset_gate_output_ = calibrateWithMinRange("scale_reset_gate_output",
        bitwidth_config.reset_gate_output_, bitwidth_config.reset_gate_output_symmetric_,
        quant_ranges.min_reset_gate_output_, quant_ranges.max_reset_gate_output_);
    quant_params.new_gate_output_ = calibrateWithMinRange("scale_new_gate_output",
        bitwidth_config.new_gate_output_, bitwidth_config.new_gate_output_symmetric_,
        quant_ranges.min_new_gate_output_, quant_ranges.max_new_gate_output_);

    // 中间计算
    quant_params.mul_reset_hidden_ = calibrateScalar(bitwidth_config.mul_reset_hidden_,
        bitwidth_config.mul_reset_hidden_symmetric_, quant_ranges.min_mul_reset_hidden_, quant_ranges.max_mul_reset_hidden_);
    quant_params.mul_new_contribution_ = calibrateScalar(bitwidth_config.mul_new_contribution_,
        bitwidth_config.mul_new_contribution_symmetric_, quant_ranges.min_mul_new_contribution_, quant_ranges.max_mul_new_contribution_);
    quant_params.mul_old_contribution_ = calibrateScalar(bitwidth_config.mul_old_contribution_,
        bitwidth_config.mul_old_contribution_symmetric_, quant_ranges.min_mul_old_contribution_, quant_ranges.max_mul_old_contribution_);

    // 生成 LUT 并存储到参数中
    generate_piecewise_linear_lut_to_params(quant_params);
    return quant_params;
}

// =====================================================================
// 统一前向/反向传播接口（主入口）
// =====================================================================

// 量化模式选择宏：默认使用浮点存储版（方案2）
// 如需使用 INT32 版本（方案1），注释掉此行
#define USE_FP_STORAGE 1


// =====================================================================
// Haste GRU 前向传播实现
// =====================================================================

void hasteGRUForward(bool is_training, const int time_steps, const int batch_size,
                     const int input_size, const int hidden_size, const float *W, const float *R,
                     const float *bw, const float *br, const float *x, const float *h0,
                     const cublasHandle_t &g_blas_handle, float *h, float *v) {
    dev::vector<float> tmp_Wx_dev(time_steps * batch_size * hidden_size *
                                  3);                             // 用于存放W * x的中间结果
    dev::vector<float> tmp_Rh_dev(batch_size * hidden_size * 3);  // 用于存放R * h的中间结果

    // 处理初始隐藏状态
    const int NH = batch_size * hidden_size;
    if (h0 != nullptr) {
        // 如果提供了初始状态，复制到 h[0]
        d2d(h, h0, NH);
    } else {
        // 否则初始化为零
        cudaMemset(h, 0, NH * sizeof(float));
    }

    gru::ForwardPass<float> forward =
        gru::ForwardPass<float>(is_training,  // training: true为训练，false为推理
                                batch_size, input_size, hidden_size, g_blas_handle);

    forward.Run(time_steps, W, R, bw, br, x, h, v, tmp_Wx_dev.data(), tmp_Rh_dev.data(), 0.0f,
                nullptr);

    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in hasteGRUForward: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in hasteGRUForward: ") + err_str);
    }
}

// =====================================================================
// 反向传播实现
// =====================================================================

// ★★★ 重要：W_t、R_t、x_t 需要传入【转置后】的数据！★★★
void hasteGRUBackward(const int time_steps, const int batch_size, const int input_size,
                      const int hidden_size, const float *W_t, const float *R_t, const float *bw,
                      const float *br, const float *x_t, const float *dh_new, const float *h,
                      const float *v, const cublasHandle_t &g_blas_handle, float *dx, float *dW,
                      float *dR, float *dbw, float *dbr, float *dh) {
    dev::vector<float> dp_dev(time_steps * batch_size * hidden_size *
                              3);  // 临时缓存梯度（内部结构用）
    dev::vector<float> dq_dev(time_steps * batch_size * hidden_size *
                              3);  // 临时缓存梯度（内部结构用）

    gru::BackwardPass<float> backward(batch_size, input_size, hidden_size, g_blas_handle);

    backward.Run(time_steps, W_t, R_t, bw, br, x_t, h, v, dh_new, dx, dW, dR, dbw, dbr, dh,
                 dp_dev.data(), dq_dev.data(), nullptr);

    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in hasteGRUBackward: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in hasteGRUBackward: ") + err_str);
    }
}

// =====================================================================
// 量化 GRU 反向传播（支持 QAT mask）
// =====================================================================

void quantGRUBackward(const int time_steps, const int batch_size, const int input_size,
                      const int hidden_size, const float *W_t, const float *R_t, const float *bw,
                      const float *br, const float *x_t, const float *dh_new, const float *h,
                      const float *v, const cublasHandle_t &g_blas_handle, float *dx, float *dW,
                      float *dR, float *dbw, float *dbr, float *dh,
                      const GRUQuantParams *quant_params,
                      const uint8_t *x_mask, const uint8_t *h0_mask,
                      const uint8_t *W_mask, const uint8_t *R_mask,
                      const uint8_t *bw_mask, const uint8_t *br_mask,
                      const uint8_t *weight_ih_linear_mask, const uint8_t *weight_hh_linear_mask,
                      const uint8_t *gate_input_mask, const uint8_t *gate_output_mask,
                      const uint8_t *h_mask) {
    // 临时缓存梯度（内部结构用）
    dev::vector<float> dp_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> dq_dev(time_steps * batch_size * hidden_size * 3);

    // 使用量化版反向传播类（支持 QAT mask）
    // 注意：不需要rescale补偿，因为反向传播使用的是反量化后的浮点值，梯度计算已经是正确的
    // quant_params 参数保留以保持接口兼容性，但不再使用
    (void)quant_params;  // suppress unused parameter warning
    gru::BackwardPassQuant<float> backward(batch_size, input_size, hidden_size, g_blas_handle);

    backward.Run(time_steps, W_t, R_t, bw, br, x_t, h, v, dh_new,
                 dx, dW, dR, dbw, dbr, dh,
                 dp_dev.data(), dq_dev.data(),
                 nullptr,  // zoneout_mask
                 // QAT masks
                 x_mask, h0_mask, W_mask, R_mask, bw_mask, br_mask,
                 weight_ih_linear_mask, weight_hh_linear_mask,
                 gate_input_mask, gate_output_mask, h_mask);

    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in quantGRUBackward: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in quantGRUBackward: ") + err_str);
    }
}

// =====================================================================
// 权重量化实现（统一 int32_t 输出）
// =====================================================================

// GPU 权重量化（统一模板接口，支持训练模式 mask）
template <bool Training>
void quantitativeWeight(const int input_size, const int hidden_size, const float *W, const float *R,
                        const float *bw, const float *br,
                        const GRUQuantParams &quant_parms, int32_t *W_quant,
                        int32_t *R_quant, int32_t *bw_quant, int32_t *br_quant,
                        uint8_t *W_mask, uint8_t *R_mask, uint8_t *bw_mask, uint8_t *br_mask) {
    // 由单一权威种子派生 per-channel FixedPointScale（量化 boundary 使用）
    const bool usePOT2 = quant_parms.bitwidth_config_.usePOT2_;
    dev::vector<FixedPointScale> shift_W_dev(toFixedScales(quant_parms.W_, usePOT2));
    dev::vector<FixedPointScale> shift_R_dev(toFixedScales(quant_parms.R_, usePOT2));
    dev::vector<FixedPointScale> shift_bw_dev(toFixedScales(quant_parms.bw_, usePOT2));
    dev::vector<FixedPointScale> shift_br_dev(toFixedScales(quant_parms.br_, usePOT2));

    // 统一 int32_t 输出，使用 clamp_by_bitwidth 限制到实际位宽
    const auto &bw_cfg = quant_parms.bitwidth_config_;
    dev::quantificationPerChannelBitwidth<Training>(W, W_quant, W_mask, input_size, 3 * hidden_size, 
                                                    shift_W_dev, bw_cfg.W_);
    dev::quantificationPerChannelBitwidth<Training>(R, R_quant, R_mask, hidden_size, 3 * hidden_size, 
                                                    shift_R_dev, bw_cfg.R_);
    dev::quantificationPerChannelBitwidth<Training>(bw, bw_quant, bw_mask, 1, 3 * hidden_size, 
                                                    shift_bw_dev, bw_cfg.bw_);
    dev::quantificationPerChannelBitwidth<Training>(br, br_quant, br_mask, 1, 3 * hidden_size, 
                                                     shift_br_dev, bw_cfg.br_);

    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in quantitativeWeight: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in quantitativeWeight: ") + err_str);
    }
}

// 显式实例化模板
template void quantitativeWeight<false>(const int, const int, const float *, const float *,
                                         const float *, const float *, const GRUQuantParams &,
                                         int32_t *, int32_t *, int32_t *, int32_t *,
                                         uint8_t *, uint8_t *, uint8_t *, uint8_t *);
template void quantitativeWeight<true>(const int, const int, const float *, const float *,
                                        const float *, const float *, const GRUQuantParams &,
                                        int32_t *, int32_t *, int32_t *, int32_t *,
                                        uint8_t *, uint8_t *, uint8_t *, uint8_t *);

// CPU 权重量化（统一接口）
void quantitativeWeightCPU(const int input_size, const int hidden_size, const float *W, const float *R,
                           const float *bw, const float *br,
                           const GRUQuantParams &quant_parms, int32_t *W_quant,
                           int32_t *R_quant, int32_t *bw_quant, int32_t *br_quant) {
    const int hidden3 = hidden_size * 3;
    const auto &bw_cfg = quant_parms.bitwidth_config_;
    const bool usePOT2 = bw_cfg.usePOT2_;
    const std::vector<FixedPointScale> fW = toFixedScales(quant_parms.W_, usePOT2);
    const std::vector<FixedPointScale> fR = toFixedScales(quant_parms.R_, usePOT2);
    const std::vector<FixedPointScale> fbw = toFixedScales(quant_parms.bw_, usePOT2);
    const std::vector<FixedPointScale> fbr = toFixedScales(quant_parms.br_, usePOT2);

    // 量化权重矩阵（per-channel）
    for (int k = 0; k < input_size; k++) {
        for (int m = 0; m < hidden3; m++) {
            int idx = k * hidden3 + m;
            W_quant[idx] = quantize<false>(W[idx], fW[m], 0, bw_cfg.W_);
        }
    }

    for (int k = 0; k < hidden_size; k++) {
        for (int m = 0; m < hidden3; m++) {
            int idx = k * hidden3 + m;
            R_quant[idx] = quantize<false>(R[idx], fR[m], 0, bw_cfg.R_);
        }
    }

    // 量化偏置（per-channel）
    for (int m = 0; m < hidden3; m++) {
        bw_quant[m] = quantize<false>(bw[m], fbw[m], 0, bw_cfg.bw_);
        br_quant[m] = quantize<false>(br[m], fbr[m], 0, bw_cfg.br_);
    }
}

// =====================================================================
// GRU 权重量化统一接口（封装 W, R, bw, br）- 浮点存储版本
// =====================================================================

// GRU 权重量化统一接口（根据 granularity 自动选择量化方式）
template <bool Training>
void quantizeGRUWeights(const float *W, const float *R, const float *bw, const float *br,
                        float *W_q_out, float *R_q_out, float *bw_q_out, float *br_q_out,
                        uint8_t *W_mask, uint8_t *R_mask, uint8_t *bw_mask, uint8_t *br_mask,
                        size_t input_size, size_t hidden_size,
                        const GRUQuantParams &quant_params) {
    const auto &bw_cfg = quant_params.bitwidth_config_;
    
    // 占位符空 vector（当不是 PER_CHANNEL 时使用，函数内部不会访问）
    static const dev::vector<FixedPointScale> empty_shift;

    // 从 per-channel fixed_scale 向量派生 per-tensor / per-gate 标量
    auto gate_fixed = [hidden_size](const std::vector<FixedPointScale> &v) -> std::array<FixedPointScale, 3> {
        std::array<FixedPointScale, 3> g{};
        if (!v.empty()) {
            g[0] = v[0];
            g[1] = v[std::min(hidden_size, v.size() - 1)];
            g[2] = v[std::min(2 * hidden_size, v.size() - 1)];
        }
        return g;
    };
    auto tensor_fixed = [](const std::vector<FixedPointScale> &v) -> FixedPointScale {
        return v.empty() ? FixedPointScale{} : v[0];
    };

    // 只在 PER_CHANNEL 粒度时创建 fixed_scale 数组
    const bool usePOT2 = bw_cfg.usePOT2_;
    const std::vector<FixedPointScale> fW = toFixedScales(quant_params.W_, usePOT2);
    const std::vector<FixedPointScale> fR = toFixedScales(quant_params.R_, usePOT2);
    const std::vector<FixedPointScale> fbw = toFixedScales(quant_params.bw_, usePOT2);
    const std::vector<FixedPointScale> fbr = toFixedScales(quant_params.br_, usePOT2);

    dev::vector<FixedPointScale> shift_W_dev, shift_R_dev, shift_bw_dev, shift_br_dev;
    if (bw_cfg.W_granularity_ == OperatorQuantConfig::PER_CHANNEL) {
        shift_W_dev = dev::vector<FixedPointScale>(fW);
    }
    if (bw_cfg.R_granularity_ == OperatorQuantConfig::PER_CHANNEL) {
        shift_R_dev = dev::vector<FixedPointScale>(fR);
    }
    if (bw_cfg.bw_granularity_ == OperatorQuantConfig::PER_CHANNEL) {
        shift_bw_dev = dev::vector<FixedPointScale>(fbw);
    }
    if (bw_cfg.br_granularity_ == OperatorQuantConfig::PER_CHANNEL) {
        shift_br_dev = dev::vector<FixedPointScale>(fbr);
    }
    
    // 量化 W
    dev::quantificationWeightFP<Training>(W, W_q_out, W_mask, input_size, hidden_size,
                                          bw_cfg.W_granularity_,
                                          tensor_fixed(fW),
                                          gate_fixed(fW),
                                          bw_cfg.W_granularity_ == OperatorQuantConfig::PER_CHANNEL ? shift_W_dev : empty_shift,
                                          bw_cfg.W_);
    
    // 量化 R
    dev::quantificationWeightFP<Training>(R, R_q_out, R_mask, hidden_size, hidden_size,
                                          bw_cfg.R_granularity_,
                                          tensor_fixed(fR),
                                          gate_fixed(fR),
                                          bw_cfg.R_granularity_ == OperatorQuantConfig::PER_CHANNEL ? shift_R_dev : empty_shift,
                                          bw_cfg.R_);
    
    // 量化 bw
    dev::quantificationWeightFP<Training>(bw, bw_q_out, bw_mask, 1, hidden_size,
                                          bw_cfg.bw_granularity_,
                                          tensor_fixed(fbw),
                                          gate_fixed(fbw),
                                          bw_cfg.bw_granularity_ == OperatorQuantConfig::PER_CHANNEL ? shift_bw_dev : empty_shift,
                                          bw_cfg.bw_);
    
    // 量化 br
    dev::quantificationWeightFP<Training>(br, br_q_out, br_mask, 1, hidden_size,
                                          bw_cfg.br_granularity_,
                                          tensor_fixed(fbr),
                                          gate_fixed(fbr),
                                          bw_cfg.br_granularity_ == OperatorQuantConfig::PER_CHANNEL ? shift_br_dev : empty_shift,
                                          bw_cfg.br_);
}

// 显式实例化模板
template void quantizeGRUWeights<false>(const float *W, const float *R, const float *bw, const float *br,
                                        float *W_q_out, float *R_q_out, float *bw_q_out, float *br_q_out,
                                        uint8_t *W_mask, uint8_t *R_mask, uint8_t *bw_mask, uint8_t *br_mask,
                                        size_t input_size, size_t hidden_size,
                                        const GRUQuantParams &quant_params);
template void quantizeGRUWeights<true>(const float *W, const float *R, const float *bw, const float *br,
                                       float *W_q_out, float *R_q_out, float *bw_q_out, float *br_q_out,
                                       uint8_t *W_mask, uint8_t *R_mask, uint8_t *bw_mask, uint8_t *br_mask,
                                       size_t input_size, size_t hidden_size,
                                       const GRUQuantParams &quant_params);

// 反量化 GRU 权重（W, R, bw, br）- 使用统一接口，内部根据 granularity 自动选择
void dequantizeGRUWeights(float *W_q, float *R_q, float *bw_q, float *br_q,
                          size_t input_size, size_t hidden_size,
                          const GRUQuantParams &quant_params) {
    const auto &bw_cfg = quant_params.bitwidth_config_;
    
    // 占位符空 vector（当不是 PER_CHANNEL 时使用，函数内部不会访问）
    static const dev::vector<FixedPointScale> empty_shift;

    // 从 per-channel fixed_scale 向量派生 per-tensor / per-gate 标量
    auto gate_fixed = [hidden_size](const std::vector<FixedPointScale> &v) -> std::array<FixedPointScale, 3> {
        std::array<FixedPointScale, 3> g{};
        if (!v.empty()) {
            g[0] = v[0];
            g[1] = v[std::min(hidden_size, v.size() - 1)];
            g[2] = v[std::min(2 * hidden_size, v.size() - 1)];
        }
        return g;
    };
    auto tensor_fixed = [](const std::vector<FixedPointScale> &v) -> FixedPointScale {
        return v.empty() ? FixedPointScale{} : v[0];
    };

    // 只在 PER_CHANNEL 粒度时创建 fixed_scale 数组
    const bool usePOT2 = bw_cfg.usePOT2_;
    const std::vector<FixedPointScale> fW = toFixedScales(quant_params.W_, usePOT2);
    const std::vector<FixedPointScale> fR = toFixedScales(quant_params.R_, usePOT2);
    const std::vector<FixedPointScale> fbw = toFixedScales(quant_params.bw_, usePOT2);
    const std::vector<FixedPointScale> fbr = toFixedScales(quant_params.br_, usePOT2);

    dev::vector<FixedPointScale> shift_W_dev, shift_R_dev, shift_bw_dev, shift_br_dev;
    if (bw_cfg.W_granularity_ == OperatorQuantConfig::PER_CHANNEL) {
        shift_W_dev = dev::vector<FixedPointScale>(fW);
    }
    if (bw_cfg.R_granularity_ == OperatorQuantConfig::PER_CHANNEL) {
        shift_R_dev = dev::vector<FixedPointScale>(fR);
    }
    if (bw_cfg.bw_granularity_ == OperatorQuantConfig::PER_CHANNEL) {
        shift_bw_dev = dev::vector<FixedPointScale>(fbw);
    }
    if (bw_cfg.br_granularity_ == OperatorQuantConfig::PER_CHANNEL) {
        shift_br_dev = dev::vector<FixedPointScale>(fbr);
    }
    
    // 反量化 W
    dev::dequantificationWeightFPInplace(W_q, input_size, hidden_size,
                                         bw_cfg.W_granularity_,
                                         tensor_fixed(fW),
                                         gate_fixed(fW),
                                         bw_cfg.W_granularity_ == OperatorQuantConfig::PER_CHANNEL ? shift_W_dev : empty_shift);
    
    // 反量化 R
    dev::dequantificationWeightFPInplace(R_q, hidden_size, hidden_size,
                                        bw_cfg.R_granularity_,
                                        tensor_fixed(fR),
                                        gate_fixed(fR),
                                        bw_cfg.R_granularity_ == OperatorQuantConfig::PER_CHANNEL ? shift_R_dev : empty_shift);
    
    // 反量化 bw
    dev::dequantificationWeightFPInplace(bw_q, 1, hidden_size,
                                        bw_cfg.bw_granularity_,
                                        tensor_fixed(fbw),
                                        gate_fixed(fbw),
                                        bw_cfg.bw_granularity_ == OperatorQuantConfig::PER_CHANNEL ? shift_bw_dev : empty_shift);
    
    // 反量化 br
    dev::dequantificationWeightFPInplace(br_q, 1, hidden_size,
                                        bw_cfg.br_granularity_,
                                        tensor_fixed(fbr),
                                        gate_fixed(fbr),
                                        bw_cfg.br_granularity_ == OperatorQuantConfig::PER_CHANNEL ? shift_br_dev : empty_shift);
}

// =====================================================================
// 纯定点 GRU 前向传播（GPU 核心实现）
// =====================================================================

// GPU 纯定点 GRU 前向传播（int32 输入/输出）
// 这是量化 GRU 的核心计算，所有高层接口都调用此函数
// 输入必须是已量化的 int32_t 值，量化应在外部完成
void quantGRUForwardInt32(
    bool is_training, int time_steps, int batch_size, int input_size, int hidden_size,
    const int32_t *W_q, const int32_t *R_q, const int32_t *bw_q, const int32_t *br_q,
    const int32_t *x_q, const int32_t *h0_q,
    const GRUQuantParams &quant_params,
    const cublasHandle_t &g_blas_handle,
    int32_t *h_q, int32_t *v_q,
    // 计算过程 mask（外部分配，nullptr=不保存）
    uint8_t *weight_ih_linear_mask,
    uint8_t *weight_hh_linear_mask,
    uint8_t *gate_input_mask,
    uint8_t *gate_output_mask,
    uint8_t *h_mask) {
    
    const int NH = batch_size * hidden_size;
    
    // 初始化 h_q[0]
    if (h0_q != nullptr) {
        // 从 h0_q 复制到 h_q[0]
        cudaMemcpy(h_q, h0_q, NH * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    } else {
        // 使用零点值初始化
        dev::fill_n(h_q, NH, quant_params.h_.zero_point);
    }
    
    // 分配中间值缓冲区
    dev::vector<int32_t> v_internal;
    int32_t* v_ptr = v_q;
    if (v_q == nullptr) {
        // 内部分配临时缓冲区
        v_internal.resize(time_steps * batch_size * hidden_size * 4);
        v_ptr = v_internal.data();
    }
    
    // 创建 ForwardPassQuant 对象
    gru::ForwardPassQuant forward(is_training, batch_size, input_size, hidden_size, g_blas_handle);
    forward.setRescaleParam(quant_params);
    
    // 运行前向传播（传递 mask 指针）
    forward.Run(time_steps, W_q, R_q, bw_q, br_q, x_q, h_q, v_ptr, 0.0f, nullptr,
                weight_ih_linear_mask, weight_hh_linear_mask, gate_input_mask, gate_output_mask, h_mask);
    
    // 同步 CUDA 操作
    cudaDeviceSynchronize();
    
    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in quantGRUForwardInt32: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in quantGRUForwardInt32: ") + err_str);
    }
}

// =====================================================================
// 量化 GRU 前向传播（浮点接口）
// =====================================================================

// 量化 GRU 前向传播（整数版，int32_t 存储，浮点输入/输出，内部自动量化权重和激活）
void quantGRUForwardInt(bool is_training, const int time_steps, const int batch_size,
                     const int input_size, const int hidden_size, const float *W,
                     const float *R, const float *bw, const float *br, const float *x,
                     const float *h0, const GRUQuantParams &quant_parms,
                     const cublasHandle_t &g_blas_handle, float *h, float *v,
                     // 输入量化 mask
                     uint8_t *x_mask,
                     uint8_t *h0_mask,
                     uint8_t *W_mask,
                     uint8_t *R_mask,
                     uint8_t *bw_mask,
                     uint8_t *br_mask,
                     // 计算过程 mask
                     uint8_t *weight_ih_linear_mask,
                     uint8_t *weight_hh_linear_mask,
                     uint8_t *gate_input_mask,
                     uint8_t *gate_output_mask,
                     uint8_t *h_mask,
                     // int32 量化值输出（int_storage 路径使用，nullptr=内部临时分配）
                     int32_t *W_q_out,
                     int32_t *R_q_out,
                     int32_t *bw_q_out,
                     int32_t *br_q_out,
                     int32_t *x_q_out) {
    const int hidden3 = hidden_size * 3;
    const std::size_t x_size = time_steps * batch_size * input_size;
    const std::size_t h_size = (time_steps + 1) * batch_size * hidden_size;
    const int NH = batch_size * hidden_size;
    const auto &bw_cfg = quant_parms.bitwidth_config_;
    const bool usePOT2 = bw_cfg.usePOT2_;

    // 量化值缓冲区：优先使用外部传入的 int32 buffer（int_storage 路径），
    // 否则退回内部临时 dev::vector（推理/独立调用）。
    dev::vector<int32_t> W_q_tmp, R_q_tmp, bw_q_tmp, br_q_tmp, x_q_tmp;
    int32_t *W_q = W_q_out;
    int32_t *R_q = R_q_out;
    int32_t *bw_q = bw_q_out;
    int32_t *br_q = br_q_out;
    int32_t *x_q = x_q_out;
    if (W_q == nullptr) { W_q_tmp.resize(input_size * hidden3); W_q = W_q_tmp.data(); }
    if (R_q == nullptr) { R_q_tmp.resize(hidden_size * hidden3); R_q = R_q_tmp.data(); }
    if (bw_q == nullptr) { bw_q_tmp.resize(hidden3); bw_q = bw_q_tmp.data(); }
    if (br_q == nullptr) { br_q_tmp.resize(hidden3); br_q = br_q_tmp.data(); }
    if (x_q == nullptr) { x_q_tmp.resize(x_size); x_q = x_q_tmp.data(); }

    // 1. 量化权重（W, R, bw, br）- 使用统一接口
    if (is_training) {
        quantitativeWeight<true>(input_size, hidden_size, W, R, bw, br, quant_parms,
                                 W_q, R_q, bw_q, br_q,
                                 W_mask, R_mask, bw_mask, br_mask);
    } else {
        quantitativeWeight<false>(input_size, hidden_size, W, R, bw, br, quant_parms,
                                  W_q, R_q, bw_q, br_q,
                                  nullptr, nullptr, nullptr, nullptr);
    }

    // 2. 量化输入 x（使用统一接口）
    if (is_training) {
        dev::quantificationBitwidth<true>(x, x_q, x_mask, x_size,
                                          toFixedScale(quant_parms.x_, usePOT2), quant_parms.x_.zero_point, bw_cfg.x_);
    } else {
        dev::quantificationBitwidth<false>(x, x_q, nullptr, x_size,
                                           toFixedScale(quant_parms.x_, usePOT2), quant_parms.x_.zero_point, bw_cfg.x_);
    }

    // 3. 量化 h0（如提供）
    dev::vector<int32_t> h0_quant;
    const int32_t *h0_q_ptr = nullptr;
    if (h0 != nullptr) {
        h0_quant.resize(NH);
        if (is_training) {
            dev::quantificationBitwidth<true>(h0, h0_quant.data(), h0_mask, NH,
                                             toFixedScale(quant_parms.h_, usePOT2), quant_parms.h_.zero_point, bw_cfg.h_);
        } else {
            dev::quantificationBitwidth<false>(h0, h0_quant.data(), nullptr, NH,
                                              toFixedScale(quant_parms.h_, usePOT2), quant_parms.h_.zero_point, bw_cfg.h_);
        }
        h0_q_ptr = h0_quant.data();
    }

    // 4. 分配输出缓冲区
    dev::vector<int32_t> h_quant(h_size);
    dev::vector<int32_t> v_quant(v != nullptr ? time_steps * batch_size * hidden_size * 4 : 0);

    // 5. 调用核心定点计算（quantGRUForwardInt32 接受已量化的输入）
    quantGRUForwardInt32(is_training, time_steps, batch_size, input_size, hidden_size,
                         W_q, R_q, bw_q, br_q,
                         x_q, h0_q_ptr,
                         quant_parms, g_blas_handle,
                         h_quant.data(), v != nullptr ? v_quant.data() : nullptr,
                         weight_ih_linear_mask, weight_hh_linear_mask, 
                         gate_input_mask, gate_output_mask, h_mask);

    // 6. 反量化输出 h
    dev::dequantification(h_quant.data(), h, h_size, toFixedScale(quant_parms.h_, usePOT2), quant_parms.h_.zero_point);

    // 7. 反量化中间值 v（如需要）
    if (v != nullptr) {
        dev::dequantificationV(v_quant.data(), v, time_steps, batch_size, hidden_size,
                               toFixedScale(quant_parms.update_gate_output_, usePOT2), quant_parms.update_gate_output_.zero_point,
                               toFixedScale(quant_parms.reset_gate_output_, usePOT2), quant_parms.reset_gate_output_.zero_point,
                               toFixedScale(quant_parms.new_gate_output_, usePOT2), quant_parms.new_gate_output_.zero_point,
                               toFixedScale(quant_parms.weight_hh_linear_, usePOT2), quant_parms.weight_hh_linear_.zero_point);
    }

    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in quantGRUForwardInt: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in quantGRUForwardInt: ") + err_str);
    }
}

// =====================================================================
// quantGRUForwardIntIO: 纯定点 int 进 int 出（上游已量化输入）
// =====================================================================
// 与 quantGRUForwardInt 的区别：
//   - 输入 x_q / h0_q 已是 int32 量化值（上游层在同一 scale_x/zp_x、scale_h/zp_h
//     网格上产生），内部不再量化输入。
//   - 仅量化权重（float master weight → int32，每次调用按 quant_params 量化）。
//   - 输出 h_q 为 int32 量化隐藏状态（不反量化）。
// 用于 AIMET INT16_FIXED_EVAL 打通的纯定点链路（int-in / int-out）。
void quantGRUForwardIntIO(
    bool is_training,
    const int time_steps, const int batch_size, const int input_size, const int hidden_size,
    const float *W, const float *R, const float *bw, const float *br,
    const int32_t *x_q, const int32_t *h0_q,
    const GRUQuantParams &quant_parms, const cublasHandle_t &g_blas_handle,
    int32_t *h_q, int32_t *v_q,
    // 权重量化 mask（训练时外部分配，推理时可为 nullptr）
    uint8_t *W_mask,
    uint8_t *R_mask,
    uint8_t *bw_mask,
    uint8_t *br_mask,
    // 计算过程 mask（训练时外部分配，推理时可为 nullptr）
    uint8_t *weight_ih_linear_mask,
    uint8_t *weight_hh_linear_mask,
    uint8_t *gate_input_mask,
    uint8_t *gate_output_mask,
    uint8_t *h_mask) {
    const int hidden3 = hidden_size * 3;

    // 1. 量化权重（int32 临时缓冲；输入 x_q/h0_q 已是量化值，无需再量化）
    dev::vector<int32_t> W_q(input_size * hidden3);
    dev::vector<int32_t> R_q(hidden_size * hidden3);
    dev::vector<int32_t> bw_q(hidden3);
    dev::vector<int32_t> br_q(hidden3);
    if (is_training) {
        quantitativeWeight<true>(input_size, hidden_size, W, R, bw, br, quant_parms,
                                 W_q.data(), R_q.data(), bw_q.data(), br_q.data(),
                                 W_mask, R_mask, bw_mask, br_mask);
    } else {
        quantitativeWeight<false>(input_size, hidden_size, W, R, bw, br, quant_parms,
                                  W_q.data(), R_q.data(), bw_q.data(), br_q.data(),
                                  nullptr, nullptr, nullptr, nullptr);
    }

    // 2. 直接调用纯定点核心（x_q / h0_q 已是 int32）
    quantGRUForwardInt32(is_training, time_steps, batch_size, input_size, hidden_size,
                         W_q.data(), R_q.data(), bw_q.data(), br_q.data(),
                         x_q, h0_q,
                         quant_parms, g_blas_handle,
                         h_q, v_q,
                         weight_ih_linear_mask, weight_hh_linear_mask,
                         gate_input_mask, gate_output_mask, h_mask);

    // 同步并检查 CUDA 错误
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in quantGRUForwardIntIO: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in quantGRUForwardIntIO: ") + err_str);
    }
}

// =====================================================================
// CPU 量化 GRU 前向传播实现
// =====================================================================

// CPU 量化 GRU 前向传播（量化权重输入版本）
// 所有量化值使用 int32_t 存储，实际值通过位宽配置限制
void quantGRUForwardCPU(bool is_training, int time_steps, int batch_size, int input_size,
                        int hidden_size, const int32_t *W, const int32_t *R, const int32_t *bw,
                        const int32_t *br, const float *x, const float *h0,
                        const GRUQuantParams &quant_parms, float *h, float *v) {
    const std::size_t x_size = time_steps * batch_size * input_size;
    const std::size_t h_total_size = (time_steps + 1) * batch_size * hidden_size;
    const auto &bw_cfg = quant_parms.bitwidth_config_;
    const bool usePOT2 = bw_cfg.usePOT2_;

    // 量化输入序列
    std::vector<int32_t> x_quant(x_size);
    for (std::size_t i = 0; i < x_size; i++) {
        x_quant[i] = quantize(x[i], toFixedScale(quant_parms.x_, usePOT2), quant_parms.x_.zero_point, bw_cfg.x_);
    }

    // 分配隐藏状态缓冲区
    std::vector<int32_t> h_quant(h_total_size);

    // 初始化 h0 为零点值
    for (int i = 0; i < batch_size * hidden_size; i++) {
        h_quant[i] = quant_parms.h_.zero_point;
    }

    // 如果提供了初始状态，量化到 h_quant[0]
    if (h0 != nullptr) {
        for (int i = 0; i < batch_size * hidden_size; i++) {
            h_quant[i] = quantize(h0[i], toFixedScale(quant_parms.h_, usePOT2), quant_parms.h_.zero_point, bw_cfg.h_);
        }
    }

    // 分配中间值缓冲区（v: update_gate, reset_gate, new_gate, weight_hh_linear_g）
    std::vector<int32_t> v_quant(time_steps * batch_size * hidden_size * 4);

    // 创建 CPU 版本的 ForwardPassQuantCPU
    cpu::ForwardPassQuantCPU forward(is_training, batch_size, input_size, hidden_size);
    forward.setRescaleParam(quant_parms);

    // 运行前向传播
    forward.Run(time_steps, W, R, bw, br, x_quant.data(), h_quant.data(),
                v != nullptr ? v_quant.data() : nullptr, 0.0f, nullptr);

    // 反量化隐藏状态输出
    for (std::size_t i = 0; i < h_total_size; i++) {
        h[i] = dequantize(h_quant[i], toFixedScale(quant_parms.h_, usePOT2), quant_parms.h_.zero_point);
    }

    // 如果需要中间值，反量化 v
    if (v != nullptr) {
        const int hidden3 = hidden_size * 3;
        for (int t = 0; t < time_steps; t++) {
            for (int b = 0; b < batch_size; b++) {
                for (int j = 0; j < hidden_size; j++) {
                    const int base_idx = (t * batch_size + b) * hidden_size * 4 + j;
                    // v[0] = update_gate, v[1] = reset_gate, v[2] = new_gate, v[3] = weight_hh_linear_g
                    v[base_idx + hidden_size * 0] = dequantize(
                        v_quant[base_idx + hidden_size * 0], toFixedScale(quant_parms.update_gate_output_, usePOT2),
                        quant_parms.update_gate_output_.zero_point);
                    v[base_idx + hidden_size * 1] = dequantize(
                        v_quant[base_idx + hidden_size * 1], toFixedScale(quant_parms.reset_gate_output_, usePOT2),
                        quant_parms.reset_gate_output_.zero_point);
                    v[base_idx + hidden_size * 2] = dequantize(
                        v_quant[base_idx + hidden_size * 2], toFixedScale(quant_parms.new_gate_output_, usePOT2),
                        quant_parms.new_gate_output_.zero_point);
                    v[base_idx + hidden_size * 3] = dequantize(
                        v_quant[base_idx + hidden_size * 3], toFixedScale(quant_parms.weight_hh_linear_, usePOT2),
                        quant_parms.weight_hh_linear_.zero_point);
                }
            }
        }
    }
}

// CPU 量化 GRU 前向传播（浮点权重输入版本，内部量化）
void quantGRUForwardCPU(bool is_training, int time_steps, int batch_size, int input_size,
                        int hidden_size, const float *W, const float *R, const float *bw,
                        const float *br, const float *x, const float *h0,
                        const GRUQuantParams &quant_parms, float *h, float *v) {
    const int hidden3 = hidden_size * 3;

    // 量化权重（使用统一接口）
    std::vector<int32_t> W_quant(input_size * hidden3);
    std::vector<int32_t> R_quant(hidden_size * hidden3);
    std::vector<int32_t> bw_quant(hidden3);
    std::vector<int32_t> br_quant(hidden3);
    
    quantitativeWeightCPU(input_size, hidden_size, W, R, bw, br, quant_parms,
                          W_quant.data(), R_quant.data(), bw_quant.data(), br_quant.data());

    // 调用量化权重版本
    quantGRUForwardCPU(is_training, time_steps, batch_size, input_size, hidden_size,
                       W_quant.data(), R_quant.data(), bw_quant.data(), br_quant.data(), x, h0,
                       quant_parms, h, v);
}

// =====================================================================
// 浮点存储版量化 GRU 前向传播（GPU-FP）
// =====================================================================

void quantGRUForwardFP(
    bool is_training,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W, const float *R, const float *bw, const float *br,
    const float *x, const float *h0,
    const GRUQuantParams &quant_params,
    const cublasHandle_t &g_blas_handle,
    float *h, float *v,
    // 输出量化后的值（必须由外部分配内存，训练和推理模式都需要）
    // 函数会直接写入这些指针指向的内存，无需拷贝
    float *W_q_out,
    float *R_q_out,
    float *bw_q_out,
    float *br_q_out,
    float *x_q_out,
    // 输入量化 mask（训练时外部分配，推理时可为 nullptr）
    uint8_t *x_mask,
    uint8_t *h0_mask,
    uint8_t *W_mask,
    uint8_t *R_mask,
    uint8_t *bw_mask,
    uint8_t *br_mask,
    // 计算过程 mask（训练时外部分配，推理时可为 nullptr）
    uint8_t *weight_ih_linear_mask,
    uint8_t *weight_hh_linear_mask,
    uint8_t *gate_input_mask,
    uint8_t *gate_output_mask,
    uint8_t *h_mask) {
    
    const int hidden3 = hidden_size * 3;
    const auto &bw_cfg = quant_params.bitwidth_config_;
    const bool usePOT2 = bw_cfg.usePOT2_;
    const std::size_t x_size = time_steps * batch_size * input_size;
    const std::size_t h_size = (time_steps + 1) * batch_size * hidden_size;
    const int NH = batch_size * hidden_size;
    
    // 量化权重（W, R, bw, br）- 使用统一接口，内部根据 granularity 自动选择
    if (is_training) {
        quantizeGRUWeights<true>(W, R, bw, br,
                                 W_q_out, R_q_out, bw_q_out, br_q_out,
                                 W_mask, R_mask, bw_mask, br_mask,
                                 input_size, hidden_size, quant_params);
        // x (始终 per-tensor)
        dev::quantificationFP<true>(x, x_q_out, x_mask, x_size, toFixedScale(quant_params.x_, usePOT2),
                                    quant_params.x_.zero_point, bw_cfg.x_);
    } else {
        quantizeGRUWeights<false>(W, R, bw, br,
                                  W_q_out, R_q_out, bw_q_out, br_q_out,
                                  nullptr, nullptr, nullptr, nullptr,
                                  input_size, hidden_size, quant_params);
        // x (始终 per-tensor)
        dev::quantificationFP<false>(x, x_q_out, nullptr, x_size, toFixedScale(quant_params.x_, usePOT2),
                                     quant_params.x_.zero_point, bw_cfg.x_);
    }
    
    // 量化 h0 到 h 的前 NH 个元素（直接使用外部 h 缓冲区）
    if (h0 != nullptr) {
        if (is_training) {
            dev::quantificationFP<true>(h0, h, h0_mask, NH, toFixedScale(quant_params.h_, usePOT2),
                                       quant_params.h_.zero_point, bw_cfg.h_);
        } else {
            dev::quantificationFP<false>(h0, h, nullptr, NH, toFixedScale(quant_params.h_, usePOT2),
                                        quant_params.h_.zero_point, bw_cfg.h_);
        }
    } else {
        // 填充零点值（表示初始隐状态为 0）
        dev::fill_n(h, NH, static_cast<float>(quant_params.h_.zero_point));
    }
    
    // 前向传播：直接使用外部 h 和 v 缓冲区（会写入量化值）
    gru::ForwardPassQuantFP forward_fp(is_training, batch_size, input_size, hidden_size,
                                       g_blas_handle, nullptr);
    forward_fp.setRescaleParam(quant_params);
    forward_fp.Run(time_steps, W_q_out, R_q_out,
                   bw_q_out, br_q_out,
                   x_q_out, h,
                   v,
                   0.0f, nullptr,
                   weight_ih_linear_mask,
                   weight_hh_linear_mask,
                   gate_input_mask,
                   gate_output_mask,
                   h_mask);
    
    // 原地反量化输出
    dev::dequantificationFPInplace(h, h_size, toFixedScale(quant_params.h_, usePOT2), quant_params.h_.zero_point);
    if (v != nullptr) {
        dev::dequantificationVFPInplace(v, time_steps, batch_size, hidden_size,
                                         toFixedScale(quant_params.update_gate_output_, usePOT2), quant_params.update_gate_output_.zero_point,
                                         toFixedScale(quant_params.reset_gate_output_, usePOT2), quant_params.reset_gate_output_.zero_point,
                                         toFixedScale(quant_params.new_gate_output_, usePOT2), quant_params.new_gate_output_.zero_point,
                                         toFixedScale(quant_params.weight_hh_linear_, usePOT2), quant_params.weight_hh_linear_.zero_point);
    }
    
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in quantGRUForwardFP: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in quantGRUForwardFP: ") + err_str);
    }
}

// =====================================================================
// GPU 直方图收集器转换函数
// =====================================================================

/**
 * @brief 将 GPU 直方图收集器转换为 CPU 版本
 * 
 * 用于与现有的 calculateGRUQuantitativeParametersFromHistograms 兼容
 */
GRUHistogramCollectors convertGPUHistogramsToCPU(const GRUGPUHistogramCollectors &gpu_collectors) {
    GRUHistogramCollectors cpu_collectors(gpu_collectors.hidden_, gpu_collectors.num_bins_);

    // 转换标量直方图
    auto convert_collector = [](HistogramCollector &dst, const GPUHistogramCollector &src) {
        if (src.is_valid()) {
            dst.histogram() = gpu_histogram_to_cpu(src.histogram());
        }
    };

    convert_collector(cpu_collectors.x_hist, gpu_collectors.x_hist);
    convert_collector(cpu_collectors.h_hist, gpu_collectors.h_hist);
    convert_collector(cpu_collectors.Wx_hist, gpu_collectors.Wx_hist);
    convert_collector(cpu_collectors.Rh_hist, gpu_collectors.Rh_hist);
    convert_collector(cpu_collectors.update_gate_input_hist, gpu_collectors.update_gate_input_hist);
    convert_collector(cpu_collectors.reset_gate_input_hist, gpu_collectors.reset_gate_input_hist);
    convert_collector(cpu_collectors.new_gate_input_hist, gpu_collectors.new_gate_input_hist);
    convert_collector(cpu_collectors.update_gate_output_hist, gpu_collectors.update_gate_output_hist);
    convert_collector(cpu_collectors.reset_gate_output_hist, gpu_collectors.reset_gate_output_hist);
    convert_collector(cpu_collectors.new_gate_output_hist, gpu_collectors.new_gate_output_hist);
    convert_collector(cpu_collectors.mul_reset_hidden_hist, gpu_collectors.mul_reset_hidden_hist);
    convert_collector(cpu_collectors.mul_new_contribution_hist, gpu_collectors.mul_new_contribution_hist);
    convert_collector(cpu_collectors.mul_old_contribution_hist, gpu_collectors.mul_old_contribution_hist);

    // 转换 per-channel 直方图（从批量结构）
    auto convert_batch = [](std::vector<HistogramCollector> &dst, 
                            const PerChannelHistogramBatch &src) {
        if (!src.is_valid()) return;
        
        // 一次性读取所有 counts 到 CPU
        std::vector<float> all_counts = src.all_counts_to_host();
        
        for (int c = 0; c < src.channel_size; ++c) {
            Histogram& hist = dst[c].histogram();
            hist.num_bins = src.num_bins;
            hist.min_val = src.mins[c];
            hist.max_val = src.maxs[c];
            hist.total_count = src.per_channel_count;
            hist.counts.assign(all_counts.begin() + c * src.num_bins,
                              all_counts.begin() + (c + 1) * src.num_bins);
        }
    };

    convert_batch(cpu_collectors.W_hist, gpu_collectors.W_batch);
    convert_batch(cpu_collectors.R_hist, gpu_collectors.R_batch);
    convert_batch(cpu_collectors.bw_hist, gpu_collectors.bw_batch);
    convert_batch(cpu_collectors.br_hist, gpu_collectors.br_batch);
    
    // 转换 per-tensor 直方图
    convert_collector(cpu_collectors.W_tensor_hist, gpu_collectors.W_tensor_hist);
    convert_collector(cpu_collectors.R_tensor_hist, gpu_collectors.R_tensor_hist);
    convert_collector(cpu_collectors.bw_tensor_hist, gpu_collectors.bw_tensor_hist);
    convert_collector(cpu_collectors.br_tensor_hist, gpu_collectors.br_tensor_hist);
    
    // 转换 per-gate 直方图
    for (int i = 0; i < 3; ++i) {
        convert_collector(cpu_collectors.W_gate_hist[i], gpu_collectors.W_gate_hist[i]);
        convert_collector(cpu_collectors.R_gate_hist[i], gpu_collectors.R_gate_hist[i]);
        convert_collector(cpu_collectors.bw_gate_hist[i], gpu_collectors.bw_gate_hist[i]);
        convert_collector(cpu_collectors.br_gate_hist[i], gpu_collectors.br_gate_hist[i]);
    }

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in convertGPUHistogramsToCPU: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in convertGPUHistogramsToCPU: ") + err_str);
    }

    return cpu_collectors;
}

GRUQuantParams calculateGRUQuantitativeParametersFromHistograms(
    const GRUHistogramCollectors &hist_collectors, const OperatorQuantConfig &bitwidth_config,
    bool use_percentile, float percentile_value) {
    ensureSymmetricWeightBiasConfig(bitwidth_config);
    GRUQuantParams quant_params;
    quant_params.hidden_ = hist_collectors.hidden_;
    quant_params.bitwidth_config_ = bitwidth_config;
    const bool usePOT2 = bitwidth_config.usePOT2_;
    const PotScaleMethod potMethod = bitwidth_config.pot_scale_method_;
    const float potTol = bitwidth_config.pot_scale_tolerance_;

    const int channel_size = hist_collectors.hidden_ * 3;
    const int hidden_size = hist_collectors.hidden_;
    
    // 辅助 lambda：校准单个直方图 -> 唯一权威 QuantParam（scale + zp）
    auto histCalibrate = [&](const HistogramCollector& hist, QuantBitWidth bw, bool sym,
                             const char* name) -> QuantParam {
        if (!hist.is_valid()) throw std::runtime_error(std::string(name) + " is invalid.");
        ContinuousScaleResult continuous = calibrateContinuousScaleFromHistogram(
            hist.histogram(), bw, sym, use_percentile, percentile_value);
        EncodedScaleResult encoded = encodeScaleResult(
            continuous, bw, sym, usePOT2, potMethod, potTol);
        return QuantParam{storedScaleForMode(encoded, usePOT2), encoded.zero_point};
    };

    // 权重/偏置：唯一 per-channel 权威数组，按粒度从对应直方图广播
    auto histCalibrateChannel = [&](ChannelQuantParam& out,
                                    OperatorQuantConfig::QuantizationGranularity gran,
                                    QuantBitWidth bw, bool sym,
                                    const HistogramCollector& tensor_hist,
                                    const std::array<HistogramCollector, 3>& gate_hist,
                                    const std::vector<HistogramCollector>& chan_hist,
                                    const char* name) {
        out.granularity = gran;
        out.hidden = hidden_size;
        out.resize(channel_size);
        if (gran == OperatorQuantConfig::PER_TENSOR) {
            const QuantParam qp = histCalibrate(tensor_hist, bw, sym, name);
            std::fill(out.channels.begin(), out.channels.end(), qp);
        } else if (gran == OperatorQuantConfig::PER_GATE) {
            for (int gate = 0; gate < 3; ++gate) {
                const QuantParam qp = histCalibrate(gate_hist[gate], bw, sym, name);
                for (int c = gate * hidden_size; c < (gate + 1) * hidden_size; ++c) out.channel(c) = qp;
            }
        } else {  // PER_CHANNEL
            for (int c = 0; c < channel_size; ++c) {
                out.channel(c) = histCalibrate(chan_hist[c], bw, sym, name);
            }
        }
    };

    histCalibrateChannel(quant_params.W_, bitwidth_config.W_granularity_, bitwidth_config.W_,
        bitwidth_config.W_symmetric_, hist_collectors.W_tensor_hist, hist_collectors.W_gate_hist, hist_collectors.W_hist, "W");
    histCalibrateChannel(quant_params.R_, bitwidth_config.R_granularity_, bitwidth_config.R_,
        bitwidth_config.R_symmetric_, hist_collectors.R_tensor_hist, hist_collectors.R_gate_hist, hist_collectors.R_hist, "R");
    histCalibrateChannel(quant_params.bw_, bitwidth_config.bw_granularity_, bitwidth_config.bw_,
        bitwidth_config.bw_symmetric_, hist_collectors.bw_tensor_hist, hist_collectors.bw_gate_hist, hist_collectors.bw_hist, "bw");
    histCalibrateChannel(quant_params.br_, bitwidth_config.br_granularity_, bitwidth_config.br_,
        bitwidth_config.br_symmetric_, hist_collectors.br_tensor_hist, hist_collectors.br_gate_hist, hist_collectors.br_hist, "br");

    // 标量参数（不受 granularity 影响）
    quant_params.x_ = histCalibrate(hist_collectors.x_hist, bitwidth_config.x_, bitwidth_config.x_symmetric_, "scale_x");
    quant_params.h_ = histCalibrate(hist_collectors.h_hist, bitwidth_config.h_, bitwidth_config.h_symmetric_, "scale_h");
    quant_params.weight_ih_linear_ = histCalibrate(hist_collectors.Wx_hist, bitwidth_config.weight_ih_linear_, bitwidth_config.weight_ih_linear_symmetric_, "scale_weight_ih_linear");
    quant_params.weight_hh_linear_ = histCalibrate(hist_collectors.Rh_hist, bitwidth_config.weight_hh_linear_, bitwidth_config.weight_hh_linear_symmetric_, "scale_weight_hh_linear");
    quant_params.update_gate_input_ = histCalibrate(hist_collectors.update_gate_input_hist, bitwidth_config.update_gate_input_, bitwidth_config.update_gate_input_symmetric_, "scale_update_gate_input");
    quant_params.reset_gate_input_ = histCalibrate(hist_collectors.reset_gate_input_hist, bitwidth_config.reset_gate_input_, bitwidth_config.reset_gate_input_symmetric_, "scale_reset_gate_input");
    quant_params.new_gate_input_ = histCalibrate(hist_collectors.new_gate_input_hist, bitwidth_config.new_gate_input_, bitwidth_config.new_gate_input_symmetric_, "scale_new_gate_input");
    quant_params.update_gate_output_ = histCalibrate(hist_collectors.update_gate_output_hist, bitwidth_config.update_gate_output_, bitwidth_config.update_gate_output_symmetric_, "scale_update_gate_output");
    quant_params.reset_gate_output_ = histCalibrate(hist_collectors.reset_gate_output_hist, bitwidth_config.reset_gate_output_, bitwidth_config.reset_gate_output_symmetric_, "scale_reset_gate_output");
    quant_params.new_gate_output_ = histCalibrate(hist_collectors.new_gate_output_hist, bitwidth_config.new_gate_output_, bitwidth_config.new_gate_output_symmetric_, "scale_new_gate_output");
    quant_params.mul_reset_hidden_ = histCalibrate(hist_collectors.mul_reset_hidden_hist, bitwidth_config.mul_reset_hidden_, bitwidth_config.mul_reset_hidden_symmetric_, "scale_mul_reset_hidden");
    quant_params.mul_new_contribution_ = histCalibrate(hist_collectors.mul_new_contribution_hist, bitwidth_config.mul_new_contribution_, bitwidth_config.mul_new_contribution_symmetric_, "scale_mul_new_contribution");
    quant_params.mul_old_contribution_ = histCalibrate(hist_collectors.mul_old_contribution_hist, bitwidth_config.mul_old_contribution_, bitwidth_config.mul_old_contribution_symmetric_, "scale_mul_old_contribution");

    generate_piecewise_linear_lut_to_params(quant_params);
    return quant_params;
}

/**
 * @brief 从 GPU 直方图收集器计算量化参数（GPU 加速 SQNR）
 *
 * 直接使用 GPU 上的直方图数据计算 SQNR，避免 GPU→CPU 传输
 */
GRUQuantParams calculateGRUQuantitativeParametersFromGPUHistograms(
    GRUGPUHistogramCollectors &gpu_collectors, const OperatorQuantConfig &bitwidth_config) {
    ensureSymmetricWeightBiasConfig(bitwidth_config);
    
    GRUQuantParams quant_params;
    quant_params.hidden_ = gpu_collectors.hidden_;
    quant_params.bitwidth_config_ = bitwidth_config;
    const bool usePOT2 = bitwidth_config.usePOT2_;
    const PotScaleMethod potMethod = bitwidth_config.pot_scale_method_;
    const float potTol = bitwidth_config.pot_scale_tolerance_;
    
    const int channel_size = gpu_collectors.hidden_ * 3;
    const int hidden_size = gpu_collectors.hidden_;

    // Helper: 由连续 scale 编码为唯一权威 QuantParam
    auto sqnrEncode = [&](const ContinuousScaleResult& cont, QuantBitWidth bw, bool sym) -> QuantParam {
        EncodedScaleResult encoded =
            encodeScaleResult(cont, bw, sym, usePOT2, potMethod, potTol);
        return QuantParam{storedScaleForMode(encoded, usePOT2), encoded.zero_point};
    };

    // Helper: 计算单个标量直方图的 SQNR 参数 -> QuantParam（无效直接抛错）
    auto sqnrScalar = [&](const GPUHistogramCollector& collector, bool is_symmetric,
                          QuantBitWidth quant_bw, const char* name) -> QuantParam {
        if (!collector.is_valid()) {
            throw std::runtime_error(std::string("GPU histogram ") + (name ? name : "unknown") + " is invalid");
        }
        const auto& hist = collector.histogram();
        const int64_t num_steps = static_cast<int64_t>(quant_bw.qmax_auto_scale()) - static_cast<int64_t>(quant_bw.qmin_auto_scale());
        const bool is_unsigned = quant_bw.is_unsigned_;
        ContinuousScaleResult cont = gpu_hist::searchSqnrGpu(
            hist.counts.data(), hist.min_val, hist.max_val,
            hist.num_bins, num_steps, is_symmetric, SqnrConfig(), is_unsigned);
        return sqnrEncode(cont, quant_bw, is_symmetric);
    };

    // Helper: 权重 tensor/gate 直方图无效时优雅降级（scale=1, zp=0），保持旧行为
    auto sqnrWeightScalar = [&](const GPUHistogramCollector& collector, bool is_symmetric,
                                QuantBitWidth quant_bw, const char* name) -> QuantParam {
        if (!collector.is_valid()) {
            fprintf(stderr, "Warning: %s histogram is invalid, scale will default to 1.0\n", name);
            return QuantParam{1.0f, 0};
        }
        return sqnrScalar(collector, is_symmetric, quant_bw, name);
    };

    // 标量直方图
    quant_params.x_ = sqnrScalar(gpu_collectors.x_hist, bitwidth_config.x_symmetric_, bitwidth_config.x_, "x");
    quant_params.h_ = sqnrScalar(gpu_collectors.h_hist, bitwidth_config.h_symmetric_, bitwidth_config.h_, "h");
    quant_params.weight_ih_linear_ = sqnrScalar(gpu_collectors.Wx_hist, bitwidth_config.weight_ih_linear_symmetric_, bitwidth_config.weight_ih_linear_, "weight_ih_linear");
    quant_params.weight_hh_linear_ = sqnrScalar(gpu_collectors.Rh_hist, bitwidth_config.weight_hh_linear_symmetric_, bitwidth_config.weight_hh_linear_, "weight_hh_linear");
    quant_params.update_gate_input_ = sqnrScalar(gpu_collectors.update_gate_input_hist, bitwidth_config.update_gate_input_symmetric_, bitwidth_config.update_gate_input_, "update_gate_input");
    quant_params.reset_gate_input_ = sqnrScalar(gpu_collectors.reset_gate_input_hist, bitwidth_config.reset_gate_input_symmetric_, bitwidth_config.reset_gate_input_, "reset_gate_input");
    quant_params.new_gate_input_ = sqnrScalar(gpu_collectors.new_gate_input_hist, bitwidth_config.new_gate_input_symmetric_, bitwidth_config.new_gate_input_, "new_gate_input");
    quant_params.update_gate_output_ = sqnrScalar(gpu_collectors.update_gate_output_hist, bitwidth_config.update_gate_output_symmetric_, bitwidth_config.update_gate_output_, "update_gate_output");
    quant_params.reset_gate_output_ = sqnrScalar(gpu_collectors.reset_gate_output_hist, bitwidth_config.reset_gate_output_symmetric_, bitwidth_config.reset_gate_output_, "reset_gate_output");
    quant_params.new_gate_output_ = sqnrScalar(gpu_collectors.new_gate_output_hist, bitwidth_config.new_gate_output_symmetric_, bitwidth_config.new_gate_output_, "new_gate_output");
    quant_params.mul_reset_hidden_ = sqnrScalar(gpu_collectors.mul_reset_hidden_hist, bitwidth_config.mul_reset_hidden_symmetric_, bitwidth_config.mul_reset_hidden_, "mul_reset_hidden");
    quant_params.mul_new_contribution_ = sqnrScalar(gpu_collectors.mul_new_contribution_hist, bitwidth_config.mul_new_contribution_symmetric_, bitwidth_config.mul_new_contribution_, "mul_new_contribution");
    quant_params.mul_old_contribution_ = sqnrScalar(gpu_collectors.mul_old_contribution_hist, bitwidth_config.mul_old_contribution_symmetric_, bitwidth_config.mul_old_contribution_, "mul_old_contribution");
    
    // 权重/偏置：唯一 per-channel 权威数组，按粒度从对应直方图广播
    auto sqnrChannel = [&](ChannelQuantParam& out,
                           OperatorQuantConfig::QuantizationGranularity gran,
                           bool is_symmetric, QuantBitWidth quant_bw,
                           const GPUHistogramCollector& tensor_hist,
                           const std::array<GPUHistogramCollector, 3>& gate_hist,
                           const PerChannelHistogramBatch& batch,
                           const char* name) {
        out.granularity = gran;
        out.hidden = hidden_size;
        out.resize(channel_size);
        if (gran == OperatorQuantConfig::PER_TENSOR) {
            const QuantParam qp = sqnrWeightScalar(tensor_hist, is_symmetric, quant_bw, name);
            std::fill(out.channels.begin(), out.channels.end(), qp);
        } else if (gran == OperatorQuantConfig::PER_GATE) {
            for (int gate = 0; gate < 3; ++gate) {
                const QuantParam qp = sqnrWeightScalar(gate_hist[gate], is_symmetric, quant_bw, name);
                for (int c = gate * hidden_size; c < (gate + 1) * hidden_size; ++c) out.channel(c) = qp;
            }
        } else {  // PER_CHANNEL
            if (!batch.is_valid()) return;
            const int64_t num_steps = static_cast<int64_t>(quant_bw.qmax_auto_scale()) - static_cast<int64_t>(quant_bw.qmin_auto_scale());
            const bool is_unsigned = quant_bw.is_unsigned_;
            std::vector<ContinuousScaleResult> continuous_results;
            gpu_hist::searchSqnrPerChannelGpu(batch, num_steps, is_symmetric, continuous_results,
                                              SqnrConfig(), is_unsigned);
            for (int c = 0; c < batch.channel_size; ++c) {
                out.channel(c) = sqnrEncode(continuous_results[c], quant_bw, is_symmetric);
            }
        }
    };

    sqnrChannel(quant_params.W_, bitwidth_config.W_granularity_, bitwidth_config.W_symmetric_, bitwidth_config.W_,
        gpu_collectors.W_tensor_hist, gpu_collectors.W_gate_hist, gpu_collectors.W_batch, "W");
    sqnrChannel(quant_params.R_, bitwidth_config.R_granularity_, bitwidth_config.R_symmetric_, bitwidth_config.R_,
        gpu_collectors.R_tensor_hist, gpu_collectors.R_gate_hist, gpu_collectors.R_batch, "R");
    sqnrChannel(quant_params.bw_, bitwidth_config.bw_granularity_, bitwidth_config.bw_symmetric_, bitwidth_config.bw_,
        gpu_collectors.bw_tensor_hist, gpu_collectors.bw_gate_hist, gpu_collectors.bw_batch, "bw");
    sqnrChannel(quant_params.br_, bitwidth_config.br_granularity_, bitwidth_config.br_symmetric_, bitwidth_config.br_,
        gpu_collectors.br_tensor_hist, gpu_collectors.br_gate_hist, gpu_collectors.br_batch, "br");

    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    // 生成 LUT 并存储到参数中
    generate_piecewise_linear_lut_to_params(quant_params);
    
    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in calculateGRUQuantitativeParametersFromGPUHistograms: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error: ") + err_str);
    }

    return quant_params;
}

// Percentile 校准：使用 calculateGRUQuantitativeParametersFromHistograms() 并设置 use_percentile=true

// =====================================================================
// 统一校准前向传播（GPU）
// =====================================================================

void forwardWithCalibrationGPU(
    bool is_training,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W, const float *R, const float *bw, const float *br, const float *x,
    const float *h0,
    const cublasHandle_t &g_blas_handle,
    CalibrationMethod calib_method,
    GRUQuantizationRanges *quant_ranges,
    GRUGPUHistogramCollectors *gpu_hist_collectors,
    const OperatorQuantConfig &bitwidth_config,
    float *h, float *v) {
    
    // 参数校验
    if (calib_method == CalibrationMethod::MINMAX) {
        if (!quant_ranges) {
            throw std::invalid_argument("quant_ranges is required for MINMAX calibration");
        }
    } else if (calib_method == CalibrationMethod::SQNR || 
               calib_method == CalibrationMethod::PERCENTILE) {
        if (!gpu_hist_collectors) {
            throw std::invalid_argument("gpu_hist_collectors is required for SQNR/PERCENTILE calibration");
        }
        // 检查/重置直方图收集器
        if (gpu_hist_collectors->hidden_ != hidden_size) {
            gpu_hist_collectors->reset(hidden_size);
        }
    } else {
        throw std::invalid_argument("Invalid calibration method (must be MINMAX, SQNR, or PERCENTILE)");
    }
    
    // ========== 公共部分 ==========
    
    // 分配临时缓冲区
    dev::vector<float> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> tmp_Rh_dev(time_steps * batch_size * hidden_size * 3);

    // 处理初始隐藏状态
    const int NH = batch_size * hidden_size;
    if (h0 != nullptr) {
        // 如果提供了初始状态，复制到 h[0]
        d2d(h, h0, NH);
    } else {
        // 否则初始化为零
        cudaMemset(h, 0, NH * sizeof(float));
    }

    // 创建 ForwardPass 对象
    gru::ForwardPass<float> forward(is_training, batch_size, input_size, hidden_size, g_blas_handle);

    // 设置校准模式
    forward.setCalibrationMode(true);

    // 执行前向传播
    forward.Run(time_steps, W, R, bw, br, x, h, v, tmp_Wx_dev.data(), tmp_Rh_dev.data(), 0.0f, nullptr);

    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in forwardWithCalibrationGPU: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in forwardWithCalibrationGPU: ") + err_str);
    }
    
    // ========== 后处理（根据校准方法分发）==========
    
    if (calib_method == CalibrationMethod::MINMAX) {
        // MINMAX: 使用 GPU 版本原地更新量化范围（避免大量 D2H 传输）
        // 使用 Wx+bw 和 Rh+br 的结果（而非纯 GEMM 输出）进行校准
        updateGRUQuantizationRangesGPU(
            time_steps, batch_size, input_size, hidden_size,
            W, R, bw, br, x, h, v,
            forward.getWxAddBw(), forward.getRhAddBr(),
            forward.getZPres(), forward.getRPres(), forward.getGPres(),
            forward.getPresSize(),
            *quant_ranges);
    } else {
        // SQNR/Percentile: 收集直方图
        // 使用 Wx+bw 和 Rh+br 的结果（而非纯 GEMM 输出）进行直方图收集
        collectAllHistogramsGPU(*gpu_hist_collectors, x, h, v,
                                forward.getWxAddBw(), forward.getRhAddBr(),
                                W, R, bw, br,
                                time_steps, batch_size, input_size, hidden_size,
                                forward.getZPres(), forward.getRPres(), forward.getGPres(),
                                forward.getPresSize(), bitwidth_config);
    }
}

