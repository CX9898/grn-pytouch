#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "quantize_bitwidth_config.h"
#include "quantize_lut_types.h"   // 复用 LUT 结构体和生成函数（CPU/GPU 共用）
#include "quantize_ops_helper.h"  // 复用 rshift_round, quantize_* 等函数
#include "scale_encoding.h"       // encodeMShift / toFixedScale(s) / makeRescale*（setRescaleParam 用）

// ============================================================================
// gru_quant_cpu.h - 纯 C++ 定点 GRU 前向传播接口
// ============================================================================
//
// 设计说明:
//   - 完全不依赖 CUDA，可在任意 CPU 平台运行
//   - 与 GPU 版本保持数值一致性（用于验证和部署）
//   - 所有量化值使用 int32_t 统一存储，通过 bitwidth_config_ 控制实际位宽
//   - 复用 quantize_lut_types.h 中的 LUT 结构和生成函数
//   - 复用 quantize_ops_helper.h 中的通用函数
//
// ============================================================================

namespace cpu {

// ==================== CPU 专用辅助函数 ====================
// 以下函数已统一定义在 quantize_ops_helper.h 中，使用 __host__ __device__ 标记：
//   - rshift_round (int32_t / int64_t)
//   - clamp_by_bitwidth
//   - find_segment
//   - piecewise_linear_raw
//   - piecewise_linear


// ==================== CPU 量化参数 ====================
// 使用拆分设计：
//   - GateQuantParams: 门计算参数（纯标量，CPU/GPU 共用）
//   - LinearQuantParamsCPU: Linear 层 per-channel 参数（CPU 版本，std::vector）

// ==================== 前向传播类声明 ====================

/// @brief CPU 版本量化 GRU 前向传播类（非模板，统一 int32_t 存储）
class ForwardPassQuantCPU {
   public:
    ForwardPassQuantCPU(bool training, int batch_size, int input_size, int hidden_size);
    ~ForwardPassQuantCPU();

    /// @brief 设置量化 rescale 参数（使用 quantize_ops_helper.h 中定义的通用结构）
    void setRescaleParam(const GRUQuantParams &params);

    /// @brief 执行前向传播（所有量化值使用 int32_t 存储）
    void Run(int steps, const int32_t *W, const int32_t *R, const int32_t *bw, const int32_t *br,
             const int32_t *x, int32_t *h, int32_t *v, float zoneout_prob,
             const int32_t *zoneout_mask);

   private:
    struct PrivateData;
    std::unique_ptr<PrivateData> data_;

    // -------------------- 量化参数（类型驱动派发，只填充生效的那套）--------------------
    bool use_pot2_ = false;                ///< setRescaleParam 时确定的运行时模式
    OperatorQuantConfig bitwidth_config_;  ///< 位宽配置（缓冲区分配等通用逻辑使用）
    GateQuantParamsPot2 gate_params_pot2_;
    GateQuantParamsMShift gate_params_mshift_;
    LinearQuantParamsCPUT<Pot2Rescale> linear_params_pot2_;
    LinearQuantParamsCPUT<FixedPointScale> linear_params_mshift_;

    int max_steps_ = 0;
    std::vector<int32_t> tmp_weight_ih_linear_;  // Linear 变换: W*x + bw
    std::vector<int32_t> tmp_weight_hh_linear_;  // Linear 变换: R*h + br
    std::vector<int64_t> W_sum_mul_x_zp_;
    std::vector<int64_t> R_sum_mul_h_zp_;
    bool weight_sums_computed_ = false;
    const int32_t *cached_W_ = nullptr;
    const int32_t *cached_R_ = nullptr;

    void EnsureBuffersAllocated(int steps);
    void PrecomputeWeightSums(const int32_t *W, const int32_t *R);

    // 按 rescale 表示 R 模板化的执行实现（R = Pot2Rescale / FixedPointScale）。
    template <class R>
    void RunImpl(const LinearQuantParamsCPUT<R> &lp, const GateQuantParamsT<R> &gp,
                 int steps, const int32_t *W, const int32_t *R_w, const int32_t *bw,
                 const int32_t *br, const int32_t *x, int32_t *h, int32_t *v,
                 float zoneout_prob, const int32_t *zoneout_mask);
    template <class R>
    void ComputeLinearX(const LinearQuantParamsCPUT<R> &lp, const GateQuantParamsT<R> &gp,
                        const int32_t *W, const int32_t *x, const int32_t *bw, int steps);
    template <class R>
    void ComputeLinearH(const LinearQuantParamsCPUT<R> &lp, const GateQuantParamsT<R> &gp,
                        const int32_t *R_w, const int32_t *h, const int32_t *br);
    template <class R>
    void IterateInternal(const LinearQuantParamsCPUT<R> &lp, const GateQuantParamsT<R> &gp,
                         const int32_t *R_w, const int32_t *br, const int32_t *h,
                         int32_t *h_out, int32_t *v, const int32_t *cur_linear_x,
                         float zoneout_prob, const int32_t *zoneout_mask);
};

}  // namespace cpu
