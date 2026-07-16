#pragma once

#include <cstdint>
#include "cuda_compat.h"
#include "pot_scale_encode.h"  // for PotScaleMethod（scale 编码策略字段类型）

struct QuantBitWidth {
    int8_t bits_ = 8;
    bool is_unsigned_ = false;  // false=INT(有符号), true=UINT(无符号)

    QuantBitWidth() = default;
    __host__ __device__ QuantBitWidth(int8_t b, bool is_unsigned = false) 
        : bits_(b), is_unsigned_(b >= 32 ? false : is_unsigned) {}

    __host__ __device__ int32_t qmin() const {
        if (bits_ >= 32) return static_cast<int32_t>(0x80000000);
        return is_unsigned_ ? 0 : -(1 << (bits_ - 1));
    }
    
    __host__ __device__ int32_t qmax() const {
        if (bits_ >= 32) return 0x7FFFFFFF;
        return is_unsigned_ ? (1 << bits_) - 1 : (1 << (bits_ - 1)) - 1;
    }
    
    // Auto scale专用的qmin函数
    // 所有位宽：qmin保持不变（与qmin()相同）
    __host__ __device__ int32_t qmin_auto_scale() const {
        if (bits_ >= 32) return static_cast<int32_t>(0x80000000);
        return is_unsigned_ ? 0 : -(1 << (bits_ - 1));
    }
    
    // Auto scale专用的qmax函数
    // 与 AIMET 对齐：num_steps = qmax_auto_scale - qmin_auto_scale = 2^bits - 1
    // 有符号数：qmax = (1 << (bits_ - 1)) - 1   （int8: 127, int4: 7）
    // 无符号数：qmax = (1 << bits_) - 1         （uint8: 255, uint4: 15）
    // 注意：此处必须与 qmax() 保持一致，否则计算出的 scale 会与 AIMET（及下一层）不一致。
    // 历史实现曾为对称性返回 1<<(bits_-1)（int8: 128），导致 num_steps=256≠AIMET 的 255，
    // 进而 scale=range/256≠range/255，造成上下层 scale 不一致，现已修正。
    __host__ __device__ int32_t qmax_auto_scale() const {
        if (bits_ >= 32) return 0x7FFFFFFF;
        return is_unsigned_ ? (1 << bits_) - 1 : (1 << (bits_ - 1)) - 1;
    }
    
    __host__ __device__ int64_t range() const {
        return static_cast<int64_t>(qmax()) - static_cast<int64_t>(qmin());
    }

    bool operator==(const QuantBitWidth& other) const {
        return bits_ == other.bits_ && is_unsigned_ == other.is_unsigned_;
    }
    bool operator!=(const QuantBitWidth& other) const {
        return !(*this == other);
    }
};

struct OperatorQuantConfig {
    // ========== 量化粒度控制（仅对 W, R, bw, br 有效）==========
    enum QuantizationGranularity {
        PER_TENSOR = 0,
        PER_GATE = 1,
        PER_CHANNEL = 2
    };
    
    QuantizationGranularity W_granularity_ = PER_CHANNEL;  // 默认 per-channel
    QuantizationGranularity R_granularity_ = PER_CHANNEL;
    QuantizationGranularity bw_granularity_ = PER_CHANNEL;
    QuantizationGranularity br_granularity_ = PER_CHANNEL;

    QuantBitWidth x_{8, false}, h_{8, false};
    QuantBitWidth W_{8, false}, R_{8, false}, bw_{8, false}, br_{8, false};
    QuantBitWidth weight_ih_linear_{8, false}, weight_hh_linear_{8, false};  // GEMM+bias 融合输出
    QuantBitWidth update_gate_input_{8, false}, update_gate_output_{8, true};   // update_gate_output: UINT
    QuantBitWidth reset_gate_input_{8, false}, reset_gate_output_{8, true};     // reset_gate_output: UINT
    QuantBitWidth new_gate_input_{8, false}, new_gate_output_{8, false};
    QuantBitWidth mul_reset_hidden_{8, false};  // r * weight_hh_linear (new gate中)
    QuantBitWidth mul_old_contribution_{8, false}, mul_new_contribution_{8, false};

    bool x_symmetric_ = false, h_symmetric_ = false;
    bool W_symmetric_ = true, R_symmetric_ = true, bw_symmetric_ = true, br_symmetric_ = true;
    bool weight_ih_linear_symmetric_ = false, weight_hh_linear_symmetric_ = false;
    bool update_gate_input_symmetric_ = false, update_gate_output_symmetric_ = false;
    bool reset_gate_input_symmetric_ = false, reset_gate_output_symmetric_ = false;
    bool new_gate_input_symmetric_ = false, new_gate_output_symmetric_ = true;
    bool mul_reset_hidden_symmetric_ = false;
    bool mul_old_contribution_symmetric_ = false, mul_new_contribution_symmetric_ = false;
    // true: POT2 (multiplier=1, shift-only fast path), false: affine M+shift
    bool usePOT2_ = false;
    // POT2 取整策略（仅 usePOT2_=true 时生效）；默认 CoverRange，与 AIMET apply_power_of_2 一致
    PotScaleMethod pot_scale_method_ = PotScaleMethod::CoverRange;
    // CoverRange 下判断 real_range 是否接近 2^k 的相对容差
    float pot_scale_tolerance_ = 0.02f;

    OperatorQuantConfig& setAllBitWidths(int8_t bits) {
        QuantBitWidth* signed_members[] = {
            &x_, &h_, &W_, &R_, &bw_, &br_, &weight_ih_linear_, &weight_hh_linear_,
            &update_gate_input_, &reset_gate_input_, &new_gate_input_, &new_gate_output_,
            &mul_reset_hidden_, &mul_old_contribution_, &mul_new_contribution_
        };
        for (auto* m : signed_members) {
            m->bits_ = bits;
            m->is_unsigned_ = false;  // 有符号
        }
        update_gate_output_ = {bits, true};   // UINT
        reset_gate_output_ = {bits, true};    // UINT
        return *this;
    }

    OperatorQuantConfig& setUsePOT2(bool usePOT2) {
        usePOT2_ = usePOT2;
        return *this;
    }

    OperatorQuantConfig& setPotScaleMethod(PotScaleMethod method) {
        pot_scale_method_ = method;
        return *this;
    }

    OperatorQuantConfig& setPotScaleTolerance(float tolerance) {
        pot_scale_tolerance_ = tolerance;
        return *this;
    }
};
