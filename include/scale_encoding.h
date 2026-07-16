#pragma once

// ============================================================================
// scale_encoding.h - 量化 scale 编码层
// ============================================================================
//
// 职责：把「校准得到的连续 scale」编码为设备可执行的量化参数表示，包含：
//   1. scale/rescale 派生算子：连续 scale ↔ M·2^-shift 仿射编码、POT2 shift、
//      QuantParam → FixedPointScale、源/目标 QuantParam → rescale 执行表示。
//   2. 高层编码入口：ContinuousScaleResult → EncodedScaleResult（POT2 或仿射），
//      MinMax / SQNR / Percentile 校准前端统一经 encodeScaleResult 决定最终 scale。
//
// 分层说明：
//   - 最底层策略/取整（PotScaleMethod / scaleToPowerOfTwo）在 pot_scale_encode.h；
//   - 本层依赖量化参数数据结构（quantize_param_types.h）与基础舍入（quantize_ops_helper.h）；
//   - 校准（直方图/SQNR/Percentile/MinMax range）逻辑在 pot_sqnr_calibrator.h，不在此。
//
// ============================================================================

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "pot_scale_encode.h"       // PotScaleMethod / scaleToPowerOfTwo / checkedShiftToInt8
#include "quantize_param_types.h"   // QuantParam / ChannelQuantParam / FixedPointScale / *Rescale
#include "quantize_ops_helper.h"    // round_f / round_to_int（基础舍入）

// ============================================================================
// 编码结果数据结构
// ============================================================================

/**
 * 连续 scale 校准结果（SQNR / Percentile / MinMax 前端的输出）
 */
struct ContinuousScaleResult {
    float scale;      // 连续 scale
    float min;        // 量化范围最小值
    float max;        // 量化范围最大值
    float noise;      // 估计噪声（仅 SQNR 使用）
};

/**
 * POT scale 结果
 */
struct PotScaleResult {
    int8_t exp2_inv;    // POT 指数 (scale = 2^(-exp2_inv))
    float po2_scale;    // POT scale 值
    int32_t zero_point; // 零点
};

struct AffineScaleResult {
    FixedPointScale fixed_scale;
    float effective_scale;
    int32_t zero_point;
};

struct EncodedScaleResult {
    float continuous_scale;
    FixedPointScale fixed_scale;
    float effective_scale;
    int8_t pot_shift;
    int32_t zero_point;
};

/**
 * 按量化模式选择落盘/导出的 scale：
 *   - POT2: effective_scale (po2_scale = 2^-shift, M=1)
 *   - Affine: continuous_scale (编译器侧再 encodeMShift)
 */
inline float storedScaleForMode(const EncodedScaleResult& encoded, bool use_pot2) {
    return use_pot2 ? encoded.effective_scale : encoded.continuous_scale;
}

// ============================================================================
// 单一权威种子 -> 执行表示派生（host 端）
// ============================================================================
// 把唯一权威 QuantParam（scale + zero_point）按需派生为各 kernel 消费的执行
// 表示（FixedPointScale / Pot2Rescale / FloatRescale）。这些是纯量化参数转换，
// 不含任何校准（直方图/SQNR）逻辑。setRescaleParam / 权重量化 /
// 量化-反量化 boundary 使用。均为 host-only（含 throw、std::frexp 等）。

// 连续 scale -> 16bit 乘子 + 移位定点（scale ≈ multiplier * 2^-shift）。
inline FixedPointScale encodeMShift(float scale) {
    if (!(scale > 0.0f)) {
        throw std::runtime_error("Invalid scale <= 0 in encodeMShift");
    }
    int exp2 = 0;
    double mant = std::frexp(static_cast<double>(scale), &exp2);  // scale = mant * 2^exp2
    uint32_t m = static_cast<uint32_t>(std::llround(mant * 65536.0));
    if (m == 65536u) {
        m = 32768u;
        exp2 += 1;
    }
    if (m < 32768u) {
        m = 32768u;
    }
    const int shift = 16 - exp2;
    if (shift < std::numeric_limits<int8_t>::min() || shift > std::numeric_limits<int8_t>::max()) {
        throw std::runtime_error("encodeMShift shift out of int8 range");
    }
    return FixedPointScale{static_cast<uint16_t>(m), static_cast<int8_t>(shift)};
}

inline float decodeMShift(const FixedPointScale &s) {
    return static_cast<float>(s.multiplier) * std::ldexp(1.0f, -static_cast<int>(s.shift));
}

// POT2 整数移位：scale = 2^-shift  =>  shift = round(-log2(scale))。
// 仅当 scale 已是 2 的幂（POT2 模式权威）时精确；仿射连续 scale 请改用 encodeMShift。
inline int8_t pot2Shift(const QuantParam &q) {
    if (!(q.scale > 0.0f)) return 0;
    return static_cast<int8_t>(std::lround(-std::log2(static_cast<double>(q.scale))));
}

// 量化 boundary 用：把单个 QuantParam 派生为 FixedPointScale。
//   - POT2 : {1, shift}（shift 由 scale 无损还原）
//   - 仿射 : encodeMShift(连续 scale)
inline FixedPointScale toFixedScale(const QuantParam &q, bool use_pot2) {
    if (use_pot2) return FixedPointScale{1, pot2Shift(q)};
    return encodeMShift(q.scale);
}

inline std::vector<FixedPointScale> toFixedScales(const ChannelQuantParam &c, bool use_pot2) {
    std::vector<FixedPointScale> out(c.channels.size());
    for (size_t i = 0; i < c.channels.size(); ++i) out[i] = toFixedScale(c.channels[i], use_pot2);
    return out;
}

// makeRescale<R>: 由源/目标权威 QuantParam 推导 rescale 执行表示。
//   - POT2 : 整数移位差（与现状 bit 级一致，零 float）
//   - 仿射 : 原始连续 scale 比值 + encodeMShift（绝不经 fixed_scale 往返）
//   - FP   : src_scale / dst_scale
// 提供单值与"两源乘积"（GEMM：src_scale = a.scale * b.scale）两组。
template <class R>
R makeRescale(const QuantParam &src, const QuantParam &dst);

template <>
inline Pot2Rescale makeRescale<Pot2Rescale>(const QuantParam &src, const QuantParam &dst) {
    return Pot2Rescale{static_cast<int8_t>(pot2Shift(src) - pot2Shift(dst))};
}

template <>
inline FixedPointScale makeRescale<FixedPointScale>(const QuantParam &src, const QuantParam &dst) {
    return encodeMShift(src.scale / dst.scale);
}

template <>
inline FloatRescale makeRescale<FloatRescale>(const QuantParam &src, const QuantParam &dst) {
    return FloatRescale{src.scale / dst.scale};
}

// 两源乘积版（GEMM：源 scale = a.scale * b.scale）
template <class R>
R makeRescaleProduct(const QuantParam &a, const QuantParam &b, const QuantParam &dst);

template <>
inline Pot2Rescale makeRescaleProduct<Pot2Rescale>(const QuantParam &a, const QuantParam &b, const QuantParam &dst) {
    return Pot2Rescale{static_cast<int8_t>(pot2Shift(a) + pot2Shift(b) - pot2Shift(dst))};
}

template <>
inline FixedPointScale makeRescaleProduct<FixedPointScale>(const QuantParam &a, const QuantParam &b, const QuantParam &dst) {
    return encodeMShift((a.scale * b.scale) / dst.scale);
}

template <>
inline FloatRescale makeRescaleProduct<FloatRescale>(const QuantParam &a, const QuantParam &b, const QuantParam &dst) {
    return FloatRescale{(a.scale * b.scale) / dst.scale};
}

// 单源 -> "两源乘积"目标（GEMM bias：bias 从自身 scale 进入累加器 scale = a.scale * b.scale）
template <class R>
R makeRescaleToProduct(const QuantParam &src, const QuantParam &a, const QuantParam &b);

template <>
inline Pot2Rescale makeRescaleToProduct<Pot2Rescale>(const QuantParam &src, const QuantParam &a, const QuantParam &b) {
    return Pot2Rescale{static_cast<int8_t>(pot2Shift(src) - (pot2Shift(a) + pot2Shift(b)))};
}

template <>
inline FixedPointScale makeRescaleToProduct<FixedPointScale>(const QuantParam &src, const QuantParam &a, const QuantParam &b) {
    return encodeMShift(src.scale / (a.scale * b.scale));
}

template <>
inline FloatRescale makeRescaleToProduct<FloatRescale>(const QuantParam &src, const QuantParam &a, const QuantParam &b) {
    return FloatRescale{src.scale / (a.scale * b.scale)};
}

// ============================================================================
// 高层编码入口：连续 scale → POT2 / 仿射
// ============================================================================

/**
 * 兼容别名：等价于 scaleToPowerOfTwo(scale, Round)。
 */
inline std::pair<float, int8_t> roundScaleToPowerOfTwo(float scale) {
    return scaleToPowerOfTwo(scale, PotScaleMethod::Round);
}

/**
 * 计算 zero-point
 */
inline int32_t computeZeroPoint(float continuous_min, float po2_scale,
                                int64_t quant_min, bool is_symmetric) {
    if (is_symmetric) {
        return 0;
    } else {
        float zp_fp = static_cast<float>(quant_min) - continuous_min / po2_scale;
        return round_to_int(zp_fp);
    }
}

/**
 * 连续 scale → POT（公共实现，MinMax / 直方图共用）。
 *
 * @param continuous_scale 连续 scale
 * @param continuous_min   浮点范围 min（zp 与 CoverRange 用）
 * @param continuous_max   浮点范围 max（CoverRange 用）
 * @param bw               位宽
 * @param is_symmetric     是否对称
 * @param method           POT 取整策略（默认 CoverRange）
 * @param tolerance        CoverRange 相对容差
 */
inline PotScaleResult convertToPot(
    float continuous_scale,
    float continuous_min,
    float continuous_max,
    QuantBitWidth bw,
    bool is_symmetric,
    PotScaleMethod method = PotScaleMethod::CoverRange,
    float tolerance = 0.02f) {

    const int64_t quant_min = bw.qmin();

    float po2_scale;
    int8_t n;
    std::tie(po2_scale, n) = scaleToPowerOfTwo(
        continuous_scale,
        method,
        continuous_min,
        continuous_max,
        tolerance);

    int32_t zp;
    if (is_symmetric) {
        zp = 0;
    } else if (method == PotScaleMethod::Floor) {
        // 与历史 calibrateQuantParams(MinMax floor) 一致：min 对齐到 POT 网格后再算 zp
        const float aligned_min = std::floor(continuous_min / po2_scale) * po2_scale;
        zp = round_to_int(static_cast<float>(quant_min) - aligned_min / po2_scale);
    } else {
        // Round / CoverRange：与 AIMET 一致，保持 continuous_min，按新 scale 算 zp
        zp = computeZeroPoint(continuous_min, po2_scale, quant_min, is_symmetric);
    }

    return PotScaleResult{n, po2_scale, zp};
}

/** 兼容旧签名：无 max 时 CoverRange 退化为仅用 min 推 range（max=min 时 range=0 → Floor 分支）。 */
inline PotScaleResult convertToPot(
    float continuous_scale,
    float continuous_min,
    QuantBitWidth bw,
    bool is_symmetric,
    bool coverage_round) {
    const PotScaleMethod method =
        coverage_round ? PotScaleMethod::Floor : PotScaleMethod::Round;
    // 旧 Round 路径无 max；旧 Floor 不依赖 range 判定
    return convertToPot(continuous_scale, continuous_min, continuous_min, bw, is_symmetric, method);
}

inline AffineScaleResult convertToAffineScale(
    float continuous_scale,
    float continuous_min,
    QuantBitWidth bw,
    bool is_symmetric) {
    FixedPointScale fs = encodeMShift(continuous_scale);
    float effective = decodeMShift(fs);
    int32_t zp = computeZeroPoint(continuous_min, effective, bw.qmin(), is_symmetric);
    return AffineScaleResult{fs, effective, zp};
}

/**
 * 公共编码入口：连续校准结果 → POT2 或仿射 EncodedScaleResult。
 * MinMax / SQNR / Percentile 均应只调用此函数决定最终 scale。
 */
inline EncodedScaleResult encodeScaleResult(
    const ContinuousScaleResult& cont,
    QuantBitWidth bw,
    bool is_symmetric,
    bool use_pot2,
    PotScaleMethod method = PotScaleMethod::CoverRange,
    float tolerance = 0.02f) {
    if (use_pot2) {
        PotScaleResult pot = convertToPot(
            cont.scale, cont.min, cont.max, bw, is_symmetric, method, tolerance);
        return EncodedScaleResult{
            cont.scale,
            FixedPointScale{1u, pot.exp2_inv},
            pot.po2_scale,
            pot.exp2_inv,
            pot.zero_point
        };
    }

    AffineScaleResult affine = convertToAffineScale(cont.scale, cont.min, bw, is_symmetric);
    int8_t compat_shift = static_cast<int8_t>(round_f(-std::log2(affine.effective_scale)));
    return EncodedScaleResult{
        cont.scale,
        affine.fixed_scale,
        affine.effective_scale,
        compat_shift,
        affine.zero_point
    };
}

/** 兼容旧签名（无 max / 用 coverage_round bool）。 */
inline EncodedScaleResult encodeScaleResult(
    float continuous_scale,
    float continuous_min,
    QuantBitWidth bw,
    bool is_symmetric,
    bool use_pot2,
    bool coverage_round = false) {
    ContinuousScaleResult cont;
    cont.scale = continuous_scale;
    cont.min = continuous_min;
    cont.max = continuous_min;  // 无 max 时 CoverRange 的 near-2^k 判断无效 → 走 Floor
    cont.noise = 0.0f;
    const PotScaleMethod method =
        coverage_round ? PotScaleMethod::Floor : PotScaleMethod::Round;
    return encodeScaleResult(cont, bw, is_symmetric, use_pot2, method);
}

// ============================================================================
// 调试函数
// ============================================================================

/// @brief 打印 GRU 量化参数（调试用；依赖 pot2Shift，故随派生算子族置于本层）
inline void printParms(const GRUQuantParams &quant_parms) {
    printf("GRUQuantParams:\n");
    printf("  hidden = %d\n", quant_parms.hidden_);

    // 输入/隐状态
    printf("  x:  exp2_inv=%2d, zp=%d\n", pot2Shift(quant_parms.x_), quant_parms.x_.zero_point);
    printf("  h:  exp2_inv=%2d, zp=%d\n", pot2Shift(quant_parms.h_), quant_parms.h_.zero_point);

    // Per-channel 权重（唯一权威 per-channel 数组）
    auto print_vec = [](const char *name, const ChannelQuantParam &c) {
        printf("  %s (size %zu): ", name, c.channels.size());
        for (size_t i = 0; i < c.channels.size() && i < 5; ++i) printf("%d ", pot2Shift(c.channels[i]));
        if (c.channels.size() > 5) printf("...");
        printf("\n");
    };
    print_vec("W ", quant_parms.W_);
    print_vec("R ", quant_parms.R_);
    print_vec("bw", quant_parms.bw_);
    print_vec("br", quant_parms.br_);

    // Linear 输出
    printf("  weight_ih_linear: shift=%2d, zp=%d\n", pot2Shift(quant_parms.weight_ih_linear_), quant_parms.weight_ih_linear_.zero_point);
    printf("  weight_hh_linear: shift=%2d, zp=%d\n", pot2Shift(quant_parms.weight_hh_linear_), quant_parms.weight_hh_linear_.zero_point);

    // 门参数
    printf("  update_gate_input:  exp2_inv=%2d, zp=%d\n", pot2Shift(quant_parms.update_gate_input_), quant_parms.update_gate_input_.zero_point);
    printf("  update_gate_output: exp2_inv=%2d, zp=%d\n", pot2Shift(quant_parms.update_gate_output_), quant_parms.update_gate_output_.zero_point);
    printf("  reset_gate_input:   exp2_inv=%2d, zp=%d\n", pot2Shift(quant_parms.reset_gate_input_), quant_parms.reset_gate_input_.zero_point);
    printf("  reset_gate_output:  exp2_inv=%2d, zp=%d\n", pot2Shift(quant_parms.reset_gate_output_), quant_parms.reset_gate_output_.zero_point);
    printf("  new_gate_input:     exp2_inv=%2d, zp=%d\n", pot2Shift(quant_parms.new_gate_input_), quant_parms.new_gate_input_.zero_point);
    printf("  new_gate_output:    exp2_inv=%2d, zp=%d\n", pot2Shift(quant_parms.new_gate_output_), quant_parms.new_gate_output_.zero_point);

    // 中间计算
    printf("  mul_reset_hidden:   exp2_inv=%2d, zp=%d\n", pot2Shift(quant_parms.mul_reset_hidden_), quant_parms.mul_reset_hidden_.zero_point);

    // 隐状态更新
    printf("  mul_new_contribution: exp2_inv=%2d, zp=%d\n", pot2Shift(quant_parms.mul_new_contribution_),
           quant_parms.mul_new_contribution_.zero_point);
    printf("  mul_old_contribution: exp2_inv=%2d, zp=%d\n", pot2Shift(quant_parms.mul_old_contribution_),
           quant_parms.mul_old_contribution_.zero_point);
}
