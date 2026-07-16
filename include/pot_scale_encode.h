#pragma once

/**
 * pot_scale_encode.h - 连续 scale → POT2 的底层编码工具（策略 + 数值转换）
 *
 * 职责：POT2 scale 计算的最底层、无项目内头依赖的工具。校准前端
 * (MinMax / SQNR / Percentile) 产出连续 scale 后，是否 / 如何落到 2 的幂次
 * 一律经此处的 scaleToPowerOfTwo；高层 encodeScaleResult（含仿射编码 + zp）
 * 在 pot_sqnr_calibrator.h（因其依赖 quantize_ops_helper.h 的 encodeMShift）。
 *
 * 与 AIMET find_closest_power_of_2_scale 对齐：
 *   - Round:      n = round(-log2(s))
 *   - CoverRange: real_range≈2^k（相对容差内）则 round，否则 floor(n) 保证 new_scale ≥ s
 *   - Floor:      始终 floor(n)，覆盖优先
 *
 * 依赖：仅标准库；可被配置头 / 算子头 / 校准头安全包含，不引入循环依赖。
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

/**
 * 连续 scale → POT2 (2^-n) 的取整策略。
 * 作为 OperatorQuantConfig.pot_scale_method_ 的字段类型，故定义在此编码层。
 */
enum class PotScaleMethod : int8_t {
    Round = 0,       ///< 四舍五入到最近 2^-n（可能缩小 scale、裁剪范围）
    CoverRange = 1,  ///< AIMET 默认：real_range≈2^k 则 Round，否则 Floor 覆盖原范围
    Floor = 2,       ///< 始终 floor(-log2(s))，保证 po2_scale >= continuous_scale
};

/**
 * 将 POT shift 安全收窄到 int8。32bit 量化下 scale 极小、shift 可能很大，
 * 若静默截断成 int8 会得到错误的 scale，因此在标定阶段直接抛错暴露问题。
 * 仅供 host 端标定代码调用。
 */
inline int8_t checkedShiftToInt8(float n, const char* ctx) {
    if (n < static_cast<float>(std::numeric_limits<int8_t>::min()) ||
        n > static_cast<float>(std::numeric_limits<int8_t>::max())) {
        throw std::runtime_error(std::string("POT shift out of int8 range in ") + ctx);
    }
    return static_cast<int8_t>(n);
}

/**
 * 判断 value 是否相对接近某个 2^k（|v-2^k|/2^k < tolerance）。
 * 用于 CoverRange：判断校准 real_range 是否“已经是”2 的幂。
 * 与 AIMET is_power_of_2(relative_tolerance) 一致。
 */
inline bool isNearPowerOfTwo(float value, float relative_tolerance = 0.02f) {
    if (!(value > 0.0f)) {
        return false;
    }
    const float log2_val = std::log2(value);
    const float nearest = std::pow(2.0f, std::round(log2_val));
    return std::abs(value - nearest) / nearest < relative_tolerance;
}

/**
 * 将连续 scale 编码为 POT2：po2_scale = 2^(-n)。
 *
 * @param scale     连续 scale（必须 > 0）
 * @param method    Round / CoverRange / Floor
 * @param rmin/rmax CoverRange 需要的浮点范围；其它 method 可忽略
 * @param tolerance CoverRange 下 real_range 接近 2^k 的相对容差（默认 2%）
 * @return {po2_scale, n}
 */
inline std::pair<float, int8_t> scaleToPowerOfTwo(
    float scale,
    PotScaleMethod method = PotScaleMethod::CoverRange,
    float rmin = 0.0f,
    float rmax = 0.0f,
    float tolerance = 0.02f) {
    if (!(scale > 0.0f)) {
        throw std::runtime_error("Invalid scale <= 0 in scaleToPowerOfTwo");
    }

    const float n = -std::log2(scale);
    float n_chosen;
    switch (method) {
        case PotScaleMethod::Round:
            n_chosen = std::round(n);
            break;
        case PotScaleMethod::Floor:
            n_chosen = std::floor(n);
            break;
        case PotScaleMethod::CoverRange:
        default: {
            const float real_range = std::abs(rmax - rmin);
            // real_range≈2^k 则 round；否则 floor(n) ⇒ 2^{-n} ≥ scale，覆盖原范围
            n_chosen = isNearPowerOfTwo(real_range, tolerance) ? std::round(n) : std::floor(n);
            break;
        }
    }

    const int8_t n_out = checkedShiftToInt8(n_chosen, "scaleToPowerOfTwo");
    const float po2_scale = std::pow(2.0f, -static_cast<float>(n_out));
    return {po2_scale, n_out};
}
