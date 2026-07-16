#pragma once

/**
 * POT 量化校准模块
 * 
 * 四个独立模块（均为独立函数，无状态类）：
 * 1. 直方图收集 - 在 histogram_collector.h
 * 2. Percentile 校准 - calibratePercentile()
 * 3. SQNR 校准 - calibrateSqnr()
 * 4. POT 转换 - encodeScaleResult() / convertToPot()（策略由 PotScaleMethod 决定）
 * 
 * 调用流程：
 *   MinMax/Histogram → ContinuousScaleResult → encodeScaleResult(method) → EncodedScaleResult
 * 
 * MinMax 与直方图共用同一编码入口；默认 PotScaleMethod::CoverRange（与 AIMET 一致）。
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "histogram_collector.h"
#include "quantize_param_types.h"
#include "quantize_bitwidth_config.h"  // for QuantBitWidth
#include "scale_encoding.h"            // ContinuousScaleResult / encodeScaleResult / convertToPot
#include "quantize_ops_helper.h"       // for round_f, round_to_int

// ============================================================================
// 模块 2: Percentile 校准
// ============================================================================

/**
 * Percentile 校准配置
 */
struct PercentileConfig {
    float percentile = 99.99f;  // 百分位数 (如 99.99 表示保留 99.99% 数据)
};

/**
 * Percentile 校准：基于百分位数裁剪计算连续 scale
 * 
 * 与 AIMET PercentileEncodingAnalyzer.compute_encodings_from_stats + adjust_min_max 完全一致
 * 
 * @param hist 直方图数据
 * @param num_steps 量化级数 (quant_max - quant_min)
 * @param is_symmetric 是否对称量化
 * @param config 配置参数
 * @param is_unsigned 是否无符号量化（UINT）
 * @return ContinuousScaleResult
 */
inline ContinuousScaleResult calibratePercentile(
    const Histogram& hist,
    int64_t num_steps,
    bool is_symmetric,
    const PercentileConfig& config = PercentileConfig(),
    bool is_unsigned = false) {
    
    if (!hist.is_valid()) {
        throw std::runtime_error("Histogram is invalid in calibratePercentile");
    }
    
    // 与 AIMET 一致：检查 num_steps
    if (num_steps <= 0) {
        throw std::runtime_error("num_steps must be > 0 in calibratePercentile");
    }
    
    // 与 AIMET _get_minimum_scale 一致
    const float minimum_scale = get_minimum_scale(num_steps);
    
    // 获取百分位数范围
    float clip_ratio = (100.0f - config.percentile) / 100.0f;
    auto [pmin, pmax] = hist.getPercentileRange(clip_ratio);
    
    // 与 AIMET adjust_min_max 一致：确保范围包含 0
    pmin = std::min(pmin, 0.0f);
    pmax = std::max(pmax, 0.0f);
    
    // 与 AIMET adjust_min_max 一致：确保 finite（clamp 到合理范围）
    constexpr float float_max = std::numeric_limits<float>::max();
    constexpr float float_min = std::numeric_limits<float>::lowest();
    pmin = std::max(float_min, std::min(pmin, 0.0f));
    pmax = std::min(float_max, std::max(pmax, 0.0f));
    
    // 与 AIMET adjust_min_max 一致：确保范围不太小（使用 minimum_scale）
    float tensor_threshold = (pmax - pmin) / static_cast<float>(num_steps);
    if (tensor_threshold < minimum_scale) {
        if (is_symmetric && !is_unsigned) {
            // INT 对称量化：两边扩展
            int64_t num_neg_steps = (num_steps + 1) / 2;  // ceil
            int64_t num_pos_steps = num_steps / 2;        // floor
            pmin -= minimum_scale * static_cast<float>(num_neg_steps);
            pmax += minimum_scale * static_cast<float>(num_pos_steps);
        } else {
            // 非对称量化 或 UINT 对称量化：只扩展 max
            pmax += minimum_scale * static_cast<float>(num_steps);
        }
    }
    
    ContinuousScaleResult result;
    
    if (is_symmetric) {
        if (is_unsigned) {
            // UINT + symmetric: 数据范围 [0, max]，量化范围 [0, num_steps]
            float data_max = std::max(pmax, minimum_scale);
            result.scale = data_max / static_cast<float>(num_steps);
            result.scale = std::max(result.scale, minimum_scale);
            result.min = 0.0f;
            result.max = result.scale * static_cast<float>(num_steps);
        } else {
            // INT + symmetric: 与 AIMET adjust_min_max 对称量化处理完全一致
            int64_t num_pos_steps = num_steps / 2;        // floor
            int64_t num_neg_steps = (num_steps + 1) / 2;  // ceil
            
            // 边缘情况：num_steps=1 时 num_pos_steps=0，需要防止除零
            // AIMET 在这种情况下会得到 inf，但最终会被 minimum_scale 约束
            float delta_from_max = (num_pos_steps > 0) 
                ? pmax / static_cast<float>(num_pos_steps) 
                : std::numeric_limits<float>::max();
            float delta_from_min = (num_neg_steps > 0) 
                ? -pmin / static_cast<float>(num_neg_steps) 
                : std::numeric_limits<float>::max();
            
            result.scale = std::max(delta_from_max, delta_from_min);
            result.min = -static_cast<float>(num_neg_steps) * result.scale;
            result.max = static_cast<float>(num_pos_steps) * result.scale;
        }
    } else {
        result.scale = (pmax - pmin) / static_cast<float>(num_steps);
        result.min = pmin;
        result.max = pmax;
    }
    
    // 最终保护：确保 scale 不小于 minimum_scale（与 AIMET 一致）
    result.scale = std::max(result.scale, minimum_scale);
    result.noise = 0.0f;  // Percentile 不计算噪声
    
    return result;
}

// ============================================================================
// 模块 3: SQNR 校准
// ============================================================================

/**
 * SQNR 校准配置
 */
struct SqnrConfig {
    int symmetric_delta_candidates = 101;   // AIMET 默认 101
    int asymmetric_delta_candidates = 17;   // AIMET 默认 17
    int offset_candidates = 21;             // AIMET 默认 21
    float gamma = 3.0f;                     // AIMET 默认 3.0
    float p = 2.0f;                         // Lp 范数 (p=2 = MSE)
};

namespace sqnr_detail {

/**
 * 估算量化噪声（与 AIMET _estimate_clip_and_quant_noise 完全一致）
 */
inline float estimateNoise(const Histogram& hist, float delta, float offset,
                           int64_t num_steps, float gamma, float p) {
    if (delta <= 0) return std::numeric_limits<float>::max();
    
    float bin_width = hist.bin_width();
    float total_noise = 0.0f;
    
    for (int i = 0; i < hist.num_bins; ++i) {
        float count = hist.counts[i];
        if (count < 1e-6f) continue;
        
        float x = hist.min_val + (i + 0.5f) * bin_width;
        
        // AIMET 公式：q = round(x / delta - offset)
        float q = round_f(x / delta - offset);
        
        bool clipped = (q < 0) || (q > static_cast<float>(num_steps));
        q = std::max(0.0f, std::min(static_cast<float>(num_steps), q));
        float x_recon = (q + offset) * delta;
        
        float error = std::pow(std::abs(x_recon - x), p);
        if (clipped && gamma != 1.0f) error *= gamma;
        
        total_noise += error * count;
    }
    return total_noise;
}

/**
 * 对称量化搜索（与 AIMET _pick_test_candidates_symmetric 完全一致）
 * 
 * @param is_unsigned 是否 UINT 类型
 */
inline ContinuousScaleResult searchSymmetric(
    const Histogram& hist,
    float min_val, float max_val,
    int64_t num_steps,
    const SqnrConfig& config,
    bool is_unsigned = false) {
    
    const float minimum_scale = get_minimum_scale(num_steps);
    
    ContinuousScaleResult best{0, 0, 0, std::numeric_limits<float>::max()};
    
    float max_delta;
    float offset;
    
    if (is_unsigned) {
        // UINT + symmetric: 数据范围 [0, max]，量化范围 [0, num_steps]
        // offset = 0 (zp = 0)
        max_delta = max_val / static_cast<float>(num_steps);
        offset = 0.0f;
    } else {
        // INT + symmetric: 对称量化，offset = (-num_steps) // 2
        max_delta = 2.0f * std::max(max_val, -min_val) / static_cast<float>(num_steps);
        offset = -static_cast<float>((num_steps + 1) / 2);
    }
    
    // 边缘保护：确保 max_delta 不为 0（避免 SQNR 搜索退化）
    max_delta = std::max(max_delta, minimum_scale);
    
    // 边缘保护：确保除数不为 0
    const int divisor = std::max(config.symmetric_delta_candidates - 1, 1);
    
    for (int d = 1; d <= config.symmetric_delta_candidates; ++d) {
        float delta = max_delta * d / divisor;
        delta = std::max(delta, minimum_scale);
        
        float noise = estimateNoise(hist, delta, offset, num_steps, config.gamma, config.p);
        
        if (noise < best.noise) {
            best.noise = noise;
            best.scale = delta;
            best.min = offset * delta;
            best.max = best.min + num_steps * delta;
        }
    }
    
    if (best.noise == std::numeric_limits<float>::max()) {
        throw std::runtime_error("calibrateSqnr: searchSymmetric failed to find valid scale");
    }
    return best;
}

/**
 * 非对称量化搜索（与 AIMET _pick_test_candidates_asymmetric 完全一致）
 */
inline ContinuousScaleResult searchAsymmetric(
    const Histogram& hist,
    float min_val, float max_val,
    int64_t num_steps,
    const SqnrConfig& config) {
    
    float max_delta = (max_val - min_val) / static_cast<float>(num_steps);
    
    // 计算 observed_min/observed_max（量化对齐的范围）
    float observed_offset = round_f(min_val / max_delta);
    float observed_min = max_delta * observed_offset;
    float observed_max = observed_min + max_delta * static_cast<float>(num_steps);
    
    const float minimum_scale = get_minimum_scale(num_steps);
    
    // 生成 offset 候选值
    const int num_offsets = std::min(static_cast<int>(num_steps + 2), config.offset_candidates);
    std::vector<float> offsets(num_offsets);
    float offset_step = static_cast<float>(num_steps) / (num_offsets - 2);
    for (int o = 0; o < num_offsets - 1; ++o) {
        offsets[o] = round_f(-static_cast<float>(num_steps) + o * offset_step);
    }
    offsets[num_offsets - 1] = observed_offset;
    
    ContinuousScaleResult best{0, 0, 0, std::numeric_limits<float>::max()};
    
    for (int d = 1; d <= config.asymmetric_delta_candidates; ++d) {
        float delta = max_delta * d / (config.asymmetric_delta_candidates - 1);
        delta = std::max(delta, minimum_scale);
        
        for (int o = 0; o < num_offsets; ++o) {
            float off = offsets[o];
            
            // 与 AIMET _clamp_delta_offset_values 完全一致
            float test_min = delta * off;
            float test_max = test_min + delta * static_cast<float>(num_steps);
            test_min = std::max(observed_min, test_min);
            test_max = std::min(observed_max, test_max);
            float clamped_delta = (test_max - test_min) / static_cast<float>(num_steps);
            clamped_delta = std::max(clamped_delta, minimum_scale);
            float clamped_offset = round_f(test_min / clamped_delta);
            
            float noise = estimateNoise(hist, clamped_delta, clamped_offset, num_steps,
                                       config.gamma, config.p);
            
            if (noise < best.noise) {
                best.noise = noise;
                best.scale = clamped_delta;
                best.min = clamped_offset * clamped_delta;
                best.max = best.min + num_steps * clamped_delta;
            }
        }
    }
    
    if (best.noise == std::numeric_limits<float>::max()) {
        throw std::runtime_error("calibrateSqnr: searchAsymmetric failed to find valid scale");
    }
    return best;
}

}  // namespace sqnr_detail

/**
 * SQNR 校准：基于 AIMET SqnrEncodingAnalyzer 的 SQNR 优化搜索
 * 
 * @param hist 直方图数据
 * @param num_steps 量化级数 (quant_max - quant_min)
 * @param is_symmetric 是否对称量化
 * @param config 配置参数
 * @param is_unsigned 是否无符号量化（UINT）
 * @return ContinuousScaleResult
 */
inline ContinuousScaleResult calibrateSqnr(
    const Histogram& hist,
    int64_t num_steps,
    bool is_symmetric,
    const SqnrConfig& config = SqnrConfig(),
    bool is_unsigned = false) {
    
    if (!hist.is_valid()) {
        throw std::runtime_error("Histogram is invalid in calibrateSqnr");
    }
    
    // 与 AIMET _pick_test_candidates 完全一致的范围预处理
    const float minimum_scale = get_minimum_scale(num_steps);
    
    // 确保范围包含 0
    float min_val = std::min(hist.min_val, 0.0f);
    float max_val = std::max(hist.max_val, 0.0f);
    
    // 确保范围有效（与 AIMET 一致）
    float min_range_limit = min_val + minimum_scale * static_cast<float>(num_steps);
    max_val = (max_val > min_range_limit) ? max_val : min_range_limit;
    
    if (is_symmetric) {
        return sqnr_detail::searchSymmetric(hist, min_val, max_val, num_steps, config, is_unsigned);
    } else {
        return sqnr_detail::searchAsymmetric(hist, min_val, max_val, num_steps, config);
    }
}

inline ContinuousScaleResult calibrateContinuousScaleFromHistogram(
    const Histogram &hist,
    QuantBitWidth bw,
    bool is_symmetric,
    bool use_percentile = false,
    float percentile = 99.99f) {
    if (!hist.is_valid()) {
        throw std::runtime_error("Histogram is invalid in calibrateContinuousScaleFromHistogram");
    }
    const int64_t num_steps = static_cast<int64_t>(bw.qmax_auto_scale()) - static_cast<int64_t>(bw.qmin_auto_scale());
    const bool is_unsigned = bw.is_unsigned_;
    if (use_percentile) {
        PercentileConfig config;
        config.percentile = percentile;
        return calibratePercentile(hist, num_steps, is_symmetric, config, is_unsigned);
    }
    SqnrConfig config;
    return calibrateSqnr(hist, num_steps, is_symmetric, config, is_unsigned);
}

inline ContinuousScaleResult calibrateContinuousScaleFromRange(
    float min_val,
    float max_val,
    QuantBitWidth bw,
    bool is_symmetric) {
    const int32_t quant_min = bw.qmin_auto_scale();
    const int32_t quant_max = bw.qmax_auto_scale();
    const int64_t num_steps = static_cast<int64_t>(quant_max) - static_cast<int64_t>(quant_min);
    if (num_steps <= 0) {
        throw std::runtime_error("num_steps must be > 0 in calibrateContinuousScaleFromRange");
    }
    const float minimum_scale = get_minimum_scale(num_steps);

    float min_with_zero = std::min(min_val, 0.0f);
    float max_with_zero = std::max(max_val, 0.0f);
    const float tensor_diff = (max_with_zero - min_with_zero) / static_cast<float>(num_steps);
    const float adjustment_step = (tensor_diff < minimum_scale) ? minimum_scale : 0.0f;

    float updated_min = min_with_zero;
    float updated_max = max_with_zero;
    if (is_symmetric) {
        if (bw.is_unsigned_) {
            updated_max = max_with_zero + static_cast<float>(num_steps) * adjustment_step;
            updated_min = 0.0f;
        } else {
            updated_max = max_with_zero + std::floor(static_cast<float>(num_steps) / 2.0f) * adjustment_step;
            updated_min = min_with_zero - std::ceil(static_cast<float>(num_steps) / 2.0f) * adjustment_step;
        }
    } else {
        updated_max = max_with_zero + static_cast<float>(num_steps) * adjustment_step;
    }

    float continuous_scale = minimum_scale;
    float aligned_min = updated_min;
    float aligned_max = updated_max;
    if (is_symmetric) {
        if (bw.is_unsigned_) {
            const float data_max = std::max(updated_max, minimum_scale);
            continuous_scale = std::max(data_max / static_cast<float>(quant_max), minimum_scale);
            aligned_min = 0.0f;
            aligned_max = continuous_scale * static_cast<float>(quant_max);
        } else {
            const int64_t num_pos_steps = num_steps / 2;
            const int64_t num_neg_steps = (num_steps + 1) / 2;
            const int additional_step = (num_steps == 3) ? 1 : 0;
            const float delta_from_max = (num_pos_steps + additional_step > 0)
                                             ? updated_max / static_cast<float>(num_pos_steps + additional_step)
                                             : 0.0f;
            const float delta_from_min =
                (num_neg_steps > 0) ? -updated_min / static_cast<float>(num_neg_steps) : 0.0f;
            continuous_scale = std::max(std::max(delta_from_max, delta_from_min), minimum_scale);
            aligned_min = -static_cast<float>(num_neg_steps) * continuous_scale;
            aligned_max = static_cast<float>(num_pos_steps) * continuous_scale;
        }
    } else {
        const float range = std::max(updated_max - updated_min, minimum_scale * static_cast<float>(num_steps));
        continuous_scale = std::max(range / static_cast<float>(num_steps), minimum_scale);
        aligned_min = updated_min;
        aligned_max = updated_min + continuous_scale * static_cast<float>(num_steps);
    }

    ContinuousScaleResult out;
    out.scale = continuous_scale;
    out.min = aligned_min;
    out.max = aligned_max;
    out.noise = 0.0f;
    return out;
}

// ============================================================================
// 统一入口函数
// ============================================================================

/**
 * 从直方图计算 POT 量化参数
 * 
 * 调用流程: Histogram → ContinuousScaleResult → encodeScaleResult(CoverRange) → POT
 */
inline void calibrateQuantParamsFromHistogram(
    const Histogram& hist,
    QuantBitWidth bw,
    bool is_symmetric,
    int8_t& exp2_inv,
    int32_t& zp,
    const char* name = nullptr,
    bool use_percentile = false,
    float percentile = 99.99f,
    PotScaleMethod method = PotScaleMethod::CoverRange,
    float tolerance = 0.02f) {
    
    if (!hist.is_valid()) {
        throw std::runtime_error("Histogram is invalid in calibrateQuantParamsFromHistogram");
    }
    
    ContinuousScaleResult continuous_result = calibrateContinuousScaleFromHistogram(
        hist, bw, is_symmetric, use_percentile, percentile);
    
    EncodedScaleResult encoded = encodeScaleResult(
        continuous_result, bw, is_symmetric, /*use_pot2=*/true, method, tolerance);
    
    exp2_inv = encoded.pot_shift;
    zp = encoded.zero_point;
    
#ifdef DEBUG
    if (name && name[0]) {
        const char* scheme_name = use_percentile ? "PERC" : "SQNR";
        const bool is_unsigned = bw.is_unsigned_;
        printf("[%s][%s] unsigned=%d range=[%.4f,%.4f] cont_scale=%.6f po2=%.6f(1/2^%d) zp=%d\n",
               scheme_name, name, is_unsigned, hist.min_val, hist.max_val, 
               continuous_result.scale, encoded.effective_scale, encoded.pot_shift, encoded.zero_point);
    }
#endif
}

