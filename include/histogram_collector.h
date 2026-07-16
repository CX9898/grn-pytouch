// ============================================================================
// histogram_collector.hpp - AIMET 风格的直方图收集器
// ============================================================================
//
// 实现类似 AIMET _HistogramObserver 的功能：
//   1. 收集数据到直方图
//   2. 支持多批次合并（带范围扩展）
//   3. 支持 SQNR 优化计算量化参数
//
// ============================================================================

#pragma once

#include <algorithm>
#include <array>
#include <cinttypes>  // for PRId64
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pot_scale_encode.h"  // checkedShiftToInt8 / isNearPowerOfTwo / scaleToPowerOfTwo

/**
 * 与 AIMET _get_minimum_scale 完全一致的实现
 * 返回给定量化步数的最小 scale
 * 
 * 定义: 最大的 s <= float32.eps 使得 -0.005 <= s * min_int < s * max_int <= 0.005
 */
inline float get_minimum_scale(int64_t num_steps) {
    constexpr float fp32_eps = 1.19209290e-07f;  // float32 epsilon
    constexpr float min_range_to_represent = 0.01f;  // (-0.005, 0.005) 范围
    return std::min(fp32_eps, min_range_to_represent / static_cast<float>(num_steps));
}

/**
 * 单个直方图结构体
 * 存储一个张量的直方图数据
 */
struct Histogram {
    std::vector<float> counts;  // 每个 bin 的计数
    float min_val;              // 直方图覆盖的最小值
    float max_val;              // 直方图覆盖的最大值
    int num_bins;               // bin 数量
    int64_t total_count;        // 总采样数

    Histogram() : min_val(0), max_val(0), num_bins(0), total_count(0) {}

    Histogram(int bins)
        : counts(bins, 0.0f),
          min_val(std::numeric_limits<float>::max()),
          max_val(std::numeric_limits<float>::lowest()),
          num_bins(bins),
          total_count(0) {}

    bool is_valid() const { return num_bins > 0 && total_count > 0; }

    float bin_width() const {
        if (num_bins <= 0 || max_val <= min_val) return 1.0f;
        return (max_val - min_val) / num_bins;
    }

    void reset(int bins = 0) {
        if (bins > 0) {
            num_bins = bins;
            counts.assign(bins, 0.0f);
        } else if (num_bins > 0) {
            counts.assign(num_bins, 0.0f);
        }
        min_val = std::numeric_limits<float>::max();
        max_val = std::numeric_limits<float>::lowest();
        total_count = 0;
    }

    void print(const char* name = nullptr) const {
        if (name) printf("[Histogram %s] ", name);
        printf("bins=%d, range=[%.6f, %.6f], total=%" PRId64 "\n", num_bins, min_val, max_val, total_count);
    }

    /**
     * 获取百分位数范围（与 AIMET PercentileEncodingAnalyzer 完全一致）
     * @param clip_percentile 裁剪百分位数 (例如 0.0001 表示裁剪两端各 0.01%，保留 99.98%)
     *                        当 clip_percentile=0 时，返回完整范围（percentile=100）
     * @return (min, max) 对应的范围
     * 
     * AIMET 实现参考：
     *   cum_sum = torch.cumsum(histogram, dim=0)
     *   max_index = torch.searchsorted(cum_sum, cum_sum[-1] * percentile / 100)
     *   min_index = torch.searchsorted(cum_sum, cum_sum[-1] * (1 - percentile / 100))
     */
    std::pair<float, float> getPercentileRange(float clip_percentile = 0.0001f) const {
        if (!is_valid() || total_count == 0) {
            return {min_val, max_val};
        }

        // 与 AIMET percentile=100 特殊处理一致
        if (clip_percentile <= 0.0f) {
            return {min_val, max_val};
        }

        // 计算累积和（与 AIMET torch.cumsum 一致）
        std::vector<float> cum_sum(num_bins);
        cum_sum[0] = counts[0];
        for (int i = 1; i < num_bins; ++i) {
            cum_sum[i] = cum_sum[i - 1] + counts[i];
        }

        float total = cum_sum[num_bins - 1];
        if (total < 1e-6f) {
            return {min_val, max_val};
        }

        float bw = bin_width();
        
        // 计算保留的百分比 (与 AIMET percentile 语义一致)
        // clip_percentile = 0.0001 表示裁剪 0.01%，即保留 99.99%
        float keep_percentile = 1.0f - clip_percentile;
        
        // 找 min_index: searchsorted(cum_sum, total * clip_percentile)
        // 即累积和首次 >= total * clip_percentile 的位置
        float min_threshold = total * clip_percentile;
        int min_index = 0;
        for (int i = 0; i < num_bins; ++i) {
            if (cum_sum[i] >= min_threshold) {
                min_index = i;
                break;
            }
        }
        
        // 找 max_index: searchsorted(cum_sum, total * keep_percentile)
        // 即累积和首次 >= total * keep_percentile 的位置
        float max_threshold = total * keep_percentile;
        int max_index = num_bins - 1;
        for (int i = 0; i < num_bins; ++i) {
            if (cum_sum[i] >= max_threshold) {
                max_index = i;
                break;
            }
        }
        
        // 使用 bin 左边界（与 AIMET bin_edges[index] 一致）
        float pmin = min_val + min_index * bw;
        float pmax = min_val + max_index * bw;
        
        // 确保范围有效
        if (pmax <= pmin) {
            pmax = max_val;
        }

        return {pmin, pmax};
    }
};

/**
 * 直方图收集器
 * 类似 AIMET 的 _HistogramObserver
 */
class HistogramCollector {
   public:
    struct Config {
        int num_bins = 2048;      // 直方图 bin 数量（AIMET 默认 2048）
        float ema_decay = 0.0f;   // EMA 衰减系数，0 表示不使用 EMA
        float growth_limit = std::numeric_limits<float>::infinity();  // 范围扩展限制（AIMET 默认无穷大）
                                   // 范围最多扩展到首次范围的 growth_limit 倍
    };

   private:
    Config config_;
    Histogram hist_;
    
    // 范围限制（首次收集后设置，与 AIMET _histogram_range_limit 一致）
    bool range_limit_set_ = false;
    float range_limit_min_ = 0.0f;
    float range_limit_max_ = 0.0f;

   public:
    HistogramCollector() : config_(), hist_() {}
    explicit HistogramCollector(const Config& config) : config_(config), hist_(config.num_bins) {}
    explicit HistogramCollector(int num_bins) : config_(), hist_(num_bins) { config_.num_bins = num_bins; }

    /**
     * 收集数据到直方图
     * 与 AIMET _HistogramObserver.merge_stats 完全一致：
     * - 首次收集：初始化直方图，设置范围限制
     * - 后续收集：在范围限制内扩展，超出部分裁剪到边界 bin
     */
    void collect(const float* data, size_t size) {
        if (size == 0) return;

        // 计算数据范围（与 AIMET _get_min_max 一致：过滤 inf/NaN 值）
        float data_min = std::numeric_limits<float>::max();
        float data_max = std::numeric_limits<float>::lowest();
        for (size_t i = 0; i < size; ++i) {
            float val = data[i];
            if (std::isfinite(val)) {  // 过滤 inf 和 NaN
                data_min = std::min(data_min, val);
                data_max = std::max(data_max, val);
            }
        }
        
        // 如果所有值都是 inf/NaN，使用默认范围
        if (data_min > data_max) {
            data_min = 0.0f;
            data_max = 0.0f;
        }

        // 与 AIMET 完全一致的处理流程：
        // 1. 先计算原始 min/max（用于存储和 range_limit 计算）
        // 2. 如果 min == max，在 _add_to_histogram 时使用扩展的 bin_edges（与 _create_bin_edges 一致）
        // 3. 首次收集时，计算 range_limit 时使用 minimum_range 机制
        
        float original_min = data_min;
        float original_max = data_max;
        
        // 与 AIMET minimum_scale 逻辑完全一致
        float minimum_scale = get_minimum_scale(config_.num_bins);
        float minimum_range = minimum_scale * config_.num_bins;
        float input_range = original_max - original_min;
        
        // 用于 range_limit 计算的 min/max（与 AIMET merge_stats 首次收集一致）
        float range_limit_min = original_min;
        float range_limit_max = original_max;
        
        // 如果范围太小，使用最小范围并确保 0 在范围内
        // 与 AIMET merge_stats 中的 zero_range_mask 处理一致
        if (input_range < minimum_range || std::isnan(input_range) || std::isinf(input_range)) {
            // 确保 0 在范围内
            range_limit_min = std::min(original_min, 0.0f);
            range_limit_max = std::max(original_max, 0.0f);
            input_range = minimum_range;
            // 注意：AIMET 中 input0_max 保持为 input0_max_safe，不更新为 input0_min_safe + minimum_range
            // 但 range_limit 的计算使用 input_range = minimum_range
        }

        if (!hist_.is_valid()) {
            // 首次收集：初始化直方图
            // 与 AIMET 一致：存储原始的 min/max（不扩展）
            hist_.reset(config_.num_bins);
            hist_.min_val = original_min;
            hist_.max_val = original_max;
            
            // 与 AIMET collect_stats 一致：如果 min == max，在计算 histogram 时使用扩展的 bin_edges
            // 但存储的 min_val/max_val 仍然是原始的（不扩展）
            if (original_min == original_max) {
                // 使用扩展的范围计算 histogram（与 _create_bin_edges 一致）
                float expanded_min = original_min - 0.5f;
                float expanded_max = original_max + 0.5f;
                _add_to_histogram_with_range(data, size, expanded_min, expanded_max);
            } else {
                _add_to_histogram(data, size);
            }
            
            // 设置范围限制（与 AIMET growth_limit 完全一致）
            // 使用 range_limit_min/range_limit_max 和 input_range（可能已更新为 minimum_range）
            range_limit_min_ = range_limit_min - input_range * config_.growth_limit / 2.0f;
            range_limit_max_ = range_limit_max + input_range * config_.growth_limit / 2.0f;
            range_limit_set_ = true;
        } else {
            // 后续收集：应用范围限制（与 AIMET clamp 逻辑完全一致）
            float updated_min = hist_.min_val;
            float updated_max = hist_.max_val;
            
            if (range_limit_set_) {
                // new_stats.min.clamp(left_limit, curr_stats.min)
                updated_min = std::max(range_limit_min_, std::min(data_min, hist_.min_val));
                // new_stats.max.clamp(curr_stats.max, right_limit)
                updated_max = std::min(range_limit_max_, std::max(data_max, hist_.max_val));
            } else {
                updated_min = std::min(data_min, hist_.min_val);
                updated_max = std::max(data_max, hist_.max_val);
            }
            
            if (updated_min == hist_.min_val && updated_max == hist_.max_val) {
                // 范围不变，直接添加（超出范围的值会被裁剪到边界 bin）
                _add_to_histogram(data, size);
            } else {
                // 需要扩展范围
                _merge_with_extended_range(data, size, updated_min, updated_max);
            }
        }
    }

    /**
     * 合并另一个直方图（与 AIMET merge_stats 行为一致）
     * - 首次合并：设置范围限制
     * - 后续合并：应用范围限制，支持范围扩展
     */
    void merge(const Histogram& other) {
        if (!other.is_valid()) return;

        if (!hist_.is_valid()) {
            // 首次合并：复制直方图并设置范围限制
            hist_ = other;
            
            // 设置范围限制（与 collect 一致）
            float input_range = other.max_val - other.min_val;
            
            // 处理零范围情况
            float minimum_scale = get_minimum_scale(config_.num_bins);
            float minimum_range = minimum_scale * config_.num_bins;
            if (input_range < minimum_range) {
                input_range = minimum_range;
            }
            
            range_limit_min_ = other.min_val - input_range * config_.growth_limit / 2.0f;
            range_limit_max_ = other.max_val + input_range * config_.growth_limit / 2.0f;
            range_limit_set_ = true;
            return;
        }

        // 后续合并：应用范围限制（与 collect 一致）
        float updated_min = hist_.min_val;
        float updated_max = hist_.max_val;
        
        if (range_limit_set_) {
            // 应用范围限制（与 AIMET clamp 逻辑一致）
            updated_min = std::max(range_limit_min_, std::min(other.min_val, hist_.min_val));
            updated_max = std::min(range_limit_max_, std::max(other.max_val, hist_.max_val));
        } else {
            updated_min = std::min(other.min_val, hist_.min_val);
            updated_max = std::max(other.max_val, hist_.max_val);
        }

        if (updated_min == hist_.min_val && updated_max == hist_.max_val) {
            // 范围相同，直接合并
            _merge_same_range(other);
        } else {
            // 范围不同，需要重新分配
            _merge_different_range(other, updated_min, updated_max);
        }
    }

    /**
     * 获取当前直方图
     */
    const Histogram& histogram() const { return hist_; }
    Histogram& histogram() { return hist_; }

    /**
     * 重置直方图和范围限制
     */
    void reset() { 
        hist_.reset(config_.num_bins);
        range_limit_set_ = false;
        range_limit_min_ = 0.0f;
        range_limit_max_ = 0.0f;
    }

    /**
     * 是否有有效数据
     */
    bool is_valid() const { return hist_.is_valid(); }

   private:
    /**
     * 将数据添加到当前直方图（与 AIMET 完全一致）
     * - 先构建直方图（只处理范围内的值）
     * - 再额外统计边界外的值并加到边界 bin
     * - inf/NaN 被忽略（与 AIMET torch.histc 行为一致）
     */
    void _add_to_histogram(const float* data, size_t size) {
        float bin_width = hist_.bin_width();
        float inv_bin_width = 1.0f / bin_width;
        
        // 与 AIMET torch.histc + 边界外统计完全一致
        for (size_t i = 0; i < size; ++i) {
            float val = data[i];
            
            // 跳过 inf/NaN（与 AIMET torch.histc 行为一致）
            if (!std::isfinite(val)) continue;
            
            // 计算 bin 索引
            int bin_idx = static_cast<int>((val - hist_.min_val) * inv_bin_width);
            
            // 与 AIMET 一致：边界外的值加到边界 bin
            if (val < hist_.min_val) {
                hist_.counts[0] += 1.0f;  // histogram[0] += sum(input < bin_edges[0])
            } else if (val > hist_.max_val) {
                hist_.counts[hist_.num_bins - 1] += 1.0f;  // histogram[-1] += sum(input > bin_edges[-1])
            } else {
                // 范围内的值：与 AIMET _get_bin_num 一致（只有上界 clamp）
                bin_idx = std::min(bin_idx, hist_.num_bins - 1);
                hist_.counts[bin_idx] += 1.0f;
            }
        }
        hist_.total_count += size;
    }

    /**
     * 使用指定的范围将数据添加到直方图（用于首次收集时 min == max 的情况）
     * 与 AIMET collect_stats 中的 _create_bin_edges 扩展逻辑一致
     */
    void _add_to_histogram_with_range(const float* data, size_t size, float expanded_min, float expanded_max) {
        float bin_width = (expanded_max - expanded_min) / config_.num_bins;
        float inv_bin_width = 1.0f / bin_width;
        
        // 与 AIMET torch.histc + 边界外统计完全一致
        for (size_t i = 0; i < size; ++i) {
            float val = data[i];
            
            // 跳过 inf/NaN（与 AIMET torch.histc 行为一致）
            if (!std::isfinite(val)) continue;
            
            // 计算 bin 索引（使用扩展的范围）
            int bin_idx = static_cast<int>((val - expanded_min) * inv_bin_width);
            
            // 与 AIMET 一致：边界外的值加到边界 bin
            if (val < expanded_min) {
                hist_.counts[0] += 1.0f;
            } else if (val > expanded_max) {
                hist_.counts[config_.num_bins - 1] += 1.0f;
            } else {
                // 范围内的值：与 AIMET _get_bin_num 一致（只有上界 clamp）
                bin_idx = std::min(bin_idx, config_.num_bins - 1);
                hist_.counts[bin_idx] += 1.0f;
            }
        }
        hist_.total_count += size;
    }

    /**
     * 扩展范围并合并新数据
     * 与 AIMET _HistogramObserver.merge_stats 完全一致：
     * - 精确按比例分割源 bin 到目标 bin
     * - 处理源 bin 跨越多个目标 bin 的情况
     */
    void _merge_with_extended_range(const float* data, size_t size, float new_min, float new_max) {
        // 创建新直方图
        std::vector<float> new_counts(config_.num_bins, 0.0f);
        float dest_bin_width = (new_max - new_min) / config_.num_bins;
        float src_bin_width = hist_.bin_width();

        // 检查源 bin 宽度有效性（与 AIMET minimum_scale 完全一致）
        float minimum_scale = get_minimum_scale(config_.num_bins);
        if (std::abs(src_bin_width) < minimum_scale || std::isnan(src_bin_width) || std::isinf(src_bin_width)) {
            // 旧直方图不可用（基于常数值），直接丢弃
            // 稍后用新数据重建
        } else {
            // 重新分配旧直方图的计数（与 AIMET 完全一致）
            for (int src_idx = 0; src_idx < hist_.num_bins; ++src_idx) {
                if (hist_.counts[src_idx] <= 0) continue;

                float count = hist_.counts[src_idx];
                
                // 源 bin 的起始位置
                float src_bin_start = hist_.min_val + src_idx * src_bin_width;
                
                // 计算落入的目标 bin 索引（与 AIMET _get_bin_num 完全一致：只有上界 clamp）
                int dest_bin_index = static_cast<int>((src_bin_start - new_min) / dest_bin_width);
                dest_bin_index = std::min(dest_bin_index, config_.num_bins - 1);
                
                // 目标 bin 的结束位置
                float dest_bin_end = new_min + dest_bin_width * (dest_bin_index + 1);
                
                // 计算分割比例（与 AIMET split_hist_value 完全一致：不对 ratio clamp）
                // 使用 rintf（银行家舍入）与 PyTorch round 一致
                float split_hist_value = rintf(
                    ((dest_bin_end - src_bin_start) / src_bin_width) * count
                );
                float first_bin_count = std::min(split_hist_value, count);
                
                // 添加到第一个目标 bin
                new_counts[dest_bin_index] += first_bin_count;
                
                // 如果有剩余部分，添加到下一个 bin（与 AIMET other_bin_updates 完全一致）
                float remaining_count = count - first_bin_count;
                if (remaining_count > 0) {
                    // 与 AIMET 一致：使用 _get_bin_num 计算 other_bin_index
                    int other_bin_index = static_cast<int>((src_bin_start + dest_bin_width - new_min) / dest_bin_width);
                    other_bin_index = std::min(other_bin_index, config_.num_bins - 1);
                    new_counts[other_bin_index] += remaining_count;
                }
            }
        }

        // 添加新数据（与 AIMET torch.histc + 边界外统计完全一致）
        float inv_dest_bin_width = 1.0f / dest_bin_width;
        for (size_t i = 0; i < size; ++i) {
            float val = data[i];
            
            // 跳过 inf/NaN（与 AIMET torch.histc 行为一致）
            if (!std::isfinite(val)) continue;
            
            // 计算 bin 索引
            int bin_idx = static_cast<int>((val - new_min) * inv_dest_bin_width);
            
            // 与 AIMET 一致：边界外的值加到边界 bin
            if (val < new_min) {
                new_counts[0] += 1.0f;
            } else if (val > new_max) {
                new_counts[config_.num_bins - 1] += 1.0f;
            } else {
                bin_idx = std::min(bin_idx, config_.num_bins - 1);
                new_counts[bin_idx] += 1.0f;
            }
        }

        // 更新直方图
        hist_.counts = std::move(new_counts);
        hist_.min_val = new_min;
        hist_.max_val = new_max;
        hist_.total_count += size;
    }

    /**
     * 合并范围相同的直方图
     */
    void _merge_same_range(const Histogram& other) {
        for (int i = 0; i < hist_.num_bins; ++i) {
            hist_.counts[i] += other.counts[i];
        }
        hist_.total_count += other.total_count;
    }

    /**
     * 合并范围不同的直方图
     */
    void _merge_different_range(const Histogram& other, float new_min, float new_max) {
        std::vector<float> new_counts(config_.num_bins, 0.0f);
        float new_bin_width = (new_max - new_min) / config_.num_bins;

        // 重新分配当前直方图
        _redistribute_histogram(hist_, new_counts, new_min, new_bin_width);

        // 重新分配另一个直方图
        _redistribute_histogram(other, new_counts, new_min, new_bin_width);

        // 更新
        hist_.counts = std::move(new_counts);
        hist_.min_val = new_min;
        hist_.max_val = new_max;
        hist_.total_count += other.total_count;
    }

    /**
     * 将直方图重新分配到新的 bin
     * 与 AIMET 完全一致
     */
    void _redistribute_histogram(const Histogram& src, std::vector<float>& dst, float new_min,
                                 float dest_bin_width) {
        float src_bin_width = src.bin_width();
        
        // 检查源 bin 宽度有效性（与 AIMET 完全一致）
        float minimum_scale = get_minimum_scale(config_.num_bins);
        if (std::abs(src_bin_width) < minimum_scale || std::isnan(src_bin_width) || std::isinf(src_bin_width)) {
            return;  // 源直方图不可用（基于常数值）
        }
        
        for (int src_idx = 0; src_idx < src.num_bins; ++src_idx) {
            if (src.counts[src_idx] <= 0) continue;

            float count = src.counts[src_idx];
            float src_bin_start = src.min_val + src_idx * src_bin_width;
            
            // 计算落入的目标 bin 索引（与 AIMET _get_bin_num 完全一致：只有上界 clamp）
            int dest_bin_index = static_cast<int>((src_bin_start - new_min) / dest_bin_width);
            dest_bin_index = std::min(dest_bin_index, config_.num_bins - 1);
            
            // 目标 bin 的结束位置
            float dest_bin_end = new_min + dest_bin_width * (dest_bin_index + 1);
            
            // 计算分割比例（与 AIMET split_hist_value 完全一致：不对 ratio clamp）
            // 使用 rintf（银行家舍入）与 PyTorch round 一致
            float split_hist_value = rintf(
                ((dest_bin_end - src_bin_start) / src_bin_width) * count
            );
            float first_bin_count = std::min(split_hist_value, count);
            
            // 添加到第一个目标 bin
            dst[dest_bin_index] += first_bin_count;
            
            // 剩余部分添加到下一个 bin（与 AIMET other_bin_updates 完全一致）
            float remaining_count = count - first_bin_count;
            if (remaining_count > 0) {
                // 与 AIMET 一致：使用 _get_bin_num 计算 other_bin_index
                int other_bin_index = static_cast<int>((src_bin_start + dest_bin_width - new_min) / dest_bin_width);
                other_bin_index = std::min(other_bin_index, config_.num_bins - 1);
                dst[other_bin_index] += remaining_count;
            }
        }
    }
};

/**
 * GRU 直方图收集器
 * 为 GRU 的每个中间张量维护一个直方图
 * 
 * 命名约定（与 optimized_quantizable_gru_2.md 文档对齐）：
 *   - update_gate_input/output: update gate 的输入/输出
 *   - reset_gate_input/output: reset gate 的输入/输出
 *   - new_gate_input/output: new gate 的输入/输出
 *   - mul_reset_hidden: r * h_n 的输出
 *   - mul_new_contribution: (1-u) * n 的输出
 *   - mul_old_contribution: u * h 的输出
 */
struct GRUHistogramCollectors {
    int hidden_ = 0;
    int num_bins_ = 2048;

    // 输入和隐藏状态
    HistogramCollector x_hist;
    HistogramCollector h_hist;

    // GEMM 结果
    HistogramCollector Wx_hist;
    HistogramCollector Rh_hist;

    // 门的预激活值（gate input）
    HistogramCollector update_gate_input_hist;
    HistogramCollector reset_gate_input_hist;
    HistogramCollector new_gate_input_hist;

    // 门的输出值（gate output）
    HistogramCollector update_gate_output_hist;
    HistogramCollector reset_gate_output_hist;
    HistogramCollector new_gate_output_hist;

    // 中间计算结果
    HistogramCollector mul_reset_hidden_hist;
    HistogramCollector mul_new_contribution_hist;
    HistogramCollector mul_old_contribution_hist;

    // 权重（per-channel，每个 channel 一个直方图）
    std::vector<HistogramCollector> W_hist;
    std::vector<HistogramCollector> R_hist;
    std::vector<HistogramCollector> bw_hist;
    std::vector<HistogramCollector> br_hist;
    
    // Per-Tensor 直方图（与 GPU 版本对应）
    HistogramCollector W_tensor_hist;
    HistogramCollector R_tensor_hist;
    HistogramCollector bw_tensor_hist;
    HistogramCollector br_tensor_hist;
    
    // Per-Gate 直方图（与 GPU 版本对应）
    std::array<HistogramCollector, 3> W_gate_hist;   // [z, r, g] 三个门的直方图
    std::array<HistogramCollector, 3> R_gate_hist;
    std::array<HistogramCollector, 3> bw_gate_hist;
    std::array<HistogramCollector, 3> br_gate_hist;

    GRUHistogramCollectors() = default;

    explicit GRUHistogramCollectors(int hidden, int num_bins = 2048)
        : hidden_(hidden), num_bins_(num_bins) {
        reset(hidden, num_bins);
    }

    void reset(int hidden = -1, int num_bins = -1) {
        if (hidden > 0) hidden_ = hidden;
        if (num_bins > 0) num_bins_ = num_bins;

        HistogramCollector::Config cfg;
        cfg.num_bins = num_bins_;

        x_hist = HistogramCollector(cfg);
        h_hist = HistogramCollector(cfg);
        Wx_hist = HistogramCollector(cfg);
        Rh_hist = HistogramCollector(cfg);
        update_gate_input_hist = HistogramCollector(cfg);
        reset_gate_input_hist = HistogramCollector(cfg);
        new_gate_input_hist = HistogramCollector(cfg);
        update_gate_output_hist = HistogramCollector(cfg);
        reset_gate_output_hist = HistogramCollector(cfg);
        new_gate_output_hist = HistogramCollector(cfg);
        mul_reset_hidden_hist = HistogramCollector(cfg);
        mul_new_contribution_hist = HistogramCollector(cfg);
        mul_old_contribution_hist = HistogramCollector(cfg);

        // Per-channel 直方图
        int channel_size = hidden_ * 3;
        W_hist.assign(channel_size, HistogramCollector(cfg));
        R_hist.assign(channel_size, HistogramCollector(cfg));
        bw_hist.assign(channel_size, HistogramCollector(cfg));
        br_hist.assign(channel_size, HistogramCollector(cfg));
        
        // Per-Tensor 直方图
        W_tensor_hist = HistogramCollector(cfg);
        R_tensor_hist = HistogramCollector(cfg);
        bw_tensor_hist = HistogramCollector(cfg);
        br_tensor_hist = HistogramCollector(cfg);
        
        // Per-Gate 直方图
        for (int i = 0; i < 3; ++i) {
            W_gate_hist[i] = HistogramCollector(cfg);
            R_gate_hist[i] = HistogramCollector(cfg);
            bw_gate_hist[i] = HistogramCollector(cfg);
            br_gate_hist[i] = HistogramCollector(cfg);
        }
    }

    bool is_valid() const { return hidden_ > 0 && x_hist.is_valid(); }

    void print() const {
        printf("GRUHistogramCollectors (hidden=%d, num_bins=%d):\n", hidden_, num_bins_);
        x_hist.histogram().print("x");
        h_hist.histogram().print("h");
        Wx_hist.histogram().print("Wx");
        Rh_hist.histogram().print("Rh");
        update_gate_input_hist.histogram().print("update_gate_input");
        reset_gate_input_hist.histogram().print("reset_gate_input");
        new_gate_input_hist.histogram().print("new_gate_input");
        update_gate_output_hist.histogram().print("update_gate_output");
        reset_gate_output_hist.histogram().print("reset_gate_output");
        new_gate_output_hist.histogram().print("new_gate_output");
    }
};

