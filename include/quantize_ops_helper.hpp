#pragma once

#include <vector>
#include <algorithm>

#include "devVector.h"

struct GRUQuantitativeParameters {
  int hidden_;
  float scale_x_;
  int32_t zp_x_;
  float scale_h_;
  int32_t zp_h_;

  std::vector<float> scale_W_; // size = hidden
  std::vector<float> scale_R_; // size = hidden

  float scale_Wx_;
  int32_t zp_Wx_;
  float scale_Rh_;
  int32_t zp_Rh_;

  std::vector<float> scale_bx_;
  std::vector<float> scale_br_;

  float scale_z_pre_;
  int32_t zp_z_pre_;
  float scale_r_pre_;
  int32_t zp_r_pre_;
  float scale_g_pre_;
  int32_t zp_g_pre_;

  float scale_z_out_;
  int32_t zp_z_out_;
  float scale_r_out_;
  int32_t zp_r_out_;
  float scale_g_out_;
  int32_t zp_g_out_;

  std::vector<float> scale_Rh_add_br_;
  std::vector<float> scale_rRh_;

  float scale_one_minus_update_;
  int32_t zp_one_minus_update_;
  float scale_new_contrib_;
  int32_t zp_new_contrib_;
  float scale_old_contrib_;
  int32_t zp_old_contrib_;
};

struct QuantGRUReScale {
  int32_t zp_x_;
  int32_t zp_h_;

  dev::vector<int32_t> n_Wx_;
  int32_t zp_Wx_;
  dev::vector<int32_t> n_Rh_;
  int32_t zp_Rh_;

  // z门
  int32_t zp_z_pre_;
  int32_t zp_z_out_;
  int32_t n_Wx_to_z_; // size = hidden
  int32_t n_Rh_to_z_;
  dev::vector<int32_t> n_bx_to_z_;
  dev::vector<int32_t> n_br_to_z_;

  // r门
  int32_t zp_r_pre_;
  int32_t zp_r_out_;
  int32_t n_Wx_to_r_;
  int32_t n_Rh_to_r_;
  dev::vector<int32_t> n_bx_to_r_;
  dev::vector<int32_t> n_br_to_r_;

  // n门
  int32_t zp_g_pre_;
  int32_t zp_g_out_;
  int32_t n_Rh_to_Rh_add_br_;
  dev::vector<int32_t> n_br_to_Rh_add_br_; // br 是 per-channel
  int32_t zp_Rh_add_br_;
  dev::vector<int32_t> n_Wx_to_g_;
  dev::vector<int32_t> n_rRh_to_g_;
  dev::vector<int32_t> n_bx_to_g_;

  // h_new
  int32_t zp_old_contrib_;
  int32_t n_one_minus_update_;
  int32_t c12_; // 不需要加zp_omu
  int32_t zp_new_contrib_;
  int32_t n13_;
  int32_t n14_;
  int32_t n15_;
  int32_t n16_;
  int32_t c15_;
};
//
//struct QuantGRUScales {
//  int hidden_;
//  float scale_x_;
//  int32_t zp_x_;
//  float scale_h_;
//  int32_t zp_h_;
//  float scale_bx_;
//  float scale_br_;
//  std::vector<float> Wx; // size = hidden. per-channel
//  std::vector<float> Rh; // size = hidden
//
//  std::vector<float> Wx_add_bx; // size = hidden
//  std::vector<float> Rh_add_br; // size = hidden
//
//  std::vector<float> z_pre; // size = hidden
//  std::vector<float> r_pre; // size = hidden
//  std::vector<float> g_pre; // size = hidden
//
//  std::vector<float> z_out; // size = hidden
//  std::vector<float> r_out; // size = hidden
//  std::vector<float> g_out; // size = hidden
//};

void generate_int8_lut(float scale_z_pre, int32_t zp_z_pre, float scale_z_out, int32_t zp_z_out,
                       float scale_r_pre, int32_t zp_r_pre, float scale_r_out, int32_t zp_r_out,
                       float scale_g_pre, int32_t zp_g_pre, float scale_g_out, int32_t zp_g_out);


__host__ __device__ __forceinline__ int32_t rshift_round(int32_t x, int n) {
    if (n <= 0) return x << (-n);  // 可支持负 shift（左移）
    int32_t offset = 1 << (n - 1); // 四舍五入
    return (x >= 0 ? (x + offset) : (x - offset)) >> n;
}

/**
 * @brief 在 GPU 上将 float 数据量化为 int8
 * @tparam QuantT       目标量化类型（int8_t 或 int16_t）
 * @tparam use_inv_scale 是否使用 inv_scale（乘法而非除法）
 * @tparam symmetric    是否使用对称量化（zero_point=0）
 * @tparam clamp    是否使用饱和处理 (对bias不处理)
 * @param src_dev    输入 float 指针（GPU 内存）
 * @param dst_dev    输出 int8 指针（GPU 内存）
 * @param size       元素数量
 * @param scale      量化 scale
 * @param zero_point 量化 zero_point（非对称量化有效）
 */
template<typename QuantT, bool use_inv_scale, bool symmetric, bool clamp = true>
void quantizeFloatToInt(const float *src_dev,
                        QuantT *dst_dev,
                        uint32_t size,
                        float scale,
                        int32_t zero_point = 0);

/**
 * @brief 在 GPU 上将 float 数据量化为 int8/int16（支持每个时间步独立 scale）
 * @tparam QuantT       目标量化类型（int8_t 或 int16_t）
 * @tparam use_inv_scale 是否使用 inv_scale（乘法而非除法）
 * @tparam symmetric    是否使用对称量化（zero_point=0）
 * @tparam clamp        是否使用饱和处理
 * @param src_dev       输入 float 指针（GPU 内存）
 * @param dst_dev       输出 int8/int16 指针（GPU 内存）
 * @param size          总元素数量
 * @param scale_per_t   每个时间步的量化 scale 数组（GPU 内存，长度为 time_steps）
 * @param zero_point    每个时间步的量化 zero_point（非对称量化有效）
 * @param time_step_size 每个时间步的元素数（例如 batch_size * input_dim）
 */
template<typename QuantT, bool use_inv_scale, bool symmetric, bool clamp = true>
void quantizeFloatToIntPerStep(const float *src_dev,
                               QuantT *dst_dev,
                               size_t size,
                               const float *scale_per_t,
                               const int32_t *zero_point_per_t,
                               int time_step_size);

template<typename T>
void computeWeightSumMulzp(
    const T *W_q,// [out_dim, in_dim] 权重量化矩阵
    int32_t *weight_sum,// [out_dim] 输出数组
    int zp,
    int out_dim,// 输出通道数 (M)
    int in_dim,// 输入通道数 (K)
    cudaStream_t stream = 0);

void applyZeroPointCompensation2D(
    int32_t *Y_int32,
    const int32_t *weight_sum,
    const int32_t *x_zp,
    int out_dim,
    int batch_size,
    cudaStream_t stream = 0);

/**
 * @brief 从 GPU 上的量化数据计算 scale（使用最大最小值）
 *
 * @tparam QuantT         量化类型（int8_t 或 int16_t）
 * @param h_dev           [in] GPU 上的量化数据指针
 * @param size            [in] 数据元素数量
 * @param scale           [out] 输出的 scale
 * @param zero_point      [out] 输出的 zero_point
 * @param symmetric       [in] 是否使用对称量化
 * @param stream          [in] CUDA stream
 */
template<typename QuantT>
void calculateScaleZeroPointFromDevice(
    const QuantT *h_dev,
    size_t size,
    float &scale,
    int32_t &zero_point,
    bool symmetric = true,
    cudaStream_t stream = 0);

//template<typename T>
//T findMaxValueFromDev(const T *dev_data, size_t size);
//
//template<typename T>
//T findMinValueFromDev(const T *dev_data, size_t size);

/**
 * @brief 使用 (M, shift) 参数将量化值反量化为浮点数
 * @tparam QuantT      量化类型（int8_t / int16_t / int32_t）
 * @param quant_data   输入量化数据指针
 * @param size         数据元素数量
 * @param M            定点缩放系数（整数）
 * @param shift        缩放右移位数
 * @param dequant_data 输出反量化后的 float 数组
 */
template<typename QuantT>
void dequantizeTensorFixedPoint(const QuantT *quant_data,
                                size_t size,
                                int32_t M,
                                int shift,
                                float *dequant_data) {
    // 计算等效的scale（float），只在CPU上调试时使用
    const float scale = static_cast<float>(M) / static_cast<float>(1 << shift);

    for (size_t i = 0; i < size; ++i) {
        const int32_t q = static_cast<int32_t>(quant_data[i]);
        dequant_data[i] = q * scale;
    }
}

// 定义常量
constexpr int32_t Q15_ONE = 32768;
constexpr int32_t ALPHA_Q15 = 29491; // 0.9 * 32768
constexpr int32_t INV_QMAX = (1 << 15) / 127; // 257 in Q15

// 输入: 上一步scale参数 (M_prev, shift_prev)
// 输入: 当前步隐藏态整数张量 h_t_int[]
// 输出: 更新后的scale参数 (M_new, shift_new)
inline void updateHScaleInt8(const int8_t *h_t, size_t size,
                             int32_t &M_prev, int &shift_prev) {
    // 1. 求当前步最大值
    int max_abs = 0;
    for (size_t i = 0; i < size; ++i)
        max_abs = std::max(max_abs, abs((int) h_t[i]));

    // 2. ratio 定点化 (Q15)
    int32_t ratio_q15 = (max_abs * INV_QMAX); // Q15 格式

    // 3. EMA 更新 (Q15)
    static int32_t s_prev_q15 = Q15_ONE; // 初始scale比例=1.0
    int32_t s_new_q15 = (ALPHA_Q15 * s_prev_q15 +
                         (Q15_ONE - ALPHA_Q15) * ratio_q15 + (1 << 14)) >> 15;
    s_prev_q15 = s_new_q15;

    // 4. 更新 M (scale整数因子)
    M_prev = (M_prev * s_new_q15 + (1 << 14)) >> 15;

    // shift_prev 可视范围动态调整（或保持不变）
}

/**
 * @brief 计算缩放因子 S 对应的右移位数 n（使 S ≈ 2^(-n)）
 * @param S 待转换的浮点数缩放因子（必须 > 0）
 * @param min_n 允许的最小右移位数（默认 0，避免左移溢出）
 * @param max_n 允许的最大右移位数（默认 30，适配 32 位整数运算）
 * @return 最优右移位数 n（int 类型，在 [min_n, max_n] 范围内）
 * @throws std::invalid_argument 若 S ≤ 0，抛出异常
 */
inline int32_t calculate_right_shift_bits(float S, int32_t min_n = 0, int32_t max_n = 30) {
    // 输入合法性检查：缩放因子必须为正
    if (S <= 0.0f) {
        throw std::invalid_argument("Scale factor S must be greater than 0. Current value: " + std::to_string(S));
    }

    // 步骤1：计算理论最优n（n = -log2(S)）
    // log2f 是单精度浮点数对数函数，效率高于 log2（double 类型）
    float log2_S = log2f(S);
    float n_theory = -log2_S;

    // 步骤2：四舍五入到最近整数（C++11+ 支持 roundf 单精度 rounding）
    int32_t n_candidate = static_cast<int32_t>(roundf(n_theory));

    // 步骤3：边界裁剪，确保n在合理范围（避免移位溢出或数值归零）
    int32_t n = std::max(min_n, std::min(n_candidate, max_n));

//    // （可选）验证近似效果（调试用，发布时可注释）
//    float S_approx = powf(2.0f, -static_cast<float>(n));
//    float relative_error = (fabsf(S - S_approx) / S) * 100.0f;
//    std::cout << "[DEBUG] S: " << S << " | Approx S=2^(-" << n << "): " << S_approx
//              << " | Relative Error: " << relative_error << "%" << std::endl;

    return n;
}


/**
 * @brief 验证 S_omu 是否为 2的负幂次浮点数，并计算 1/S_omu（整数结果）
 * @param S_omu 输入的缩放因子（浮点数，需符合量化工程约定：S_omu = 2^(-m), m为整数）
 * @param tolerance 浮点数精度容差（默认 1e-6，应对浮点存储微小误差）
 * @return 1/S_omu 的整数结果（int32_t 类型，因量化场景中 m 通常≤30，2^30≈1e9，适配32位整数）
 * @throws std::invalid_argument 若 S_omu 不是 2的负幂次，或 S_omu ≤0
 */
inline int32_t calculate_one_over_Somu(float S_omu, float tolerance = 1e-6f) {
    // 1. 基础合法性检查：S_omu 必须为正
    if (S_omu <= 0.0f) {
        throw std::invalid_argument("S_omu must be positive! Current value: " + std::to_string(S_omu));
    }

    // 2. 计算理论 m 值：由 S_omu = 2^(-m) → m = -log2(S_omu)
    float log2_Somu = log2f(S_omu);
    float m_theory = -log2_Somu;

    // 3. 验证 m 是否为整数（核心：S_omu 必须是 2的负幂次）
    int32_t m = static_cast<int32_t>(roundf(m_theory)); // 四舍五入到最近整数
    float S_omu_approx = powf(2.0f, -static_cast<float>(m)); // 由 m 反推理论 S_omu

    // 4. 检查实际 S_omu 与理论值的误差是否在容差内
    float absolute_error = fabsf(S_omu - S_omu_approx);
    if (absolute_error > tolerance) {
        throw std::invalid_argument(
            "S_omu is not a negative power of 2! "
            "Current S_omu: " + std::to_string(S_omu) + ", "
                                                        "Nearest 2^(-m): " + std::to_string(S_omu_approx) + ", "
                                                                                                            "Error: " +
            std::to_string(absolute_error)
        );
    }

    // 5. 计算 1/S_omu = 2^m（整数结果），并检查溢出
    if (m < 0) {
        throw std::invalid_argument("m = " + std::to_string(m) + " is negative! S_omu is too large (exceeds 1.0f)");
    }
    if (m > 30) { // 2^30 = 1073741824，2^31 会溢出 int32_t（最大值 2147483647）
        throw std::overflow_error("m = " + std::to_string(m) + " is too large! 2^m exceeds int32_t limit");
    }

    int32_t one_over_Somu = 1 << m; // 等价于 2^m，用左移实现整数乘法（无浮点运算）
    return one_over_Somu;
}
