#pragma once

#include <vector>

struct QuantParams {
  float scale;
  int32_t zero_point = 0; // 对称量化时固定为 0
};

struct QuantParams3 { // 对应三个门: z, r, g
  QuantParams gate[3]; // 每个门一个独立的量化参数
};

struct GruQuantScales {
  std::vector<QuantParams> x;  // 每步一组 scale/zp，非对称

  QuantParams3 W; // 分为三个门: z, r, g
  QuantParams3 R; // 分为三个门: z, r, g

  QuantParams3 bx; // 分为三个门: z, r, g
  QuantParams3 br; // 分为三个门: z, r, g

  std::vector<QuantParams3> Wx;   // 每步一个 scale, 且分为三个门: z, r, g
  std::vector<QuantParams3> Rh;   // 每步一个 scale, 且分为三个门: z, r, g

  // --- 激活输入 ---
  QuantParams z_pre;  // sigmoid(z_pre) pre-activation
  QuantParams r_pre;  // sigmoid(r_pre)
  QuantParams g_pre;  // tanh(g_pre)

  // --- 激活输出 ---
  QuantParams z;  // 输出门 z
  QuantParams r;  // 输出门 r
  QuantParams g;  // 输出门 g
};

// Rescale 参数结构体
struct RescaleParam {
  int32_t M;  // M, 整数乘法系数，对齐到目标 scale 使用
  int shift;         // shift, 右移位数，用于 CUDA kernel
};

struct RescaleParam3 { // 对应三个门: z, r, g
  int32_t M[3];         // M, 整数乘法系数，对齐到目标 scale 使用
  int shift[3];         // shift, 右移位数，用于 CUDA kernel
};

/**
 * @brief 每个时间步（step）对应的量化重标定参数集合，用于 GRU 三个门（z、r、g）的定点缩放对齐。
 *
 * 说明：
 * 在 GRU 的前向传播中，不同张量（Wx、Rh、bx、br、r 等）之间的 scale 不同，
 * 需要通过整数乘法 + 右移的方式进行定点 rescale 对齐。
 * 本结构体封装了每个时间步中用于三个门（z、r、g）的所有重标定参数。
 */
struct RescaleParamsPerStep {

  /**
   * @brief Rh → Wx 对齐的 rescale 参数
   *
   * 将隐藏层乘积 Rh（R * h）重标定到 Wx 的 scale。
   * 每个门（z、r、g）分别有自己的 M 和 shift。
   *
   * 对应公式：
   *   Rh_aligned = (Rh * M[i]) >> shift[i]
   */
  RescaleParam3 Rh_to_Wx;

  /**
   * @brief bx → Wx 对齐的 rescale 参数
   *
   * 将输入偏置 bx 对齐到 Wx 的 scale。
   * 每个门（z、r、g）分别对应一个缩放因子。
   */
  RescaleParam3 bx_to_Wx;

  /**
   * @brief br → Wx 对齐的 rescale 参数
   *
   * 将隐藏偏置 br 对齐到 Wx 的 scale。
   * 每个门（z、r、g）分别对应一个缩放因子。
   */
  RescaleParam3 br_to_Wx;

  /**
   * @brief Wx → 输出门(z/r/g) 对齐的 rescale 参数
   *
   * 将累加结果 (Wx + Rh + bx + br) 对齐到门激活函数（sigmoid / tanh）输入的 scale。
   * 通常用于将中间 int32 累加结果量化回 int8。
   */
  RescaleParam3 Wx_to_out;

  /**
   * @brief r → Wx_g 对齐的 rescale 参数
   *
   * 在候选状态 g 的计算中，需要用 r * (Rh_g + br_g)，
   * 此时需要将门 r 的输出 scale 对齐到 Wx_g 的 scale。
   * 只对 g 门有效，z、r 门的该值可忽略。
   */
  RescaleParam r_to_Wx_g;
};


/**
 * @param src_scale     源张量的量化 scale (float)
 * @param dst_scale     目标张量的量化 scale (float)
 * @param fixed_shift   固定的右移位数，用于 kernel 右移，默认 15
 * @return RescaleParam 包含整数 multiplier 和 kernel 右移 shift
 */
inline RescaleParam computeRescaleParamFixedShift(float src_scale, float dst_scale, int fixed_shift = 15);


template<typename QuantT>
GruQuantScales computeGruQuantParams(const float *x, int steps, int N, int C,
                                     const float *W, int H,
                                     const float *R,
                                     const float *bx, const float *br);


/**
 * @brief 计算量化参数 scale 和 zero_point
 * @tparam QuantT     目标量化类型 (int8_t 或 int16_t)
 * @param host_data    输入数据指针 (float*)
 * @param size         输入数据元素数量
 * @param scale        输出缩放因子 (float)
 * @param zero_point   输出零点 (int32_t)
 * @param symmetric    是否使用对称量化
 */
template<typename QuantT>
void calculateScaleZeroPoint(
    const float *host_data,
    size_t size,
    float &scale,
    int32_t &zero_point,
    bool symmetric = true);

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

void computeWeightSum(
    const int8_t *W_q,// [out_dim, in_dim] 权重量化矩阵
    int32_t *weight_sum,// [out_dim] 输出数组
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
 * @brief 计算每个时间步的 RescaleParam3 参数（用于 Wx）
 *
 * @param steps           [in] 时间步数（序列长度）
 * @param x_scales        [in] 每个时间步输入 x 的量化 scale, size = steps
 * @param w_scale_z       [in] Wz 的对称量化 scale
 * @param w_scale_r       [in] Wr 的对称量化 scale
 * @param w_scale_g       [in] Wg 的对称量化 scale
 * @param rescale_params  [out] 每步输出的 RescaleParam3 数组, size = steps, 对应三个门 z,r,g
 */
void computeWxRescaleParamsFixedShift(
    int steps,
    const std::vector<QuantParams> &x_scales,
    const float w_scale_z,
    const float w_scale_r,
    const float w_scale_g,
    std::vector<RescaleParam3> &rescale_params
);
