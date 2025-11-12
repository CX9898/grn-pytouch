#pragma once

#include <vector>
#include <algorithm>

#include "devVector.h"

struct QuantParams {
  float scale;
  int32_t zero_point = 0; // 对称量化时固定为 0
};

struct QuantParams3 { // 对应三个门: z, r, g
  QuantParams gate[3]; // 每个门一个独立的量化参数
};

// scale 参数结构体
struct ScaleParam {
  int32_t M;  // M, 整数乘法系数，对齐到目标 scale 使用
  int shift;         // shift, 右移位数，用于 CUDA kernel
};

struct ScaleParam3 { // 对应三个门: z, r, g
  int32_t M[3];         // M, 整数乘法系数，对齐到目标 scale 使用
  int shift[3];         // shift, 右移位数，用于 CUDA kernel
};

struct GRUQuantScale {
  std::vector<float> Wx_scale; // size = time_steps * hidden * 3
  std::vector<float> Rh_scale; // size = time_steps * hidden * 3
  std::vector<float> z_pre; // size = time_steps * hidden
  std::vector<float> r_pre; // size = time_steps * hidden
  std::vector<float> g_pre; // size = time_steps * hidden
  std::vector<float> z_out; // size = time_steps * hidden
  std::vector<float> r_out; // size = time_steps * hidden
  std::vector<float> g_out; // size = time_steps * hidden
};

struct QuantGRUReScale { // size = time_steps * hidden
  dev::vector<ScaleParam> Rh_z_to_Wx_z;
  dev::vector<ScaleParam> Rh_r_to_Wx_r;
  dev::vector<ScaleParam> rRh_g_to_Wx_g;
  dev::vector<ScaleParam> Wx_to_z_pre;
  dev::vector<ScaleParam> Wx_to_r_pre;
  dev::vector<ScaleParam> Wx_to_g_pre;
  dev::vector<ScaleParam> zh_in_to_h_out;
  dev::vector<ScaleParam> zg_to_h_out;
};

struct QuantGRUScales {
  int steps;
  int hidden;
  std::vector<float> x; // size = steps
  std::vector<float> h; // size = steps + 1
  std::vector<float> Wx_z; // size = steps * hidden
  std::vector<float> Wx_r; // size = steps * hidden
  std::vector<float> Wx_g; // size = steps * hidden
  std::vector<float> Rh_z; // size = (steps + 1) * hidden
  std::vector<float> Rh_r; // size = (steps + 1) * hidden
  std::vector<float> Rh_g; // size = (steps + 1) * hidden
  std::vector<float> z_pre; // size = steps * hidden
  std::vector<float> r_pre; // size = steps * hidden
  std::vector<float> g_pre; // size = steps * hidden
  std::vector<float> z_out; // size = steps * hidden
  std::vector<float> r_out; // size = steps * hidden
  std::vector<float> g_out; // size = steps * hidden
};


struct GruQuantScales {
  std::vector<float> x_scale;  // 每步一组 scale/zp，非对称
  std::vector<int32_t> x_zp;

  std::vector<float> h_scale; // 动态计算

  QuantParams3 W; // 分为三个门: z, r, g
  QuantParams3 R; // 分为三个门: z, r, g

  QuantParams3 bx; // 分为三个门: z, r, g
  QuantParams3 br; // 分为三个门: z, r, g

  std::vector<QuantParams3> Wx;   // 分每步每Hidden, 且分为三个门: z, r, g
//  std::vector<QuantParams3> Rh;   // 每步一个 scale, 且分为三个门: z, r, g

  // --- 激活输入 ---
  QuantParams z_pre;  // sigmoid(z_pre) pre-activation
  QuantParams r_pre;  // sigmoid(r_pre)
  QuantParams g_pre;  // tanh(g_pre)

  // --- 激活输出 ---
  QuantParams z;  // 输出门 z
  QuantParams r;  // 输出门 r
  QuantParams g;  // 输出门 g
};


/**
 * @brief 组合两个 scale 参数，计算结果的定点比例系数
 * @param a    第一个 scale 参数（例如 W 或 R）
 * @param b    第二个 scale 参数（例如 x 或 h）
 * @return     组合后的 scale 参数（对应 a*b）
 */
inline ScaleParam combineSingleScale(const ScaleParam &a, const ScaleParam &b) {
    int64_t M_tmp = static_cast<int64_t>(a.M) * static_cast<int64_t>(b.M);
    int shift_tmp = a.shift + b.shift;

    // 归一化，防止溢出 32 位
    int norm_shift = 0;
    while (std::abs(M_tmp) > (1LL << 31)) {
        M_tmp >>= 1;
        norm_shift--;
    }

    ScaleParam result;
    result.M = static_cast<int32_t>(M_tmp);
    result.shift = shift_tmp + norm_shift;
    return result;
}

// --- ScaleParam3 × ScaleParam ---
inline ScaleParam3 combineScaleParam3(const ScaleParam3 &a, const ScaleParam &b) {
    ScaleParam3 result;
    for (int g = 0; g < 3; ++g) {
        ScaleParam single_a{a.M[g], a.shift[g]};
        ScaleParam combined = combineSingleScale(single_a, b);
        result.M[g] = combined.M;
        result.shift[g] = combined.shift;
    }
    return result;
}


template<typename T>
inline ScaleParam updateHScale(const ScaleParam &h_old, T max_val) {
    ScaleParam h_new;
    // scale 放大因子，例如 1e6 对应 20 位
    constexpr int32_t SCALE_FACTOR = 1 << 20;

    int32_t M_prev = h_old.M;   // 放大整数表示
    int32_t M_max = int32_t(max_val * SCALE_FACTOR); // float -> int

    // 更新 M，整数加权 (0.9/0.1)
    int32_t new_M = (9 * M_prev + 1 * M_max) / 10;

    // shift 一般保持不变或者可根据 new_M 调整
    int new_shift = h_old.shift;
    h_new.M = new_M;
    h_new.shift = new_shift;
}

/**
 * @brief 计算 Rh = R * h 的 ScaleParam3
 * @param R       输入 R 的量化参数 (三个门)
 * @param h       输入 h 的量化参数 (单个 M/shift)
 * @param SCALE_BITS 放大整数位数（如果 M 是放大整数）
 * @return        Rh 的量化参数 (三个门)
 */
inline ScaleParam3 combineRWithH(const ScaleParam3 &R, const ScaleParam &h, int SCALE_BITS = 20) {
    ScaleParam3 Rh;
    for (int i = 0; i < 3; ++i) {
        // M = (M_R * M_h) >> SCALE_BITS
        Rh.M[i] = int32_t((int64_t) R.M[i] * h.M >> SCALE_BITS);
        // shift = shift_R + shift_h
        Rh.shift[i] = R.shift[i] + h.shift;
    }
    return Rh;
}

// float scale -> M/shift
inline ScaleParam floatScaleToFixed(float scale) {
    // 假设使用 Q31 风格定点:
    // scale = M / (2^shift)
    // shift = ceil(log2(M/scale))
    int shift = 0;
    int32_t M = 0;

    if (scale > 0.f) {
        double q = std::frexp(scale, &shift); // scale = q * 2^shift, 0.5<=q<1
        M = static_cast<int32_t>(q * (1ll << 31)); // Q31
    } else {
        M = 0;
        shift = 0;
    }
    return {M, shift};
}

/**
 * @brief 计算将 Rh 结果重新缩放到 Wx 的定点 rescale 参数
 * @param scale_src   原始的浮点 scale
 * @param scale_dst   目标的浮点 scale
 * @param M           输出：定点乘法器（int32）
 * @param shift       输出：右移位数（int）
 */
inline ScaleParam computeRescaleParam(
    float scale_src,
    float scale_dst) {
    ScaleParam scaleParam;
    // 计算浮点比例
    double ratio = static_cast<double>(scale_src) / static_cast<double>(scale_dst);

    // 把比例转为 2^shift 形式
    int exp;
    double mantissa = std::frexp(ratio, &exp);  // ratio = mantissa * 2^exp, mantissa∈[0.5,1)

    // 将mantissa放大为整数
    const double SCALE_INT_RANGE = static_cast<double>(1 << 31);  // Q31 格式
    int64_t M64 = static_cast<int64_t>(std::round(mantissa * SCALE_INT_RANGE));

    // 调整 shift，使等式成立：ratio ≈ M / 2^(shift)
    scaleParam.M = static_cast<int32_t>(M64);
    scaleParam.shift = 31 - exp;
    return scaleParam;
}

struct GruQuantScalesFixed {
  // 输入 x（非对称量化，每步可能不同）
  std::vector<ScaleParam> x; // size = 时间步
  std::vector<int32_t> x_zp; // size = 时间步

  // 隐藏状态 h（对称量化，每步可能不同）
  std::vector<ScaleParam> h; // size = 时间步+1, 动态更新

  // 权重和偏置，三个门
  ScaleParam3 W;
  ScaleParam3 R;
  ScaleParam3 bx;
  ScaleParam3 br;

  // Wx 和 Rh 对齐参数
  std::vector<ScaleParam3> Wx; // size = 时间步
  std::vector<ScaleParam3> Rh; // size = 时间步+1, 动态更新

  // 激活前后
  ScaleParam z_pre, r_pre, g_pre;
  ScaleParam z, r, g;

  // ------------------------
  // 初始化函数：输入浮点 scale/zp
  // ------------------------
  void initialize(
      const GruQuantScales &gruQuantScales
  ) {
      const int seq_len = gruQuantScales.x_scale.size();
      x.resize(seq_len);
      x_zp.resize(seq_len);
      h.resize(gruQuantScales.h_scale.size());
      Wx.resize(seq_len);
      Rh.resize(seq_len + 1);

      // 权重和偏置
      W = float3ToFixed3(gruQuantScales.W);
      R = float3ToFixed3(gruQuantScales.R);
      bx = float3ToFixed3(gruQuantScales.bx);
      br = float3ToFixed3(gruQuantScales.br);

      // x/h 每步 scale -> M/shift
      for (int t = 0; t < seq_len; ++t) {
          x[t] = floatScaleToFixed(gruQuantScales.x_scale[t]);
          x_zp[t] = gruQuantScales.x_zp[t];

          ScaleParam3 wx_p;
          for (int g = 0; g < 3; ++g) {
              ScaleParam w_gate{W.M[g], W.shift[g]};
              auto combined = combineSingleScale(w_gate, x[t]);
              wx_p.M[g] = combined.M;
              wx_p.shift[g] = combined.shift;
          }
          Wx[t] = wx_p;
      }

      // 时间步0的h_scale, 后续动态更新
      h[0] = floatScaleToFixed(gruQuantScales.h_scale[0]);
      {
          ScaleParam3 Rh_p;
          for (int g = 0; g < 3; ++g) {
              ScaleParam R_gate{R.M[g], R.shift[g]};
              auto combined = combineSingleScale(R_gate, h[0]);
              Rh_p.M[g] = combined.M;
              Rh_p.shift[g] = combined.shift;
          }
          Rh[0] = Rh_p;
      }

      // 激活前后默认用 M=1, shift=0
//      z_pre = {1, 0, 0};
//      r_pre = {1, 0, 0};
//      g_pre = {1, 0, 0};
//      z = {1, 0, 0};
//      r = {1, 0, 0};
//      g = {1, 0, 0};
  }

 private:


  ScaleParam3 float3ToFixed3(const QuantParams3 &fp) {
      ScaleParam3 out;
      for (int i = 0; i < 3; i++) {
          ScaleParam scale = floatScaleToFixed(fp.gate[i].scale);
          out.M[i] = scale.M;
          out.shift[i] = scale.shift;
      }
      return out;
  }
};


/**
 * @brief 计算一个张量的定点缩放参数对齐到另一个张量的 scale
 * @param M_src       源张量的 M
 * @param shift_src   源张量的 shift
 * @param M_dst       目标张量的 M
 * @param shift_dst   目标张量的 shift
 * @param N           固定精度（通常取31）
 * @return            对齐参数 (M_align, shift_align)
 */
inline ScaleParam computeRescaleTo(
    const ScaleParam src,
    const ScaleParam dst,
    int N = 31) {
    ScaleParam p;

    // 检查边界情况
    if (dst.M == 0 || src.M == 0) {
        p.M = 1;
        p.shift = 0;
        return p;
    }

    // 计算新的 M （放大到 QN 精度后再除）
    // scale_ratio = (src.M / (2^src.shift)) / (dst.M / (2^dst.shift))
    //             = (src.M * 2^dst.shift) / (dst.M * 2^src.shift)
    // 使用 QN 精度：M_rescale = (2^N * src.M) / dst.M
    // shift_rescale = N + src.shift - dst.shift
    // 这样 rescale 后的 scale = M_rescale / (2^shift_rescale) = (src.M * 2^dst.shift) / (dst.M * 2^src.shift)

    // 计算 numerator = 2^N * src.M
    int64_t numerator = (int64_t(1) << N) * (int64_t) src.M;

    // 四舍五入除法
    int64_t tmp = (numerator + (dst.M / 2)) / dst.M;

    // 检查溢出
    if (tmp > std::numeric_limits<int32_t>::max()) {
        tmp = std::numeric_limits<int32_t>::max();
    } else if (tmp < std::numeric_limits<int32_t>::min()) {
        tmp = std::numeric_limits<int32_t>::min();
    }

    p.M = static_cast<int32_t>(tmp);
    p.shift = N + src.shift - dst.shift;

    // 检查 shift 是否合理
    if (p.shift < 0 || p.shift > 63) {
        // 如果 shift 不合理，使用默认值
        p.M = 1;
        p.shift = 0;
    }

    return p;
}

/**
 * @brief 对齐 Rh 的 scale 到 Wx 的 scale，计算 rescale 参数
 * @param Rh        Rh 的量化参数 (三个门)
 * @param Wx        Wx 的量化参数 (三个门，目标 scale)
 * @return          Rh_to_Wx 的 rescale 参数 (三个门)
 *
 * 说明：
 * - Rh 的 scale = M_Rh / (2^shift_Rh)
 * - Wx 的 scale = M_Wx / (2^shift_Wx)
 * - 要将 Rh 对齐到 Wx，需要：Rh_aligned = Rh_val * (M_Rh/M_Wx) * (2^(shift_Wx - shift_Rh))
 * - 这个函数计算的是 rescale 参数，使得 rescale(Rh_val, M, shift) 的结果 scale 等于 Wx 的 scale
 */
inline ScaleParam3 alignRhToWxShift(const ScaleParam3 &Rh, const ScaleParam3 &Wx) {
    ScaleParam3 Rh_to_Wx;

    for (int i = 0; i < 3; ++i) {
        // 检查边界情况
        if (Wx.M[i] == 0 || Rh.M[i] == 0) {
            // 如果 M 为 0，使用默认值
            Rh_to_Wx.M[i] = 1;
            Rh_to_Wx.shift[i] = 0;
            continue;
        }

        // 使用 computeRescaleTo 来计算对齐参数
        // 这将 Rh 的 scale 对齐到 Wx 的 scale
        ScaleParam src{Rh.M[i], Rh.shift[i]};
        ScaleParam dst{Wx.M[i], Wx.shift[i]};
        ScaleParam aligned = computeRescaleTo(src, dst, 31);

        // 检查结果是否有效
        if (aligned.M == 0) {
            // 如果计算出的 M 为 0，使用默认值
            Rh_to_Wx.M[i] = 1;
            Rh_to_Wx.shift[i] = 0;
        } else {
            Rh_to_Wx.M[i] = aligned.M;
            Rh_to_Wx.shift[i] = aligned.shift;
        }
    }

    return Rh_to_Wx;
}

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
  ScaleParam3 Rh_to_Wx;

  /**
   * @brief bx → Wx 对齐的 rescale 参数
   *
   * 将输入偏置 bx 对齐到 Wx 的 scale。
   * 每个门（z、r、g）分别对应一个缩放因子。
   */
  ScaleParam3 bx_to_Wx;

  /**
   * @brief br → Wx 对齐的 rescale 参数
   *
   * 将隐藏偏置 br 对齐到 Wx 的 scale。
   * 每个门（z、r、g）分别对应一个缩放因子。
   */
  ScaleParam3 br_to_Wx;

  /**
   * @brief Wx → 输出门(z/r/g) 对齐的 rescale 参数
   *
   * 将累加结果 (Wx + Rh + bx + br) 对齐到门激活函数（sigmoid / tanh）输入的 scale。
   * 通常用于将中间 int32 累加结果量化回 int8。
   */
  ScaleParam3 Wx_to_out;

  /**
   * @brief r → Wx_g 对齐的 rescale 参数
   *
   * 在候选状态 g 的计算中，需要用 r * (Rh_g + br_g)，
   * 此时需要将门 r 的输出 scale 对齐到 Wx_g 的 scale。
   * 只对 g 门有效，z、r 门的该值可忽略。
   */
  ScaleParam r_to_Wx_g;
};


/**
 * @param src_scale     源张量的量化 scale (float)
 * @param dst_scale     目标张量的量化 scale (float)
 * @param fixed_shift   固定的右移位数，用于 kernel 右移，默认 15
 * @return ScaleParam 包含整数 multiplier 和 kernel 右移 shift
 */
inline ScaleParam computeRescaleParamFixedShift(float src_scale, float dst_scale, int fixed_shift = 15);


template<typename QuantT>
GruQuantScales computeGruQuantParams(const float *x, int steps, int N, int C,
                                     const float *W, int H,
                                     const float *R,
                                     const float *bx, const float *br);


/**
 * @brief 计算量化参数（对称或非对称）
 * @tparam QuantT   目标量化类型（如 int8_t 或 int16_t）
 * @param data      输入浮点数据指针
 * @param size      数据长度
 * @param symmetric 是否使用对称量化（默认为 true）
 * @return QuantParams 量化参数结构体
 */
template<typename QuantT>
QuantParams calculateQuantParams(
    const float *data,
    size_t size,
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
void computeWeightSum(
    const T *W_q,// [out_dim, in_dim] 权重量化矩阵
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
 * @brief 计算每个时间步的 ScaleParam3 参数（用于 Wx）
 *
 * @param steps           [in] 时间步数（序列长度）
 * @param x_scales        [in] 每个时间步输入 x 的量化 scale, size = steps
 * @param w_scale_z       [in] Wz 的对称量化 scale
 * @param w_scale_r       [in] Wr 的对称量化 scale
 * @param w_scale_g       [in] Wg 的对称量化 scale
 * @param rescale_params  [out] 每步输出的 ScaleParam3 数组, size = steps, 对应三个门 z,r,g
 */
void computeWxRescaleParamsFixedShift(
    int steps,
    const std::vector<float> &x_scales,
    const float w_scale_z,
    const float w_scale_r,
    const float w_scale_g,
    std::vector<ScaleParam3> &rescale_params
);

/**
 * @brief 计算每个时间步的所有 RescaleParamsPerStep 参数
 *
 * @param steps           [in] 时间步数（序列长度）
 * @param gruQuantScales  [in] GRU 量化参数
 * @param h_scales        [in] 每个时间步隐藏状态 h 的量化 scale, size = steps + 1 (包含初始 h[0])
 * @param rescale_params  [out] 每步输出的 RescaleParamsPerStep 数组, size = steps
 * @param fixed_shift     [in] 固定的右移位数，默认 15
 */
void computeGruRescaleParamsPerStep(
    int steps,
    GruQuantScales &gruQuantScales,
    std::vector<RescaleParamsPerStep> &rescale_params,
    int fixed_shift = 15
);

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

template<typename T>
T findMaxValueFromDev(const T *dev_data, size_t size);

template<typename T>
T findMinValueFromDev(const T *dev_data, size_t size);

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
 * @brief 计算量化参数（对称或非对称）
 * @tparam QuantT   目标量化类型（如 int8_t 或 int16_t）
 * @param data      输入浮点数据指针
 * @param size      数据长度
 * @param symmetric 是否使用对称量化（默认为 true）
 * @return QuantParams 量化参数结构体
 */
template<typename QuantT>
inline QuantParams calculateQuantParams(float max_val, float min_val, bool symmetric = true) {
    QuantParams params;

    // -----------------------------
    // 获取目标类型范围
    // -----------------------------
    const int32_t int_min = static_cast<int32_t>(std::numeric_limits<QuantT>::min());
    const int32_t int_max = static_cast<int32_t>(std::numeric_limits<QuantT>::max());

    // -----------------------------
    // 根据模式计算 scale 和 zero_point
    // -----------------------------
    if (symmetric) {
        // 对称量化：zero_point = 0
        float abs_max = std::max(std::abs(max_val), std::abs(min_val));
        params.scale = abs_max / static_cast<float>(int_max);
        params.zero_point = 0;
    } else {
        // 非对称量化：线性映射 [min_val, max_val] → [int_min, int_max]
        params.scale = (max_val - min_val) / static_cast<float>(int_max - int_min);
        if (params.scale < 1e-12f) params.scale = 1e-12f; // 避免除0
        params.zero_point = static_cast<int32_t>(std::round(int_min - min_val / params.scale));
        params.zero_point = std::clamp(params.zero_point, int_min, int_max);
    }

    return params;
}
