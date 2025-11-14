#pragma once

#include <cuda_runtime.h>
#include <cstdint>

constexpr int32_t shift_Rh = 24;
constexpr int32_t shift_br = 24;

extern __constant__ int8_t d_sigmoid_int8_z_lut[256];
extern __constant__ int8_t d_sigmoid_int8_r_lut[256];
extern __constant__ int8_t d_tanh_int8_g_lut[256];

void initLut();

namespace dev {

template<typename T>
__device__ __forceinline__ T clamp(int x);

template<>
__device__ __forceinline__ int8_t clamp(int x) {
    return static_cast<int8_t>(max(-128, min(127, x)));
}

template<>
__device__ __forceinline__ int16_t clamp(int x) {
    return static_cast<int8_t>(max(-32768, min(32767, x)));
}

template<>
__device__ __forceinline__ int32_t clamp(int x) {
    return static_cast<int8_t>(max(-32768, min(32767, x)));
}

/**
 * @brief 将 float 转 int8（GPU device 函数），支持 use_inv_scale 和对称量化
 * @tparam use_inv_scale 是否使用 inv_scale（乘法而非除法）
 * @tparam symmetric    是否使用对称量化（zero_point=0）
 */
template<bool use_inv_scale, bool symmetric>
__device__ __forceinline__ int8_t quantize_float_to_int8(
    const float value,
    const float scale_param,
    const int32_t zero_point
) {
    // 1. 编译期分支选择 scale 计算方式
    const float scaled = [value, scale_param]() {
      if constexpr (use_inv_scale) {
          return value * scale_param;
      } else {
          return value / scale_param;
      }
    }();

    // 2. 对称量化时 zero_point 固定为 0，非对称时使用传入 zero_point
    const int32_t zp = symmetric ? 0 : zero_point;

    // 3. 添加 zero_point
    const float shifted = scaled + static_cast<float>(zp);

    // 4. 四舍五入并截断到 int8 范围
    const int32_t rounded = __float2int_rn(shifted);
    const int32_t clamped = ::max(-128, ::min(127, rounded));

    return static_cast<int8_t>(clamped);
}

__device__ __forceinline__ int8_t quantize_i32_to_i8(
    const int32_t value,
    const int32_t M,
    const int32_t shift,
    const int32_t zero_point = 0) {
    int64_t tmp64;
    if (shift > 0) {
        tmp64 = static_cast<int64_t>(value) * M + (1LL << (shift - 1));
        tmp64 >>= shift;
    } else if (shift == 0) {
        tmp64 = static_cast<int64_t>(value) * M;
    } else { // shift < 0
        tmp64 = static_cast<int64_t>(value) * M << (-shift);
    }

    tmp64 += zero_point;
    tmp64 = max(static_cast<int64_t>(-128), min(static_cast<int64_t>(127), tmp64));
    return static_cast<int8_t>(tmp64);
}

__device__ __forceinline__ int16_t quantize_i32_to_i16(
    const int32_t value,
    const int32_t M,
    const int32_t shift,
    const int32_t zero_point) {
    int32_t tmp = (value * M + (1 << (shift - 1))) >> shift;
    tmp += zero_point;
    tmp = max(-32768, min(32767, tmp));
    return static_cast<int16_t>(tmp);
}

__device__ __forceinline__ int8_t sigmoid_int8_lut(int8_t x, const int8_t* lut) {
    // x in [-128,127], lut 长度 = 256
    const int8_t x_clamped = max(-128, min(127, x));
    return lut[static_cast<uint8_t>(x_clamped)]; // uint8_t 转索引 [0,255]
}

__device__ __forceinline__ int8_t tanh_int8_lut(int8_t x, const int8_t* lut) {
    const int8_t x_clamped = max(-128, min(127, x));
    return lut[static_cast<uint8_t>(x_clamped)];
}

__device__ __forceinline__ int8_t sigmoid_int16_lut(int16_t x) { // (TODO: 二项式拟合查表方式)
    // 将 int16_t 范围 [-32768, 32767] 映射到 int8_t 范围 [-128, 127]
    // 公式：idx = round( (x + 32768) * (255.0f / 65535.0f) ) - 128
    // 整数优化：避免浮点运算，用移位实现近似缩放
    int32_t tmp = static_cast<int32_t>(x) + 32768; // 转为 [0, 65535]
    tmp = (tmp * 255 + 65535 / 2) / 65535; // 四舍五入缩放到 [0, 255]
    int8_t idx = static_cast<int8_t>(tmp - 128); // 转为 [-128, 127]
    return d_sigmoid_lut[static_cast<uint8_t>(idx)];

    // -10到10分成N32段, 每段用二次多项式拟合

    // PDQ
    // QAT 训练
}

__device__ __forceinline__ int8_t tanh_int16_lut(int16_t x) { // (TODO: 二项式拟合查表方式)
    // 与 sigmoid 完全相同的索引映射逻辑
    int32_t tmp = static_cast<int32_t>(x) + 32768; // int16_t [-32768, 32767] → [0, 65535]
    tmp = (tmp * 255 + 65535 / 2) / 65535; // 缩放到 [0, 255]（四舍五入）
    int8_t idx = static_cast<int8_t>(tmp - 128); // → [-128, 127]
    return d_tanh_lut[static_cast<uint8_t>(idx)]; // 用索引访问 tanh LUT
}

__device__ __forceinline__ int32_t round_shift(int32_t val, int shift) {
    if (shift <= 0) {
        return val; // 右移位数≤0时，无需计算，直接返回原数（避免移位错误）
    }
    const int32_t half = 1 << (shift - 1); // 偏移量（正数的半值）
    // 区分正负：正数加half，负数减half，确保四舍五入方向正确
    if (val >= 0) {
        return (val + half) >> shift;
    } else {
        return (val - half) >> shift;
    }
}

template<typename T>
__device__ __forceinline__ int32_t rescale(
    T val,            // 输入整数（通常 int8_t/int16_t/int32_t）
    int32_t M,        // 预计算的定点缩放系数（int32）
    int shift,        // 实际右移位数（>=0 表示右移；<0 表示左移）
    int32_t zp = 0)   // zero point，可选，默认0表示对称量化
{
    int32_t v = static_cast<int32_t>(val);
    int32_t scaled;

    if (shift > 0) {
        // 四舍五入偏移，注意 int32 可能溢出，需要保证 M 不大
        int64_t temp = static_cast<int64_t>(v) * static_cast<int64_t>(M);
        temp += int64_t(1) << (shift - 1);  // 四舍五入
        scaled = static_cast<int32_t>(temp >> shift);
    } else if (shift == 0) {
        scaled = v * M;
    } else { // shift < 0 -> 左移
        scaled = v * M << (-shift);
    }

    // 加上 zero point
    scaled += zp;

    // 可选 saturate 到 int32
    scaled = ::max(::min(scaled, INT32_MAX), INT32_MIN);

    return scaled;
}

template<typename T>
struct QuantLimits;

template<>
struct QuantLimits<int8_t> {
  static __device__ __forceinline__ constexpr int32_t min() { return -128; }

  static __device__ __forceinline__ constexpr int32_t max() { return 127; }
};

template<>
struct QuantLimits<int16_t> {
  static __device__ __forceinline__ constexpr int32_t min() { return -32768; }

  static __device__ __forceinline__ constexpr int32_t max() { return 32767; }
};

// int32_t 特化
template<>
struct QuantLimits<int32_t> {
  static __host__ __device__ constexpr int min() { return -2147483648; }

  static __host__ __device__ constexpr int max() { return 2147483647; }
};

/**
 * @brief 对单个 float 元素执行量化 (设备端函数)
 * @tparam QuantT       目标量化类型（int8_t 或 int16_t）
 * @tparam use_inv_scale 是否使用 inv_scale（乘法而非除法）
 * @tparam symmetric    是否使用对称量化（zero_point=0）
 * @tparam clamp        是否使用饱和处理
 * @param x             输入值（float）
 * @param scale         当前 scale
 * @param zero_point    zero_point（仅非对称量化有效）
 * @return 量化后的整数值 (QuantT)
 */
template<typename QuantT, bool use_inv_scale, bool symmetric, bool clamp = true>
__device__ __forceinline__ QuantT quantizeElement(
    float x,
    float scale,
    int32_t zero_point) {

    // 1. scale 运算
    float scaled;
    if constexpr (use_inv_scale)
        scaled = x * scale;
    else
        scaled = x / scale;

    // 2. zero_point (仅非对称)
    if constexpr (!symmetric)
        scaled += zero_point;

    // 3. 四舍五入
    int32_t rounded = __float2int_rn(scaled);

    // 4. 饱和裁剪
    if constexpr (clamp) {
        constexpr int32_t qmin = QuantLimits<QuantT>::min();
        constexpr int32_t qmax = QuantLimits<QuantT>::max();
        rounded = min(max(rounded, qmin), qmax);
    }

    return static_cast<QuantT>(rounded);
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
__global__ void quantizeFloatToInt(
    const float *src_dev,
    QuantT *dst_dev,
    size_t size,
    float scale,
    int32_t zero_point) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // 调用 device 函数执行单元素量化
    dst_dev[idx] = quantizeElement<QuantT, use_inv_scale, symmetric, clamp>(
        src_dev[idx], scale, zero_point);
}

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
 * @param zero_point    量化 zero_point（非对称量化有效）
 * @param time_step_size 每个时间步的元素数（例如 batch_size * input_dim）
 */
template<typename QuantT, bool use_inv_scale, bool symmetric, bool clamp = true>
__global__ void quantizeFloatToIntPerStep(
    const float *src_dev,
    QuantT *dst_dev,
    size_t size,
    const float *scale_per_t,
    const int32_t *zero_point_per_t,
    int time_step_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // 当前时间步
    int t = idx / time_step_size;
    const float scale = scale_per_t[t];
    const int32_t zero_point = zero_point_per_t[t];

    // 调用 device 函数执行单元素量化
    dst_dev[idx] = quantizeElement<QuantT, use_inv_scale, symmetric, clamp>(
        src_dev[idx], scale, zero_point);
}

} // dev namespace




//template<typename T>
//void calculateScaleZeroPoint(const T *dev_ptr, size_t size, float &scale, int32_t &zero_point) {
//
//    auto max_it = thrust::max_element(dev_ptr, dev_ptr + size);
//    auto min_it = thrust::min_element(dev_ptr, dev_ptr + size);
//
//    T max_val, min_val;
//    thrust::copy(max_it, max_it + 1, &max_val);
//    thrust::copy(min_it, min_it + 1, &min_val);
//
//    constexpr int32_t int_min = std::numeric_limits<T>::min();
//    constexpr int32_t int_max = std::numeric_limits<T>::max();
//
//    // scale计算
//    scale = static_cast<float>(max_val - min_val) / static_cast<float>(int_max - int_min);
//    if (scale == 0.f) scale = 1e-8f;
//
//    // zero-point
//    int32_t zp_temp = static_cast<int32_t>(std::round(-static_cast<float>(min_val) / scale)) + int_min;
//    zero_point = std::clamp(zp_temp, int_min, int_max);
//}
