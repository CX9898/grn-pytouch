#pragma once

#include <cuda_runtime.h>
#include <cstdint>

constexpr int32_t shift_Rh = 24;
constexpr int32_t shift_br = 24;

__constant__ int8_t d_sigmoid_lut[256]; // 全局常量
__constant__ int8_t d_tanh_lut[256]; // 全局常量

void initLut();

namespace dev {

template<bool use_inv_scale>
__device__ __forceinline__ int8_t quantize_float_to_int8(
    const float value,
    const float scale_param,
    const int32_t zero_point
) {
    // 编译期分支：根据use_inv_scale选择计算方式(无运行时开销)
    const float scaled = [value, scale_param]() {
      if constexpr (use_inv_scale) {
          // 分支1：用inv_scale，乘法(编译期确定，仅当use_inv_scale=true时保留)
          return value * scale_param;
      } else {
          // 分支2：用scale，除法(编译期确定，仅当use_inv_scale=false时保留)
          return value / scale_param;
      }
    }();

    const float shifted = scaled + static_cast<float>(zero_point);
    const int32_t rounded = __float2int_rn(shifted); // 四舍五入
    const int32_t clamped = ::max(-128, ::min(127, rounded)); // 范围截断
    return static_cast<int8_t>(clamped);
}

__device__ __forceinline__ int16_t quantize_i32_to_i8(
    const int32_t value,
    const int32_t M,
    const int32_t shift,
    const int32_t zero_point = 0) {
    int32_t tmp = (value * M + (1 << (shift - 1))) >> shift;
    tmp += zero_point;
    tmp = max(-128, min(127, tmp));
    return static_cast<int16_t>(tmp);
}

__device__ __forceinline__ int16_t quantize_i32_to_i16(
    const int32_t value,
    const int32_t M,
    const int32_t shift,
    const int32_t zero_point = 0) {
    int32_t tmp = (value * M + (1 << (shift - 1))) >> shift;
    tmp += zero_point;
    tmp = max(-32768, min(32767, tmp));
    return static_cast<int16_t>(tmp);
}

__device__ __forceinline__ int8_t sigmoid_int8_lut(int8_t x) {
    // x in [-128,127], lut 长度 = 256
    const int8_t x_clamped = max(-128, min(127, x));
    return d_sigmoid_lut[static_cast<uint8_t>(x_clamped)]; // uint8_t 转索引 [0,255]
}

__device__ __forceinline__ int8_t tanh_int8_lut(int8_t x) {
    const int8_t x_clamped = max(-128, min(127, x));
    return d_tanh_lut[static_cast<uint8_t>(x_clamped)];
}

__device__ __forceinline__ int8_t sigmoid_int16_lut(int16_t x) { // (TODO: 二项式拟合查表方式)
    // 将 int16_t 范围 [-32768, 32767] 映射到 int8_t 范围 [-128, 127]
    // 公式：idx = round( (x + 32768) * (255.0f / 65535.0f) ) - 128
    // 整数优化：避免浮点运算，用移位实现近似缩放
    int32_t tmp = static_cast<int32_t>(x) + 32768; // 转为 [0, 65535]
    tmp = (tmp * 255 + 65535 / 2) / 65535; // 四舍五入缩放到 [0, 255]
    int8_t idx = static_cast<int8_t>(tmp - 128); // 转为 [-128, 127]
    return d_sigmoid_lut[static_cast<uint8_t>(idx)];
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

} // dev namespace

template<typename T>
void calculateScaleZeroPoint(const T *dev_data, size_t size, float &scale, T &zero_point);

// Rescale 参数结构体（含义清楚）
struct RescaleParam {
  int32_t M;  // M, 整数乘法系数，对齐到目标 scale 使用
  int shift;         // shift, 右移位数，用于 CUDA kernel
};

/**
 * @param src_scale     源张量的量化 scale (float)
 * @param dst_scale     目标张量的量化 scale (float)
 * @param fixed_shift   固定的右移位数，用于 kernel 右移，默认 15
 * @return RescaleParam 包含整数 multiplier 和 kernel 右移 shift
 */
inline RescaleParam computeRescaleParamFixedShift(float src_scale, float dst_scale, int fixed_shift = 15);