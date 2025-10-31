#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <limits>
#include <algorithm>

#include "quantize_ops.cuh"

void initLut() {
    int8_t h_sigmoid_lut[256];
    int8_t h_tanh_lut[256];

    for (int i = 0; i < 256; i++) {
        int8_t x = i - 128;         // [-128,127]
        float fx = x / 128.0f;      // 转 float [-1,1]
        float s = 1.f / (1.f + expf(-fx));
        float t = tanhf(fx);

        h_sigmoid_lut[i] = static_cast<int8_t>(roundf(s * 127.f));
        h_tanh_lut[i] = static_cast<int8_t>(roundf(t * 127.f));
    }
    cudaMemcpyToSymbol(d_sigmoid_lut, h_sigmoid_lut, sizeof(int8_t) * 256); // 从host端拷贝到device端中编译期固定的地址
    cudaMemcpyToSymbol(d_tanh_lut, h_tanh_lut, sizeof(int8_t) * 256); // 从host端拷贝到device端中编译期固定的地址
}

template<typename T>
void calculateScaleZeroPoint(const T *dev_data, size_t size, float &scale, T &zero_point) {
    const auto max_it = thrust::max_element(dev_data, dev_data + size);
    const auto min_it = thrust::min_element(dev_data, dev_data + size);

    T max_val, min_val;
    thrust::copy(max_it, max_it + 1, &max_val);
    thrust::copy(min_it, min_it + 1, &min_val);

    constexpr T int_min = std::numeric_limits<T>::min();
    constexpr T int_max = std::numeric_limits<T>::max();

    scale = static_cast<float>(max_val - min_val) / static_cast<float>(int_max - int_min);

    // 安全计算zero point
    int zp_temp = static_cast<int>(std::round(-static_cast<float>(min_val) / scale)) + int_min;
    zero_point = static_cast<T>(std::clamp(zp_temp, static_cast<int>(int_min), static_cast<int>(int_max)));
}

template void calculateScaleZeroPoint<int8_t>(const int8_t *dev_data, size_t size, float &scale, int8_t &zero_point);

template void calculateScaleZeroPoint<int16_t>(const int16_t *dev_data, size_t size, float &scale, int16_t &zero_point);


inline RescaleParam computeRescaleParamFixedShift(float src_scale, float dst_scale, int fixed_shift = 15) {
    RescaleParam param;

    if (src_scale <= 0.0f || dst_scale <= 0.0f) {
        // 避免非法 scale
        param.M = 1;
        param.shift = fixed_shift;
        return param;
    }

    // 计算浮点比例
    double scale_ratio = static_cast<double>(src_scale) / static_cast<double>(dst_scale);

    // 根据固定 shift 计算整数 multiplier
    int64_t multiplier_long = static_cast<int64_t>(std::round(scale_ratio * (1LL << fixed_shift)));

    // 防止 int32 溢出
    if (multiplier_long > std::numeric_limits<int32_t>::max()) multiplier_long = std::numeric_limits<int32_t>::max();
    if (multiplier_long < std::numeric_limits<int32_t>::min()) multiplier_long = std::numeric_limits<int32_t>::min();

    param.M = static_cast<int32_t>(multiplier_long);
    param.shift = fixed_shift;

    return param;
}

