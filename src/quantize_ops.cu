#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <limits>
#include <algorithm>

#include "quantize_ops_helper.hpp"
#include "quantize_ops.cuh"

__constant__ int8_t d_sigmoid_int8_z_lut[256];
__constant__ int8_t d_sigmoid_int8_r_lut[256];
__constant__ int8_t d_tanh_int8_g_lut[256];

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
template<typename QuantT, bool use_inv_scale, bool symmetric, bool clamp>
void quantizeFloatToInt(const float *src_dev,
                        QuantT *dst_dev,
                        uint32_t size,
                        float scale,
                        int32_t zero_point) {
    uint32_t block = 512;
    uint32_t grid = (size + block - 1) / block;

    dev::quantizeFloatToInt<QuantT, use_inv_scale, symmetric, clamp>
    <<<grid, block>>>(src_dev, dst_dev, size, scale, zero_point);
}

template void quantizeFloatToInt<int8_t, true, true, true>(const float *src_dev,
                                                           int8_t *dst_dev,
                                                           uint32_t size,
                                                           float scale,
                                                           int32_t zero_point);

template void quantizeFloatToInt<int8_t, true, true, false>(const float *src_dev,
                                                            int8_t *dst_dev,
                                                            uint32_t size,
                                                            float scale,
                                                            int32_t zero_point);

template void quantizeFloatToInt<int32_t, true, true, false>(const float *src_dev,
                                                             int32_t *dst_dev,
                                                             uint32_t size,
                                                             float scale,
                                                             int32_t zero_point);

template void quantizeFloatToInt<int32_t, false, false, true>(const float *src_dev,
                                                              int32_t *dst_dev,
                                                              uint32_t size,
                                                              float scale,
                                                              int32_t zero_point);


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
 * @param zero_point_per_t    每个时间步的量化 zero_point（非对称量化有效）
 * @param time_step_size 每个时间步的元素数（例如 batch_size * input_dim）
 */
template<typename QuantT, bool use_inv_scale, bool symmetric, bool clamp>
void quantizeFloatToIntPerStep(const float *src_dev,
                               QuantT *dst_dev,
                               size_t size,
                               const float *scale_per_t,
                               const int32_t *zero_point_per_t,
                               int time_step_size) {
    uint32_t block = 512;
    uint32_t grid = (size + block - 1) / block;

    dev::quantizeFloatToIntPerStep<QuantT, use_inv_scale, symmetric, clamp>
    <<<grid, block>>>(src_dev, dst_dev, size, scale_per_t, zero_point_per_t, time_step_size);
}

template void quantizeFloatToIntPerStep<int8_t, false, false, true>(const float *src_dev,
                                                                    int8_t *dst_dev,
                                                                    size_t size,
                                                                    const float *scale_per_t,
                                                                    const int32_t *zero_point_per_t,
                                                                    int time_step_size);

template void quantizeFloatToIntPerStep<int16_t, false, false, true>(const float *src_dev,
                                                                     int16_t *dst_dev,
                                                                     size_t size,
                                                                     const float *scale_per_t,
                                                                     const int32_t *zero_point_per_t,
                                                                     int time_step_size);

std::vector<int8_t> generate_sigmoid_int8_lut(float scale_z_pre, int zp_z_pre,
                                              float scale_z, int zp_z) {
    std::vector<int8_t> lut(256);

    for (int i = 0; i < 256; i++) {
        int x_i8 = i - 128;

        const float x_fp = static_cast<float>(x_i8 - zp_z_pre) * scale_z_pre;
        const float y_fp = 1.f / (1.f + std::exp(-x_fp));

        int y_i8 = static_cast<int>(std::round(y_fp / scale_z + zp_z));
        if (y_i8 < -128) y_i8 = -128;
        if (y_i8 > 127) y_i8 = 127;

        lut[i] = static_cast<int8_t>(y_i8);
    }
    return lut;

}

std::vector<int8_t> generate_tanh_int8_lut(float scale_pre, int zp_pre,
                                           float scale_out, int zp_out) {
    std::vector<int8_t> lut(256);

    for (int i = 0; i < 256; i++) {
        int x_i8 = i - 128;

        float x_fp = (x_i8 - zp_pre) * scale_pre;
        float y_fp = std::tanh(x_fp);

        int y_i8 = static_cast<int>(std::round(y_fp / scale_out + zp_out));
        if (y_i8 < -128) y_i8 = -128;
        if (y_i8 > 127) y_i8 = 127;

        lut[i] = static_cast<int8_t>(y_i8);
    }
    return lut;

}

void generate_int8_lut(float scale_z_pre, int32_t zp_z_pre, float scale_z_out, int32_t zp_z_out,
                       float scale_r_pre, int32_t zp_r_pre, float scale_r_out, int32_t zp_r_out,
                       float scale_g_pre, int32_t zp_g_pre, float scale_g_out, int32_t zp_g_out) {
    std::vector<int8_t> sigmoid_z_lut = generate_sigmoid_int8_lut(scale_z_pre, zp_z_pre, scale_z_out, zp_z_out);
    std::vector<int8_t> sigmoid_r_lut = generate_sigmoid_int8_lut(scale_r_pre, zp_r_pre, scale_r_out, zp_r_out);
    std::vector<int8_t> tanh_int8_lut = generate_tanh_int8_lut(scale_g_pre, zp_g_pre, scale_g_out, zp_g_out);

    cudaMemcpyToSymbol(d_sigmoid_int8_z_lut, sigmoid_z_lut.data(), sizeof(int8_t) * 256); // 从host端拷贝到device端中编译期固定的地址
    cudaMemcpyToSymbol(d_sigmoid_int8_r_lut, sigmoid_r_lut.data(), sizeof(int8_t) * 256); // 从host端拷贝到device端中编译期固定的地址
    cudaMemcpyToSymbol(d_tanh_int8_g_lut, tanh_int8_lut.data(), sizeof(int8_t) * 256); // 从host端拷贝到device端中编译期固定的地址
}


//template<typename T>
//void calculateScaleZeroPoint(const T *host_data, size_t size, float &scale, T &zero_point) {
//    const auto max_it = thrust::max_element(thrust::host, host_data, host_data + size);
//    const auto min_it = thrust::min_element(thrust::host, host_data, host_data + size);
//
//    T max_val, min_val;
//    thrust::copy(max_it, max_it + 1, &max_val);
//    thrust::copy(min_it, min_it + 1, &min_val);
//
//    constexpr int int_min = std::numeric_limits<T>::min();
//    constexpr int int_max = std::numeric_limits<T>::max();
//
//    scale = static_cast<float>(max_val - min_val) / static_cast<float>(int_max - int_min);
//
//    // 安全计算zero point
//    int zp_temp = static_cast<int>(std::round(-static_cast<float>(min_val) / scale)) + int_min;
//    zero_point = static_cast<T>(std::clamp(zp_temp, static_cast<int>(int_min), static_cast<int>(int_max)));
//}
//
//template void calculateScaleZeroPoint<int8_t>(const int8_t *dev_data, size_t size, float &scale, int8_t &zero_point);
//
//template void calculateScaleZeroPoint<int16_t>(const int16_t *dev_data, size_t size, float &scale, int16_t &zero_point);

namespace kernel {

template<typename T>
__global__ void computeWeightSumTiled(
    const T *__restrict__ W_q, // [out_dim, in_dim] 权重量化矩阵
    int32_t *__restrict__ weight_sum, // [out_dim] 输出数组
    int out_dim, // 输出通道数 (M)
    int in_dim // 输入通道数 (K)
) {
    const int row = blockIdx.x;
    if (row >= out_dim) {
        return;
    }
    const int tid = threadIdx.x;

    extern __shared__ int32_t sdata[];
    int32_t local_sum = 0;

    // 每个线程处理部分列
    for (int j = tid; j < in_dim; j += blockDim.x) {
        local_sum += static_cast<int32_t>(W_q[row * in_dim + j]);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // 归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        weight_sum[row] = sdata[0];
    }
}

template<typename T>
__global__ void computeWeightSumMulzp(
    const T *__restrict__ W_q,   // [out_dim, in_dim] 权重量化矩阵
    int32_t *__restrict__ weight_sum, // [out_dim] 输出数组
    int x_zp,
    int32_t n, //
    int out_dim,                      // 输出通道数 (M)
    int in_dim                        // 输入通道数 (K)
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;

    int32_t sum = 0;
    const T *row_ptr = W_q + row * in_dim;
#pragma unroll 8
    for (int j = 0; j < in_dim; ++j) {
        sum += static_cast<int32_t>(row_ptr[j]);
    }
    weight_sum[row] = rshift_round(sum * x_zp, n);
}

template __global__ void computeWeightSumMulzp<int8_t>(
    const int8_t *__restrict__ W_q,   // [out_dim, in_dim] 权重量化矩阵
    int32_t *__restrict__ weight_sum, // [out_dim] 输出数组
    int x_zp,
    int32_t n, //
    int out_dim,                      // 输出通道数 (M)
    int in_dim                        // 输入通道数 (K)
);

template __global__ void computeWeightSumMulzp<int16_t>(
    const int16_t *__restrict__ W_q,   // [out_dim, in_dim] 权重量化矩阵
    int32_t *__restrict__ weight_sum, // [out_dim] 输出数组
    int x_zp,
    int32_t n, //
    int out_dim,                      // 输出通道数 (M)
    int in_dim                        // 输入通道数 (K)
);

__global__ void applyZeroPointCompensation2D(
    int32_t *__restrict__ Y_int32,
    const int32_t *__restrict__ weight_sum,
    const int32_t *__restrict__ x_zp,
    int out_dim,
    int batch_size
) {
    int m = blockIdx.y * blockDim.y + threadIdx.y; // 输出维度方向
    int b = blockIdx.x * blockDim.x + threadIdx.x; // batch方向

    if (m >= out_dim || b >= batch_size) return;

    int idx = m * batch_size + b;
    Y_int32[idx] -= x_zp[b] * weight_sum[m];
}

} // kernel namespace

template<typename T>
void computeWeightSumMulzp(
    const T *W_q,// [out_dim, in_dim] 权重量化矩阵
    int32_t *weight_sum,// [out_dim] 输出数组
    int x_zp,
    int32_t n, //
    int out_dim,// 输出通道数 (M)
    int in_dim,// 输入通道数 (K)
    cudaStream_t stream
) {
//    if (in_dim < 4096) {
//        int threads = 256;
//        int shared_mem = threads * sizeof(int32_t);
//        kernel::computeWeightSumTiled<<<out_dim, threads, shared_mem, stream>>>(
//            W_q, weight_sum, out_dim, in_dim
//        );
//    } else {
    int threads = 256;
    int blocks = (out_dim + threads - 1) / threads;
    kernel::computeWeightSumMulzp<<<blocks, threads, 0, stream>>>(W_q, weight_sum, x_zp, n, out_dim, in_dim);
//    }
}

template void computeWeightSumMulzp<int8_t>(
    const int8_t *W_q,// [out_dim, in_dim] 权重量化矩阵
    int32_t *weight_sum,// [out_dim] 输出数组
    int x_zp,
    int32_t n, //
    int out_dim,// 输出通道数 (M)
    int in_dim,// 输入通道数 (K)
    cudaStream_t stream
);

template void computeWeightSumMulzp<int16_t>(
    const int16_t *W_q,// [out_dim, in_dim] 权重量化矩阵
    int32_t *weight_sum,// [out_dim] 输出数组
    int x_zp,
    int32_t n, //
    int out_dim,// 输出通道数 (M)
    int in_dim,// 输入通道数 (K)
    cudaStream_t stream
);

void applyZeroPointCompensation2D(
    int32_t *Y_int32,
    const int32_t *weight_sum,
    const int32_t *x_zp,
    int out_dim,
    int batch_size,
    cudaStream_t stream
) {
    dim3 threads(16, 16);
    dim3 blocks((batch_size + 15) / 16, (out_dim + 15) / 16);
    kernel::applyZeroPointCompensation2D<<<blocks, threads, 0, stream>>>(
        Y_int32, weight_sum, x_zp, out_dim, batch_size
    );
}

/**
 * @brief 从 GPU 上的量化数据计算 scale（使用最大最小值）
 */
template<typename QuantT>
void calculateScaleZeroPointFromDevice(
    const QuantT *h_dev,
    size_t size,
    float &scale,
    int32_t &zero_point,
    bool symmetric,
    cudaStream_t stream) {
    if (h_dev == nullptr || size == 0) {
        scale = 1.0f;
        zero_point = 0;
        return;
    }

    // 使用 thrust 或自定义 kernel 计算最大最小值
    // 这里使用简单的 CPU 方法（需要将数据拷贝到 host）
    std::vector<QuantT> h_host(size);
    cudaMemcpyAsync(h_host.data(), h_dev, size * sizeof(QuantT), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 计算最大最小值
    QuantT max_val = h_host[0];
    QuantT min_val = h_host[0];
    for (size_t i = 1; i < size; ++i) {
        if (h_host[i] > max_val) max_val = h_host[i];
        if (h_host[i] < min_val) min_val = h_host[i];
    }

    // 计算 scale 和 zero_point
    const int32_t int_min = static_cast<int32_t>(std::numeric_limits<QuantT>::min());
    const int32_t int_max = static_cast<int32_t>(std::numeric_limits<QuantT>::max());

    if (symmetric) {
        float abs_max = std::max(std::abs(static_cast<float>(max_val)), std::abs(static_cast<float>(min_val)));
        scale = abs_max / static_cast<float>(int_max);
        zero_point = 0;
    } else {
        scale = static_cast<float>(max_val - min_val) / static_cast<float>(int_max - int_min);
        if (scale < 1e-12f) scale = 1e-12f;
        zero_point = static_cast<int32_t>(std::round(int_min - static_cast<float>(min_val) / scale));
        zero_point = std::clamp(zero_point, int_min, int_max);
    }
}

template void calculateScaleZeroPointFromDevice<int8_t>(
    const int8_t *h_dev, size_t size, float &scale, int32_t &zero_point, bool symmetric, cudaStream_t stream);

template void calculateScaleZeroPointFromDevice<int16_t>(
    const int16_t *h_dev, size_t size, float &scale, int32_t &zero_point, bool symmetric, cudaStream_t stream);
//
//template<typename T>
//T findMaxValueFromDev(const T *dev_data, size_t size) {
//    if (size == 0) {
//        // 边界处理：空数据返回最小值（避免访问非法内存）
//        return std::numeric_limits<T>::lowest();
//    }
//
//    // 直接用 thrust::max_element 找设备端最大值的迭代器
//    const T *max_it = thrust::max_element(thrust::device, dev_data, dev_data + size);
//
//    T max_val;
//    // 把设备端的最大值拷贝到主机端
//    thrust::copy(thrust::device, max_it, max_it + 1, &max_val);
//
//    return max_val;
//
////    return thrust::reduce(thrust::device, dev_data, dev_data + size,
////                          std::numeric_limits<T>::lowest(),
////                          thrust::maximum<T>());
//}
//
//template int8_t findMaxValueFromDev<int8_t>(const int8_t *dev_data, size_t size);
//
//template int16_t findMaxValueFromDev<int16_t>(const int16_t *dev_data, size_t size);
//
//template float findMaxValueFromDev<float>(const float *dev_data, size_t size);
//
//template<typename T>
//T findMinValueFromDev(const T *dev_data, size_t size) {
//    if (size == 0) {
//        // 边界处理：空数据返回最大值（避免访问非法内存）
//        return std::numeric_limits<T>::max();
//    }
//
//    // 直接用 thrust::max_element 找设备端最大值的迭代器
//    const T *min_it = thrust::min_element(thrust::device, dev_data, dev_data + size);
//
//    T min_val;
//    // 把设备端的最大值拷贝到主机端
//    thrust::copy(thrust::device, min_it, min_it + 1, &min_val);
//
//    return min_val;
//
////    return thrust::reduce(thrust::device, dev_data, dev_data + size,
////                          std::numeric_limits<T>::lowest(),
////                          thrust::maximum<T>());
//}
//
//template int8_t findMinValueFromDev<int8_t>(const int8_t *dev_data, size_t size);
//
//template int16_t findMinValueFromDev<int16_t>(const int16_t *dev_data, size_t size);
//
//template float findMinValueFromDev<float>(const float *dev_data, size_t size);
