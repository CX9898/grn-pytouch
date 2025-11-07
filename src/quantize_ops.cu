#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <limits>
#include <algorithm>

#include "quantize_ops_helper.hpp"
#include "quantize_ops.cuh"

__constant__ int8_t d_sigmoid_lut[256];
__constant__ int8_t d_tanh_lut[256];

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

__global__ void computeWeightSumTiled(
    const int8_t *__restrict__ W_q, // [out_dim, in_dim] 权重量化矩阵
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

__global__ void computeWeightSum(
    const int8_t *__restrict__ W_q,   // [out_dim, in_dim] 权重量化矩阵
    int32_t *__restrict__ weight_sum, // [out_dim] 输出数组
    int out_dim,                      // 输出通道数 (M)
    int in_dim                        // 输入通道数 (K)
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;

    int32_t sum = 0;
    const int8_t *row_ptr = W_q + row * in_dim;
#pragma unroll 8
    for (int j = 0; j < in_dim; ++j) {
        sum += static_cast<int32_t>(row_ptr[j]);
    }
    weight_sum[row] = sum;
}

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

void computeWeightSum(
    const int8_t *W_q,// [out_dim, in_dim] 权重量化矩阵
    int32_t *weight_sum,// [out_dim] 输出数组
    int out_dim,// 输出通道数 (M)
    int in_dim,// 输入通道数 (K)
    cudaStream_t stream
) {
    if (in_dim < 4096) {
        int threads = 256;
        int shared_mem = threads * sizeof(int32_t);
        kernel::computeWeightSumTiled<<<out_dim, threads, shared_mem, stream>>>(
            W_q, weight_sum, out_dim, in_dim
        );
    } else {
        int threads = 256;
        int blocks = (out_dim + threads - 1) / threads;
        kernel::computeWeightSum<<<blocks, threads, 0, stream>>>(W_q, weight_sum, out_dim, in_dim);
    }
}

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

inline RescaleParam computeRescaleParamFixedShift(float src_scale, float dst_scale, int fixed_shift) {
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
    bool symmetric) {
    if (host_data == nullptr || size == 0) {
        scale = 1.0f;
        zero_point = 0;
        return;
    }

    // -----------------------------
    // 1. 获取目标类型范围
    // -----------------------------
    const int32_t int_min = static_cast<int32_t>(std::numeric_limits<QuantT>::min());
    const int32_t int_max = static_cast<int32_t>(std::numeric_limits<QuantT>::max());

    // -----------------------------
    // 2. 并行找出最大最小值
    // -----------------------------
    float max_val = host_data[0];
    float min_val = host_data[0];

#pragma omp parallel for reduction(max:max_val) reduction(min:min_val)
    for (size_t i = 1; i < size; ++i) {
        const float v = host_data[i];
        if (v > max_val) max_val = v;
        if (v < min_val) min_val = v;
    }

    // -----------------------------
    // 3. 根据模式计算 scale 和 zero_point
    // -----------------------------
    if (symmetric) {
        // 对称量化：zero_point = 0
        float abs_max = std::max(std::abs(max_val), std::abs(min_val));
        scale = abs_max / static_cast<float>(int_max);
        zero_point = 0;
    } else {
        // 非对称量化：线性映射 [min_val, max_val] → [int_min, int_max]
        scale = (max_val - min_val) / static_cast<float>(int_max - int_min);
        if (scale < 1e-12f) scale = 1e-12f; // 避免除0
        zero_point = static_cast<int32_t>(std::round(int_min - min_val / scale));
        zero_point = std::clamp(zero_point, int_min, int_max);
    }
}

template<typename QuantT>
GruQuantScales computeGruQuantParams(const float *x, int steps, int N, int C,
                                     const float *W, int H,
                                     const float *R,
                                     const float *bx, const float *br) {
    GruQuantScales q;

    // 计算输入 x 的 scale/zp
    q.x_scale.resize(steps);
    q.x_zp.resize(steps);
    for (int i = 0; i < steps; ++i) {
        calculateScaleZeroPoint<QuantT>(x + i * N * C, N * C, q.x_scale[i], q.x_zp[i]);
    }

    // W 分片: [C, H*3] → 每 H 对应一个门
    calculateScaleZeroPoint<QuantT>(W, C * H, q.W.gate[0].scale, q.W.gate[0].zero_point);           // z门
    calculateScaleZeroPoint<QuantT>(W + C * H, C * H, q.W.gate[1].scale, q.W.gate[1].zero_point);   // r门
    calculateScaleZeroPoint<QuantT>(W + 2 * C * H, C * H, q.W.gate[2].scale, q.W.gate[2].zero_point); // h门

    // R 分片: [H, H*3]
    calculateScaleZeroPoint<QuantT>(R, H * H, q.R.gate[0].scale, q.R.gate[0].zero_point);
    calculateScaleZeroPoint<QuantT>(R + H * H, H * H, q.R.gate[1].scale, q.R.gate[1].zero_point);
    calculateScaleZeroPoint<QuantT>(R + 2 * H * H, H * H, q.R.gate[2].scale, q.R.gate[2].zero_point);

    // bias 分片: [H*3]
    calculateScaleZeroPoint<QuantT>(bx, H, q.bx.gate[0].scale, q.bx.gate[0].zero_point);
    calculateScaleZeroPoint<QuantT>(bx + H, H, q.bx.gate[1].scale, q.bx.gate[1].zero_point);
    calculateScaleZeroPoint<QuantT>(bx + 2 * H, H, q.bx.gate[2].scale, q.bx.gate[2].zero_point);

    calculateScaleZeroPoint<QuantT>(br, H, q.br.gate[0].scale, q.br.gate[0].zero_point);
    calculateScaleZeroPoint<QuantT>(br + H, H, q.br.gate[1].scale, q.br.gate[1].zero_point);
    calculateScaleZeroPoint<QuantT>(br + 2 * H, H, q.br.gate[2].scale, q.br.gate[1].zero_point);


    // 计算Wx scale
    q.Wx.resize(steps);
#pragma omp parallel for
    for (int t = 0; t < steps; ++t) {
        for (int g = 0; g < 3; ++g) {
            // --- Wx[t] = x[t] * W ---
            q.Wx[t].gate[g].scale = q.x_scale[t] * q.W.gate[g].scale;
            q.Wx[t].gate[g].zero_point = 0;
        }
    }

    // sigmoid 输出 [0, 1]
    q.z.scale = q.r.scale = 1.0f / 255.0f;
    q.z.zero_point = q.r.zero_point = 0;

    // tanh 输出 [-1, 1]
    q.g.scale = 2.0f / 255.0f;
    q.g.zero_point = 128; // 对称中心为 0

    return q;
}

template GruQuantScales computeGruQuantParams<int8_t>(const float *x, int steps, int N, int C,
                                                      const float *W, int H,
                                                      const float *R,
                                                      const float *bx, const float *br);

template GruQuantScales computeGruQuantParams<int16_t>(const float *x, int steps, int N, int C,
                                                       const float *W, int H,
                                                       const float *R,
                                                       const float *bx, const float *br);

template GruQuantScales computeGruQuantParams<int32_t>(const float *x, int steps, int N, int C,
                                                       const float *W, int H,
                                                       const float *R,
                                                       const float *bx, const float *br);


/**
 * @brief 计算每个时间步的 RescaleParam3 参数（用于 Wx）
 *
 * @param steps           [in] 时间步数（序列长度）
 * @param x_scales        [in] 每个时间步输入 x 的量化 scale, size = steps
 * @param w_scale_z       [in] Wz 的对称量化 scale
 * @param w_scale_r       [in] Wr 的对称量化 scale
 * @param w_scale_g       [in] Wg 的对称量化 scale
 * @param rescale_params  [out] 每步输出的 RescaleParam3 数组, size = steps
 */
void computeWxRescaleParamsFixedShift(
    int steps,
    const std::vector<float> &x_scales,
    const float w_scale_z,
    const float w_scale_r,
    const float w_scale_g,
    std::vector<RescaleParam3> &rescale_params
) {
    rescale_params.resize(steps);
    constexpr int MAX_SHIFT = 31;

    for (int t = 0; t < steps; ++t) {
        // ---------- 计算每步的浮点 scale ----------
        float scale_z = x_scales[t] * w_scale_z;
        float scale_r = x_scales[t] * w_scale_r;
        float scale_g = x_scales[t] * w_scale_g;

        const float scales[3] = {scale_z, scale_r, scale_g};

        RescaleParam3 p{};
        for (int gate = 0; gate < 3; ++gate) {
            float scale_val = scales[gate];

            // ---------- 转换为定点 M 和 shift ----------
            int shift = 0;
            double M_f = 0.0;
            for (shift = 0; shift <= MAX_SHIFT; ++shift) {
                M_f = scale_val * (1LL << shift);
                if (M_f >= 1.0 && M_f < static_cast<double>(1LL << 31))
                    break;
            }

            int32_t M = static_cast<int32_t>(std::round(M_f));
            p.M[gate] = M;
            p.shift[gate] = shift;
        }

        rescale_params[t] = p;
    }
}
