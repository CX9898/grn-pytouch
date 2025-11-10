#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
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

inline ScaleParam computeRescaleParamFixedShift(float src_scale, float dst_scale, int fixed_shift) {
    ScaleParam param;

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
    bool symmetric) {

    QuantParams params{};

    if (data == nullptr || size == 0) {
        params.scale = 1.0f;
        params.zero_point = 0;
        return params;
    }

    // -----------------------------
    // 1. 获取目标类型范围
    // -----------------------------
    const int32_t int_min = static_cast<int32_t>(std::numeric_limits<QuantT>::min());
    const int32_t int_max = static_cast<int32_t>(std::numeric_limits<QuantT>::max());

    // -----------------------------
    // 2. 并行找出最大最小值
    // -----------------------------
    float max_val = data[0];
    float min_val = data[0];

#pragma omp parallel for reduction(max:max_val) reduction(min:min_val)
    for (size_t i = 1; i < size; ++i) {
        const float v = data[i];
        if (v > max_val) max_val = v;
        if (v < min_val) min_val = v;
    }

    // -----------------------------
    // 3. 根据模式计算 scale 和 zero_point
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
}

template QuantParams calculateQuantParams<int8_t>(
    const float *data,
    size_t size,
    bool symmetric);

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
        QuantParams x_quantParams = calculateQuantParams<QuantT>(x + i * N * C, N * C, false);
        q.x_scale[i] = x_quantParams.scale;
        q.x_zp[i] = x_quantParams.zero_point;
    }

    for (int gate_idx = 0; gate_idx < 3; ++gate_idx) { // z, r, g
        // W 分片: [C, H*3] → 每 H 对应一个门
        q.W.gate[gate_idx] = calculateQuantParams<QuantT>(W + gate_idx * C * H, C * H);

        // R 分片: [H, H*3]
        q.R.gate[gate_idx] = calculateQuantParams<QuantT>(R + gate_idx * H * H, H * H);

        // bias 分片: [H*3]
        q.bx.gate[gate_idx] = calculateQuantParams<QuantT>(bx + gate_idx * H, H);
        q.br.gate[gate_idx] = calculateQuantParams<QuantT>(br + gate_idx * H, H);
    }

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

    // sigmoid 输出 [0, 1]，LUT 映射到 [0, 127]
    // 所以 scale 应该是 1.0f / 127.0f，但为了与 int8 范围一致，使用 1.0f / 255.0f
    // 注意：LUT 输出范围是 [0, 127]，但 scale 使用 1.0f / 255.0f 是为了保持一致性
    q.z.scale = q.r.scale = 1.0f / 127.0f; // 修正：sigmoid LUT 输出范围是 [0, 127]
    q.z.zero_point = q.r.zero_point = 0;

    // tanh 输出 [-1, 1]，LUT 映射到 [-128, 127]
    // 所以 scale 应该是 2.0f / 255.0f
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
 * @brief 计算每个时间步的 ScaleParam3 参数（用于 Wx）
 *
 * @param steps           [in] 时间步数（序列长度）
 * @param x_scales        [in] 每个时间步输入 x 的量化 scale, size = steps
 * @param w_scale_z       [in] Wz 的对称量化 scale
 * @param w_scale_r       [in] Wr 的对称量化 scale
 * @param w_scale_g       [in] Wg 的对称量化 scale
 * @param rescale_params  [out] 每步输出的 ScaleParam3 数组, size = steps
 */
void computeWxRescaleParamsFixedShift(
    int steps,
    const std::vector<float> &x_scales,
    const float w_scale_z,
    const float w_scale_r,
    const float w_scale_g,
    std::vector<ScaleParam3> &rescale_params
) {
    rescale_params.resize(steps);
    constexpr int MAX_SHIFT = 31;

    for (int t = 0; t < steps; ++t) {
        // ---------- 计算每步的浮点 scale ----------
        float scale_z = x_scales[t] * w_scale_z;
        float scale_r = x_scales[t] * w_scale_r;
        float scale_g = x_scales[t] * w_scale_g;

        const float scales[3] = {scale_z, scale_r, scale_g};

        ScaleParam3 p{};
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

/**
 * @brief 计算每个时间步的所有 RescaleParamsPerStep 参数
 */
void computeGruRescaleParamsPerStep(
    int steps,
    GruQuantScales &gruQuantScales,
    std::vector<RescaleParamsPerStep> &rescale_params,
    int fixed_shift) {
    rescale_params.resize(steps);

    gruQuantScales.h_scale.resize(steps + 1);

    // h 的 scale 将在运行时动态计算（使用指数移动平均：90% 上一步 + 10% 当前步）
    // 初始化 rescale 参数数组的大小（实际参数将在运行时动态更新）
    rescale_params.resize(steps);

    for (int t = 0; t < steps; ++t) {
        RescaleParamsPerStep &params = rescale_params[t];

        // 计算 Wx 的 scale（每步每个门）
        const float wx_scale_z = gruQuantScales.x_scale[t] * gruQuantScales.W.gate[0].scale;
        const float wx_scale_r = gruQuantScales.x_scale[t] * gruQuantScales.W.gate[1].scale;
        const float wx_scale_g = gruQuantScales.x_scale[t] * gruQuantScales.W.gate[2].scale;

        gruQuantScales.Wx[t].gate[0].scale = wx_scale_z;
        gruQuantScales.Wx[t].gate[1].scale = wx_scale_r;
        gruQuantScales.Wx[t].gate[2].scale = wx_scale_g;

        // 计算h初始时间步的scale
//        if (t == 0) {
//            // 初始化 h_scale（使用 g 的 scale 作为初始值）
//            const float h_scale_original = gruQuantScales.g.scale;
//            gruQuantScales.h_scale[0] = h_scale_original;
//            computeRhScale(t, gruQuantScales, params, fixed_shift);
//        }

        // 计算 bx_to_Wx 的 rescale 参数（每个门）
        params.bx_to_Wx.M[0] = computeRescaleParamFixedShift(gruQuantScales.bx.gate[0].scale,
                                                             wx_scale_z,
                                                             fixed_shift).M;
        params.bx_to_Wx.shift[0] = fixed_shift;
        params.bx_to_Wx.M[1] = computeRescaleParamFixedShift(gruQuantScales.bx.gate[1].scale,
                                                             wx_scale_r,
                                                             fixed_shift).M;
        params.bx_to_Wx.shift[1] = fixed_shift;
        params.bx_to_Wx.M[2] = computeRescaleParamFixedShift(gruQuantScales.bx.gate[2].scale,
                                                             wx_scale_g,
                                                             fixed_shift).M;
        params.bx_to_Wx.shift[2] = fixed_shift;

        // 计算 br_to_Wx 的 rescale 参数（每个门）
        params.br_to_Wx.M[0] = computeRescaleParamFixedShift(gruQuantScales.br.gate[0].scale,
                                                             wx_scale_z,
                                                             fixed_shift).M;
        params.br_to_Wx.shift[0] = fixed_shift;
        params.br_to_Wx.M[1] = computeRescaleParamFixedShift(gruQuantScales.br.gate[1].scale,
                                                             wx_scale_r,
                                                             fixed_shift).M;
        params.br_to_Wx.shift[1] = fixed_shift;
        params.br_to_Wx.M[2] = computeRescaleParamFixedShift(gruQuantScales.br.gate[2].scale,
                                                             wx_scale_g,
                                                             fixed_shift).M;
        params.br_to_Wx.shift[2] = fixed_shift;

        // 计算 Wx_to_out 的 rescale 参数（每个门）
        // 激活函数输入的 scale：LUT 输入是 int8，范围 [-128, 127]
        // sigmoid/tanh LUT 的输入 scale 应该与 LUT 的设计一致
        // 从 initLut() 可以看到：x / 128.0f 映射到 [-1, 1]，所以输入 scale 是 1.0f / 128.0f
        // 但是，由于我们需要将 int32 累加结果量化回 int8，我们需要知道激活函数输入的 scale
        // 实际上，激活函数输入的 scale 应该与激活函数输出的 scale 相关
        // 对于 sigmoid：输入 scale 可以设置为一个合理的值，例如 1.0f / 128.0f（对应 LUT 输入范围）
        // 对于 tanh：同样使用 1.0f / 128.0f
        // 但是，为了简化，我们可以使用一个统一的 scale，例如 1.0f / 128.0f
        float activation_input_scale = 1.0f / 128.0f; // LUT 输入 scale：int8 [-128,127] 映射到 [-1,1]
        params.Wx_to_out.M[0] = computeRescaleParamFixedShift(wx_scale_z, activation_input_scale, fixed_shift).M;
        params.Wx_to_out.shift[0] = fixed_shift;
        params.Wx_to_out.M[1] = computeRescaleParamFixedShift(wx_scale_r, activation_input_scale, fixed_shift).M;
        params.Wx_to_out.shift[1] = fixed_shift;
        params.Wx_to_out.M[2] = computeRescaleParamFixedShift(wx_scale_g, activation_input_scale, fixed_shift).M;
        params.Wx_to_out.shift[2] = fixed_shift;

        // 计算 r_to_Wx_g 的 rescale 参数
        // 用于计算 r * (Rh_aligned_g + br_aligned_g)
        // r 输出的 scale 是 S_r = 1.0f / 255.0f（sigmoid 输出）
        // (Rh_aligned_g + br_aligned_g) 的 scale 是 S_Wx_g（已经对齐到 Wx_g）
        // r * (Rh_aligned_g + br_aligned_g) 的 scale 是 S_r * S_Wx_g
        // 需要 rescale 到 S_Wx_g，所以需要除以 S_r，即乘以 1/S_r = 255.0f
        // 但是，由于我们需要将 r 的 scale 对齐到 Wx_g，所以参数应该是 (S_r / S_Wx_g) * (1 << shift)
        // 但这样计算出来的结果 scale 是 S_r^2，不正确
        // 正确的做法是：r_to_Wx_g 应该表示将 (r * value) 的结果 rescale 到 Wx_g
        // 即：如果 value 的 scale 是 S_Wx_g，那么 r * value 的 scale 是 S_r * S_Wx_g
        // 需要 rescale 到 S_Wx_g，所以需要乘以 1/S_r = 255.0f
        // 因此，r_to_Wx_g 的参数应该是 (1.0f / S_r) / S_Wx_g * (1 << shift) = 255.0f / S_Wx_g * (1 << shift)
        float r_output_scale = gruQuantScales.r.scale; // 1.0f / 255.0f
        // 计算将 (r * value_with_scale_Wx_g) 的结果 rescale 到 Wx_g 的参数
        // 即：将 scale (S_r * S_Wx_g) rescale 到 S_Wx_g，需要乘以 1/S_r
        float scale_ratio = 1.0f / r_output_scale; // 255.0f
        double multiplier = static_cast<double>(scale_ratio) * (1LL << fixed_shift);
        int64_t multiplier_long = static_cast<int64_t>(std::round(multiplier));
        if (multiplier_long > std::numeric_limits<int32_t>::max())
            multiplier_long = std::numeric_limits<int32_t>::max();
        if (multiplier_long < std::numeric_limits<int32_t>::min())
            multiplier_long = std::numeric_limits<int32_t>::min();
        params.r_to_Wx_g.M = static_cast<int32_t>(multiplier_long);
        params.r_to_Wx_g.shift = fixed_shift;
    }

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

template<typename T>
T findMaxValueFromDev(const T *dev_data, size_t size) {
//    const auto max_it = thrust::max_element(thrust::device, dev_data, dev_data + size);
//
//    T max_val;
//    thrust::copy(max_it, max_it + 1, &max_val);
//
//    return max_val;
//    thrust::device_ptr<const T> dev_ptr(dev_data);
    return thrust::reduce(thrust::device, dev_data, dev_data + size,
                          std::numeric_limits<T>::lowest(),
                          thrust::maximum<T>());

}

template int8_t findMaxValueFromDev<int8_t>(const int8_t *dev_data, size_t size);

template int16_t findMaxValueFromDev<int16_t>(const int16_t *dev_data, size_t size);
