#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <type_traits>

#include "blas.h"
#include "gru_quant.h"
#include "inline_ops.h"
#include "device_ptr.h"
#include "quantize_ops.cuh"
#include "quantize_ops_helper.hpp"
#include "devVector.h"

namespace kernel {

__device__ __forceinline__ int8_t computeZ(
    const int hidden_idx,
    const int32_t Wx_val,   // Wx 对应门的值
    const int32_t Rh_val,   // Rh 对应门的值
    const int32_t bx_val,   // bx 对应门的bias
    const int32_t br_val,   // br 对应门的bias
    const QuantGRUReScale &rescale_params
) {
    // 1. 各项对齐到 Wx 的 scale
    const int32_t Rh_aligned = dev::rescale(Rh_val + br_val,
                                            rescale_params.Rh_z_to_Wx_z[hidden_idx].M,
                                            rescale_params.Rh_z_to_Wx_z[hidden_idx].shift);
//    const int32_t bx_aligned = dev::rescale(bx_val,
//                                            rescaleParams.bx_to_Wx.M[gate_idx],
//                                            rescaleParams.bx_to_Wx.shift[gate_idx]);
//    const int32_t br_aligned = dev::rescale(br_val,
//                                            rescale_params.Rh_to_Wx[hidden_idx].M[gate_idx],
//                                            rescale_params.Rh_to_Wx[hidden_idx].shift[gate_idx]);

    // 累加求和
    const int32_t tmp_i32 = Wx_val + Rh_aligned + bx_val;

    // 量化回 int8
    const int8_t tmp_i8 = dev::quantize_i32_to_i8(tmp_i32,
                                                  rescale_params.Wx_to_out[hidden_idx].M[0],
                                                  rescale_params.Wx_to_out[hidden_idx].shift[0]);

    return tmp_i8;
}

__device__ __forceinline__ int8_t computeR(
    const int hidden_idx,
    const int32_t Wx_val,   // Wx 对应门的值
    const int32_t Rh_val,   // Rh 对应门的值
    const int32_t bx_val,   // bx 对应门的bias
    const int32_t br_val,   // br 对应门的bias
    const QuantGRUReScale &rescale_params
) {
    // 1. 各项对齐到 Wx 的 scale
    const int32_t Rh_aligned = dev::rescale(Rh_val + br_val,
                                            rescale_params.Rh_z_to_Wx_z[hidden_idx].M,
                                            rescale_params.Rh_z_to_Wx_z[hidden_idx].shift);
//    const int32_t bx_aligned = dev::rescale(bx_val,
//                                            rescaleParams.bx_to_Wx.M[gate_idx],
//                                            rescaleParams.bx_to_Wx.shift[gate_idx]);
//    const int32_t br_aligned = dev::rescale(br_val,
//                                            rescale_params.Rh_to_Wx[hidden_idx].M[gate_idx],
//                                            rescale_params.Rh_to_Wx[hidden_idx].shift[gate_idx]);

    // 累加求和
    const int32_t tmp_i32 = Wx_val + Rh_aligned + bx_val;

    // 量化回 int8
    const int8_t tmp_i8 = dev::quantize_i32_to_i8(tmp_i32,
                                                  rescale_params.Wx_to_out[hidden_idx].M[1],
                                                  rescale_params.Wx_to_out[hidden_idx].shift[1]);
    return tmp_i8;
}


__device__ inline int32_t mul_and_rescale(int8_t a, int32_t b, int32_t M, int shift) {
    return (int32_t) ((static_cast<int64_t>(a) * b * M) >> shift);
}

__device__ __forceinline__ int8_t computeG(
    const int hidden_idx,
    const int32_t Wx_val,   // Wx 对应门的值
    const int32_t Rh_val,   // Rh 对应门的值
    const int32_t bx_val,   // bx 对应门的bias
    const int32_t br_val,   // br 对应门的bias
    const int32_t r,
    const QuantGRUReScale &rescale_params
) {
    const int64_t rRh = static_cast<int64_t>(r) * static_cast<int64_t>(Rh_val + br_val);
    const int32_t rRh_aligned = dev::rescale(rRh,
                                             rescale_params.rRh_g_to_Wx_g[hidden_idx].M,
                                             rescale_params.rRh_g_to_Wx_g[hidden_idx].shift);

    // 累加求和
    const int32_t tmp_i32 = Wx_val + rRh_aligned + bx_val;

    // 量化回 int8
    const int8_t tmp_i8 = dev::quantize_i32_to_i8(tmp_i32,
                                                  rescale_params.Wx_to_out[hidden_idx].M[2],
                                                  rescale_params.Wx_to_out[hidden_idx].shift[2]);
    return tmp_i8;
}


__device__ __forceinline__ int8_t computeH(
    int hidden_idx,
    int8_t z,
    int8_t g,
    int8_t h_old,
    const QuantGRUReScale &rescale_params
) {
    const int32_t zh_aligned = dev::rescale(static_cast<int32_t>(z) * static_cast<int32_t>(h_old),
                                            rescale_params.zh_old_to_h_out[hidden_idx].M,
                                            rescale_params.zh_old_to_h_out[hidden_idx].shift);
    const int32_t zg_aligned = dev::rescale(static_cast<int32_t>(127 - z) * static_cast<int32_t>(g),
                                            rescale_params.zg_to_h_out[hidden_idx].M,
                                            rescale_params.zg_to_h_out[hidden_idx].shift);

    return dev::clamp<int8_t>(zh_aligned + zg_aligned);
}

// x : 非对称量化, scale分时间步不同
// W : 对称量化, scale分为三个门, 分为
// R : 对称量化, scale分为三个门
// bx : 对称量化, scale分为三个门
// br : 对称量化, scale分为三个门
// h : 对称量化, scale分时间步不同
//
// C = input_size(输入维度), H = hidden_size(隐藏层维度),
// T = time_steps(时间步), N = batch_size(批量大小)
template<typename T, bool Training, bool ApplyZoneout>
__global__ void PointwiseOperationsQuant(
    const int batch_dim, // 批量大小
    const int hidden_dim, // 隐藏单元数
    const int32_t *Wx, // 前向矩阵乘W * x, 包含Wz, Wr, Wh
    const int32_t *Rh, // 前向矩阵乘R * h, 包含Rz, Rr, Rh
    const int32_t *bx, // 输入偏置, 包含bz, br, bh
    const int32_t *br, // 隐藏偏置, 包含bz, br, bh
    const T *h, // 上一时间步隐藏状态
    T *h_out, // 当前时间步隐藏状态
    T *v, // 保存内部分量用于反向传播
    const T zoneout_prob, // Zoneout概率
    const T *zoneout_mask, // 训练模式用
    const QuantGRUReScale re_scale_param
) {  // Zoneout mask (only used if ApplyZoneout==true)

    /* 计算索引 */
    const int row = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程对应的隐藏单元
    const int col = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程对应的batch样本

    if (row >= hidden_dim || col >= batch_dim) return; // 边缘判断

    const int weight_idx = col * (hidden_dim * 3) + row; // 用于访问 [Wx, Rh] 的展开索引

    // Index into the `h` and `h_out` vectors (they have a stride of
    // `hidden_dim`).
    const int output_idx = col * hidden_dim + row;

    // Indicies into the Wx and Rh matrices (for each of the u, r, and e
    // components).
    const int z_idx = weight_idx + 0 * hidden_dim;
    const int r_idx = weight_idx + 1 * hidden_dim;
    const int g_idx = weight_idx + 2 * hidden_dim;

    // Indices into the bias vectors (for each of the u, r, and e components).
    const int b_z_idx = row + 0 * hidden_dim; // 更新门对应索引
    const int b_r_idx = row + 1 * hidden_dim; // 重置门对应索引
    const int b_g_idx = row + 2 * hidden_dim; // 候选状态对应索引

    /* GRU前向计算 */

    T z, r, g; // 三个门控
    if constexpr (std::is_same_v<T, int8_t>) {
        // int8 量化
        const int8_t z_tmp_i8 = computeZ(row, Wx[z_idx], Rh[z_idx], bx[b_z_idx], br[b_z_idx], re_scale_param);
        z = dev::sigmoid_int8_lut(z_tmp_i8); // 更新门z

        const int8_t r_tmp_i8 = computeR(row, Wx[r_idx], Rh[r_idx], bx[b_r_idx], br[b_r_idx], re_scale_param);
        r = dev::sigmoid_int8_lut(r_tmp_i8); // 重置门r

        const int8_t g_tmp_i8 = computeG(row, Wx[g_idx], Rh[g_idx], bx[b_g_idx], br[b_g_idx], r, re_scale_param);
        g = dev::tanh_int8_lut(g_tmp_i8); // 候选状态~ht
    } else {
        // TODO: int16 量化
    }

    /* 训练模式 */
    // Store internal activations if we're eventually going to backprop.
    if (Training) {
        const int base_v_idx = col * (hidden_dim * 4) + row;
        v[base_v_idx + 0 * hidden_dim] = z;
        v[base_v_idx + 1 * hidden_dim] = r;
        v[base_v_idx + 2 * hidden_dim] = g;
        v[base_v_idx + 3 * hidden_dim] = Rh[g_idx] + br[b_g_idx];
    }

    T cur_h_value = computeH(row, z, g, h[output_idx], re_scale_param);

    /* 启用Zoneout, 对GRU 隐藏状态的随机保留 */
    // TODO: 支持量化
//    if (ApplyZoneout) {
//        if (Training) {
//            cur_h_value = (cur_h_value - h[output_idx]) * zoneout_mask[output_idx] +
//                          h[output_idx];
//        } else {
//            cur_h_value = (zoneout_prob * h[output_idx]) +
//                          ((static_cast<T>(1.0) - zoneout_prob) * cur_h_value);
//        }
//    }

    /* 结果储存 */
    h_out[output_idx] = cur_h_value;
}

//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
//
//template<typename T, bool Training, bool ApplyZoneout>
//__global__ void PointwiseOperations(const int batch_dim, const int hidden_dim,
//                                    const half *Wx, const half *Rh,
//                                    const half *bx, const half *br,
//                                    const half *h, half *h_out, half *v,
//                                    const half zoneout_prob,
//                                    const half *zoneout_mask) {
//    device_assert_fail("FP16 is not supported on compute capability < 7.0.");
//}
//
//#endif

}  // kernel namespace


namespace gru {

template<typename T>
struct ForwardPassQuant<T>::private_data {
  bool training;
  int batch_size;
  int input_size;
  int hidden_size;
  cublasHandle_t blas_handle;
  cudaStream_t stream[2];
  cudaEvent_t event;
  cudaStream_t sync_stream;
};

template<typename T>
ForwardPassQuant<T>::ForwardPassQuant(const bool training, const int batch_size,
                                      const int input_size, const int hidden_size,
                                      const cublasHandle_t &blas_handle,
                                      const cudaStream_t &stream)
    : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->sync_stream = stream;
    cudaStreamCreate(&data_->stream[0]);
    cudaStreamCreate(&data_->stream[1]);
    cudaEventCreateWithFlags(&data_->event, cudaEventDisableTiming);
}

template<typename T>
ForwardPassQuant<T>::~ForwardPassQuant() {
    if (data_->sync_stream) {
        cudaEventRecord(data_->event, data_->stream[1]);
        cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
        cudaEventRecord(data_->event, data_->stream[0]);
        cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
    } else {
        cudaStreamSynchronize(data_->stream[1]);
        cudaStreamSynchronize(data_->stream[0]);
    }
    cudaEventDestroy(data_->event);
    cudaStreamDestroy(data_->stream[1]);
    cudaStreamDestroy(data_->stream[0]);
    delete data_;
}

template<typename T>
void ForwardPassQuant<T>::Iterate(const T *W,   // [C,H*3]
                                  const T *R,   // [H,H*3]
                                  const int32_t *bx,  // [H*3]
                                  const int32_t *br,  // [H*3]
                                  const T *x,   // [N,C]
                                  const T *h,   // [N,H]
                                  T *h_out,     // [N,H]
                                  T *v,         // [N,H*4]
                                  int32_t *tmp_Wx,    // [N,H*3]
                                  int32_t *tmp_Rh,    // [N,H*3]
                                  const float zoneout_prob,
                                  const T *zoneout_mask  // Zoneout mask [N,H]
) {
    // TODO : 支持量化
//    using alpha_beta_t = std::conditional_t<
//        std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>,
//        int,
//        T>;
//
//    static const alpha_beta_t alpha = static_cast<alpha_beta_t>(1);
//    static const alpha_beta_t beta = static_cast<alpha_beta_t>(0);
//
//    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);
//
//    const int batch_size = data_->batch_size;
//    const int input_size = data_->input_size;
//    const int hidden_size = data_->hidden_size;
//    const cublasHandle_t blas_handle = data_->blas_handle;
//    const cudaStream_t stream2 = data_->stream[1];
//    const cudaEvent_t event = data_->event;
//
//    cudaStream_t save_stream;
//    cublasGetStream(blas_handle, &save_stream);
//
//    cublasSetStream(blas_handle, stream2);
//    blas<T>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3,
//                  batch_size, input_size, &alpha, W, hidden_size * 3, x,
//                  input_size, &beta, tmp_Wx, hidden_size * 3);
//    cudaEventRecord(event, stream2);
//
//    IterateInternal(R, bx, br, h, h_out, v, tmp_Wx, tmp_Rh, zoneout_prob,
//                    zoneout_mask);
//
//    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
void ForwardPassQuant<T>::IterateInternal(
    // C = input_size(输入维度), H = hidden_size(隐藏层维度),
    // T = time_steps(时间步), N = batch_size(批量大小)
    int step_idx,
    const T *R,   // [H,H*3]
    const int32_t *bx,  // [H*3]
    const int32_t *br,  // [H*3]
    const T *h,   // [N,H]
    T *h_out,     // [N,H]
    T *v,         // [N,H*4]
    const int32_t *tmp_Wx,    // [N,H*3]
    int32_t *tmp_Rh,    // [N,H*3]
    const float zoneout_prob,
    const T *zoneout_mask // Zoneout mask [N,H]
) {
    // Constants for GEMM
    static const int32_t alpha = static_cast<int32_t>(1);
    static const int32_t beta = static_cast<int32_t>(0);

    const bool training = data_->training;
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaEvent_t event = data_->event;

    cublasSetStream(blas_handle, stream1);
    blas<T>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3,
                  batch_size, hidden_size, &alpha, R, hidden_size * 3, h,
                  hidden_size, &beta, tmp_Rh, hidden_size * 3);

    // Compute launch configuration for pointwise operations kernel.
    const dim3 blockDim(32, 16);
    const dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x,
                       (batch_size + blockDim.y - 1) / blockDim.y);

    cudaStreamWaitEvent(stream1, event, 0);

    if (training) { // 训练模式
        if (zoneout_prob && zoneout_mask) { // 启用Zoneout, 对GRU 隐藏状态的随机保留
            kernel::PointwiseOperationsQuant<T, true, true><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, v,
                    zoneout_prob, zoneout_mask, rescale_param_);
        } else {
            kernel::PointwiseOperationsQuant<T, true, false><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, v, 0.0f,
                    nullptr, rescale_param_);
        }
    } else { // 推理模式
        if (zoneout_prob && zoneout_mask) {
            kernel::PointwiseOperationsQuant<T, false, true><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, nullptr,
                    zoneout_prob, zoneout_mask, rescale_param_);
        } else {
            kernel::PointwiseOperationsQuant<T, false, false><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, nullptr,
                    0.0f, nullptr, rescale_param_);
        }
    }
}

template<typename T>
void ForwardPassQuant<T>::setRescaleParam(const QuantGRUScales &quantGruScales) {
    std::vector<ScaleParam> Rh_z_to_Wx_z;
    std::vector<ScaleParam> Rh_r_to_Wx_r;
    std::vector<ScaleParam> rRh_g_to_Wx_g;
    std::vector<ScaleParam3> Wx_to_out; // 三个门: z, r, g
    std::vector<ScaleParam> zh_in_to_h_out;
    std::vector<ScaleParam> zg_to_h_out;

    for (int t = 0; t < quantGruScales.steps; ++t) {
        for (int hidden_idx = 0; hidden_idx < quantGruScales.hidden; ++hidden_idx) {
            const int Wx_offset = t * quantGruScales.hidden * 3 + hidden_idx;
            const float Wx_z = quantGruScales.Wx[Wx_offset];
            const float Wx_r = quantGruScales.Wx[Wx_offset + 1];
            const float Wx_g = quantGruScales.Wx[Wx_offset + 2];

            const float Rh_z = quantGruScales.Rh[Wx_offset];
            const float Rh_r = quantGruScales.Rh[Wx_offset + 1];
            const float Rh_g = quantGruScales.Rh[Wx_offset + 2];
        }
    }
}

// C = input_size(输入维度), H = hidden_size(隐藏层维度),
// T = time_steps(时间步), N = batch_size(批量大小)
template<typename T>
void ForwardPassQuant<T>::Run(const int steps, // 时间步数, 序列长度T
                              const T *W,   // [C,H*3], 输入到隐藏状态的权重矩阵（Wx）, 对应 GRU 的三个门（z、r、h）。C 是输入特征维度，H 是隐藏状态维度, （行主序，计算 x @ W）
                              const T *R,   // [H,H*3], 隐状态到隐藏状态的权重矩阵（Rh），对应 GRU 的三个门（z、r、h）. （行主序，计算 h @ R）
                              const int32_t *bx,  // [H*3], 输入偏置（bias for W），对应 z、r、h 门
                              const int32_t *br,  // [H*3], 隐状态偏置（bias for R），对应 z、r、h 门
                              const T *x,   // [N,C], 输入序列，batch_size = N，特征维度 = C
                              T *h,         // [N,H], 输出隐藏状态，每个时间步保存的 GRU 隐状态
                              T *v,         // [N,H*4], 临时存储向量/中间计算值，通常保存 z, r, h_tilde, h_new 的中间值，用于后向传播或 zoneout
                              int32_t *tmp_Wx,    // [N,H*3], W * x 的临时结果
                              int32_t *tmp_Rh,    // [N,H*3], R * h 的临时结果
                              const float zoneout_prob, // Zoneout 概率，用于随机丢弃部分隐藏状态
                              const T *zoneout_mask // Zoneout mask，0/1 矩阵，控制哪些隐藏单元被保留,  // Zoneout mask [N,H]
) {
    static const int32_t alpha = static_cast<int32_t>(1);
    static const int32_t beta = static_cast<int32_t>(0);

    const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream2 = data_->stream[1];
    const cudaEvent_t event = data_->event;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    cublasSetStream(blas_handle, stream2);
    blas<T>::gemm(blas_handle,  // 提前使用cuBlas计算W * x
                  CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3, steps * batch_size,
                  input_size, &alpha, W, hidden_size * 3, x, input_size, &beta,
                  tmp_Wx, hidden_size * 3);

    // Test
    std::vector<int32_t> Wx_host = d2h(tmp_Wx, batch_size * hidden_size * 3);
    for (auto &it : Wx_host) {
        if (it == 0) {
            printf("Error!!\n");
        }
    }

    // gemm后补偿x_zp
    dev::vector<int32_t> w_sum(1);
    computeWeightSum(W, w_sum.data(), hidden_size * 3, input_size, stream2);
//    dev::vector<int32_t> x_zp(gruQuantScales_.x_zp);
//    applyZeroPointCompensation2D(tmp_Wx, w_sum.data(), x_zp.data(), hidden_size * 3, steps * batch_size, stream2);

    // 同步Wx计算
    cudaEventRecord(event, stream2);

    std::vector<int32_t> Wx_host2 = d2h(tmp_Wx, batch_size * hidden_size * 3);
    for (auto &it : Wx_host2) {
        if (it == 0) {
            printf("Error2!!\n");
        }
    }

    const int NH = batch_size * hidden_size;

    for (int i = 0; i < steps; ++i) {
//        // 计算当前时间步的Rh_scale和Rh对其到Wx
//        gruQuantScales_.Rh[i] = combineRWithH(gruQuantScales_.R, gruQuantScales_.h[i]);
//        rescaleParam_[i].Rh_to_Wx = alignRhToWxShift(gruQuantScales_.Rh[i], gruQuantScales_.Wx[i]);

        IterateInternal(i, R, bx, br, h + i * NH, h + (i + 1) * NH, v + i * NH * 4,
                        tmp_Wx + i * NH * 3, tmp_Rh, zoneout_prob,
                        zoneout_mask ? zoneout_mask + i * NH : nullptr);

//        cudaDeviceSynchronize();
//        // 计算下一步的h的scale(M, shift)
//        const T max_val = findMaxValueFromDev(h + (i + 1) * NH, NH);
//        gruQuantScales_.h[i + 1] = updateHScale(gruQuantScales_.h[i], max_val);
    }

    cublasSetStream(blas_handle, save_stream);
}

template
struct ForwardPassQuant<int8_t>;
template
struct ForwardPassQuant<int16_t>;

}  // namespace gru
