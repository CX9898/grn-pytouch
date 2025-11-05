#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "blas.h"
#include "gru.h"
#include "inline_ops.h"
#include "device_ptr.h"
#include "quantize_ops.cuh"

namespace kernel {

__device__ __forceinline__ int8_t computeGate(
    const int32_t Wx_val,   // Wx 对应门的值
    const int32_t Rh_val,   // Rh 对应门的值
    const int32_t bx_val,   // bx 对应门的bias
    const int32_t br_val,   // br 对应门的bias
    const RescaleParamsPerStep &rescaleParams,
    int gate_idx            // 门索引: 0=z, 1=r, 2=g
) {
    // 1. 各项对齐到 Wx 的 scale
    const int32_t Rh_aligned = dev::rescale(Rh_val,
                                            rescaleParams.Rh_to_Wx.M[gate_idx],
                                            rescaleParams.Rh_to_Wx.shift[gate_idx]);
    const int32_t bx_aligned = dev::rescale(bx_val,
                                            rescaleParams.bx_to_Wx.M[gate_idx],
                                            rescaleParams.bx_to_Wx.shift[gate_idx]);
    const int32_t br_aligned = dev::rescale(br_val,
                                            rescaleParams.bx_to_Wx.M[gate_idx],
                                            rescaleParams.bx_to_Wx.shift[gate_idx]);

    // 2. 累加求和
    const int32_t tmp_i32 = Wx_val + Rh_aligned + bx_aligned + br_aligned;

    // 3. 量化回 int8
    const int8_t tmp_i8 = dev::quantize_i32_to_i8(tmp_i32,
                                                  rescaleParams.Wx_to_out.M[gate_idx],
                                                  rescaleParams.Wx_to_out.shift[gate_idx]);
    return tmp_i8;
}

__device__ inline int32_t mul_and_rescale(int8_t a, int32_t b, int32_t M, int shift) {
    return (int32_t) (((int64_t) a * b * M) >> shift);
}

template<typename T, typename AccumT = typename std::conditional<std::is_integral<T>::value,
                                                                 int32_t,
                                                                 T>::type, bool Training, bool ApplyZoneout>
__global__ void PointwiseOperations(
    const int batch_dim, // 批量大小
    const int hidden_dim, // 隐藏单元数
    const AccumT *Wx, // 前向矩阵乘W * x, 包含Wz, Wr, Wh
    const AccumT *Rh, // 前向矩阵乘R * h, 包含Rz, Rr, Rh
    const AccumT *bx, // 输入偏置, 包含bz, br, bh
    const AccumT *br, // 隐藏偏置, 包含bz, br, bh
    const T *h, // 上一时间步隐藏状态
    T *h_out, // 当前时间步隐藏状态
    T *v, // 保存内部分量用于反向传播
    const T zoneout_prob, // Zoneout概率
    const T *zoneout_mask, // 训练模式用
    const RescaleParamsPerStep rescaleParams
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

    // step1: 计算 M_r
//    shift_r = 15; // 举例，8~16 比较常用
//    M_r = round((r_scale / acc_scale) * (1 << shift_r));

    T z, r, g; // 三个门控
    if constexpr (std::is_same_v<T, int8_t>) { // int8 量化


//        const int32_t Rh_aligned_z = dev::rescale(Rh[z_idx],
//                                                  rescaleParams.Rh_to_Wx.M[0],
//                                                  rescaleParams.Rh_to_Wx.shift[0]); // 对齐到 Wx 的scale
//        const int32_t bx_aligned_z = dev::rescale(bx[b_z_idx],
//                                                  rescaleParams.bx_to_Wx.M[0],
//                                                  rescaleParams.bx_to_Wx.shift[0]); // 对齐到 Wx 的scale
//        const int32_t br_aligned_z = dev::rescale(br[b_z_idx],
//                                                  rescaleParams.bx_to_Wx.M[0],
//                                                  rescaleParams.bx_to_Wx.shift[0]); // 对齐到 Wx 的scale
//        const int32_t z_tmp_i32 = Wx[z_idx] + Rh_aligned_z + bx_aligned_z + br_aligned_z;
        const int8_t z_tmp_i8 = computeGate(Wx[z_idx], Rh[z_idx], bx[b_z_idx], br[b_z_idx], rescaleParams, 0);
        z = dev::sigmoid_int8_lut(z_tmp_i8); // 更新门z

//        const int32_t Rh_aligned_r = dev::rescale(Rh[r_idx],
//                                                  rescaleParams.Rh_to_Wx.M[1],
//                                                  rescaleParams.Rh_to_Wx.shift[1]); // 对齐到 Wx 的scale
//        const int32_t bx_aligned_r = dev::rescale(bx[b_r_idx],
//                                                  rescale_bx_to_Wx_r.M,
//                                                  rescale_bx_to_Wx_r.shift); // 对齐到 Wx 的scale
//        const int32_t br_aligned_r = dev::rescale(br[b_r_idx],
//                                                  rescale_br_to_Wx_r.M,
//                                                  rescale_br_to_Wx_r.shift); // 对齐到 Wx 的scale
//        const int32_t r_tmp_i32 = Wx[r_idx] + Rh_aligned_r + bx_aligned_r + br_aligned_r;
        const int8_t r_tmp_i8 = computeGate(Wx[r_idx], Rh[r_idx], bx[b_r_idx], br[b_r_idx], rescaleParams, 1);
        r = dev::sigmoid_int8_lut(r_tmp_i8); // 重置门r

        const int32_t Rh_aligned_g = dev::rescale(Rh[g_idx],
                                                  rescaleParams.Rh_to_Wx.M[2],
                                                  rescaleParams.Rh_to_Wx.shift[2]); // 对齐到 Wx 的scale
        const int32_t bx_aligned_g = dev::rescale(bx[b_g_idx],
                                                  rescaleParams.bx_to_Wx.M[2],
                                                  rescaleParams.bx_to_Wx.shift[2]); // 对齐到 Wx 的scale
        const int32_t br_aligned_g = dev::rescale(br[b_g_idx],
                                                  rescaleParams.br_to_Wx.M[2],
                                                  rescaleParams.br_to_Wx.shift[2]); // 对齐到 Wx 的scale
        // 模拟 r(0~1) * (Rh + br)
        const int32_t Rh_modulated_g = mul_and_rescale(
            r,                               // int8, sigmoid output
            (Rh_aligned_g + br_aligned_g),   // int32, aligned linear term
            rescaleParams.r_to_Wx_g.M,       // 缩放 r*S_r → Wx_g域
            rescaleParams.r_to_Wx_g.shift);
        const int32_t g_tmp_i32 = Wx[g_idx] + Rh_modulated_g + bx_aligned_g;
        const int8_t g_tmp_i8 = dev::quantize_i32_to_i8(g_tmp_i32,
                                                        rescaleParams.Wx_to_out.M[2],
                                                        rescaleParams.Wx_to_out.shift[2]);
        g = dev::tanh_int8_lut(g_tmp_i8); // 候选状态~ht
    } else if constexpr (std::is_same_v<T, int16_t>) { // int16 量化

    } else { // 非量化. half/float/double
        z = dev::sigmoid(Wx[z_idx] + Rh[z_idx] + bx[b_z_idx] + br[b_z_idx]); // 更新门z
        r = dev::sigmoid(Wx[r_idx] + Rh[r_idx] + bx[b_r_idx] + br[b_r_idx]); // 重置门r
        g = dev::tanh(Wx[g_idx] + r * (Rh[g_idx] + br[b_g_idx]) + bx[b_g_idx]); // 候选状态~ht
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

    // TODO: scale对齐
    T cur_h_value = z * h[output_idx] + (static_cast<T>(1.0) - z) * g; // 当前时间步最终隐藏状态ht

    /* 启用Zoneout, 对GRU 隐藏状态的随机保留 */
    if (ApplyZoneout) {
        if (Training) {
            cur_h_value = (cur_h_value - h[output_idx]) * zoneout_mask[output_idx] +
                          h[output_idx];
        } else {
            cur_h_value = (zoneout_prob * h[output_idx]) +
                          ((static_cast<T>(1.0) - zoneout_prob) * cur_h_value);
        }
    }

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

template<typename T, typename AccumT>
struct ForwardPass<T, AccumT>::private_data {
  bool training;
  int batch_size;
  int input_size;
  int hidden_size;
  cublasHandle_t blas_handle;
  cudaStream_t stream[2];
  cudaEvent_t event;
  cudaStream_t sync_stream;
};

template<typename T, typename AccumT>
ForwardPass<T, AccumT>::ForwardPass(const bool training, const int batch_size,
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

template<typename T, typename AccumT>
ForwardPass<T, AccumT>::~ForwardPass() {
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

template<typename T, typename AccumT>
void ForwardPass<T, AccumT>::Iterate(const T *W,   // [C,H*3]
                                     const T *R,   // [H,H*3]
                                     const AccumT *bx,  // [H*3]
                                     const AccumT *br,  // [H*3]
                                     const T *x,   // [N,C]
                                     const T *h,   // [N,H]
                                     T *h_out,     // [N,H]
                                     T *v,         // [N,H*4]
                                     AccumT *tmp_Wx,    // [N,H*3]
                                     AccumT *tmp_Rh,    // [N,H*3]
                                     const float zoneout_prob,
                                     const T *zoneout_mask,  // Zoneout mask [N,H]
                                     const RescaleParamsPerStep &rescaleParams) {
    using alpha_beta_t = std::conditional_t<
        std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>,
        int,
        T>;

    static const alpha_beta_t alpha = static_cast<alpha_beta_t>(1);
    static const alpha_beta_t beta = static_cast<alpha_beta_t>(0);

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
    blas<T>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3,
                  batch_size, input_size, &alpha, W, hidden_size * 3, x,
                  input_size, &beta, tmp_Wx, hidden_size * 3);
    cudaEventRecord(event, stream2);

    IterateInternal(R, bx, br, h, h_out, v, tmp_Wx, tmp_Rh, zoneout_prob,
                    zoneout_mask, rescaleParams);

    cublasSetStream(blas_handle, save_stream);
}

template<typename T, typename AccumT>
void ForwardPass<T, AccumT>::IterateInternal(
    // C = input_size(输入维度), H = hidden_size(隐藏层维度),
    // T = time_steps(时间步), N = batch_size(批量大小)
    const T *R,   // [H,H*3]
    const AccumT *bx,  // [H*3]
    const AccumT *br,  // [H*3]
    const T *h,   // [N,H]
    T *h_out,     // [N,H]
    T *v,         // [N,H*4]
    const AccumT *tmp_Wx,    // [N,H*3]
    AccumT *tmp_Rh,    // [N,H*3]
    const float zoneout_prob,
    const T *zoneout_mask, // Zoneout mask [N,H]
    const RescaleParamsPerStep &rescaleParamsPer
) {
    // Constants for GEMM
    using AlphaBetaType = typename std::conditional<std::is_integral<T>::value, int, T>::type;
    static const AlphaBetaType alpha = static_cast<AlphaBetaType>(1.0);
    static const AlphaBetaType beta = static_cast<AlphaBetaType>(0.0);

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

    // TODO: 计算 Rh_scale

    // Compute launch configuration for pointwise operations kernel.
    const dim3 blockDim(32, 16);
    const dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x,
                       (batch_size + blockDim.y - 1) / blockDim.y);

    cudaStreamWaitEvent(stream1, event, 0);

    if (training) { // 训练模式
        if (zoneout_prob && zoneout_mask) { // 启用Zoneout, 对GRU 隐藏状态的随机保留
            kernel::PointwiseOperations<T, AccumT, true, true><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, v,
                    zoneout_prob, zoneout_mask, rescaleParamsPer);
        } else {
            kernel::PointwiseOperations<T, AccumT, true, false><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, v, 0.0f,
                    nullptr, rescaleParamsPer);
        }
    } else { // 推理模式
        if (zoneout_prob && zoneout_mask) {
            kernel::PointwiseOperations<T, AccumT, false, true><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, nullptr,
                    zoneout_prob, zoneout_mask, rescaleParamsPer);
        } else {
            kernel::PointwiseOperations<T, AccumT, false, false><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, nullptr,
                    0.0f, nullptr, rescaleParamsPer);
        }
    }
}

template<typename T, typename AccumT>
void ForwardPass<T, AccumT>::Run(const int steps, // 时间步数, 序列长度T
                                 const T *W,   // [C,H*3], 输入到隐藏状态的权重矩阵（Wx）, 对应 GRU 的三个门（z、r、h）。C 是输入特征维度，H 是隐藏状态维度
                                 const T *R,   // [H,H*3], 隐状态到隐藏状态的权重矩阵（Rh），对应 GRU 的三个门（z、r、h）
                                 const AccumT *bx,  // [H*3], 输入偏置（bias for W），对应 z、r、h 门
                                 const AccumT *br,  // [H*3], 隐状态偏置（bias for R），对应 z、r、h 门
                                 const T *x,   // [N,C], 输入序列，batch_size = N，特征维度 = C
                                 T *h,         // [N,H], 输出隐藏状态，每个时间步保存的 GRU 隐状态
                                 T *v,         // [N,H*4], 临时存储向量/中间计算值，通常保存 z, r, h_tilde, h_new 的中间值，用于后向传播或 zoneout
                                 AccumT *tmp_Wx,    // [N,H*3], W * x 的临时结果
                                 AccumT *tmp_Rh,    // [N,H*3], R * h 的临时结果
                                 const float zoneout_prob, // Zoneout 概率，用于随机丢弃部分隐藏状态
                                 const T *zoneout_mask, // Zoneout mask，0/1 矩阵，控制哪些隐藏单元被保留,  // Zoneout mask [N,H]
                                 const std::optional<GruQuantScales> &gruQuantParams
) {

    using AlphaBetaType = typename std::conditional<std::is_integral<T>::value, int32_t, T>::type;
    static const AlphaBetaType alpha = static_cast<T>(1.0);
    static const AlphaBetaType beta = static_cast<T>(0.0);

    const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream2 = data_->stream[1];
    const cudaEvent_t event = data_->event;

    // TODO: 尝试手写GEMM, 融合补偿x_zp[steps].

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    cublasSetStream(blas_handle, stream2);
    blas<T>::gemm(blas_handle,  // 提前使用cuBlas计算W * x
                  CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3, steps * batch_size,
                  input_size, &alpha, W, hidden_size * 3, x, input_size, &beta,
                  tmp_Wx, hidden_size * 3);
    cudaEventRecord(event, stream2);

    // TODO: 计算每个时间步的Wx_scale[steps * 3个门], 可以在一个kernel中.
//    applyZeroPointCompensation2D(tmp_Wx,W_sum,);

    std::vector<RescaleParamsPerStep> rescaleParams(steps);

    const int NH = batch_size * hidden_size;

    for (int i = 0; i < steps; ++i) {
        if (i > 0) {
            // TODO: 计算当前时间步的h_scale
        }
        IterateInternal(R, bx, br, h + i * NH, h + (i + 1) * NH, v + i * NH * 4,
                        tmp_Wx + i * NH * 3, tmp_Rh, zoneout_prob,
                        zoneout_mask ? zoneout_mask + i * NH : nullptr, rescaleParams[i]);
    }

    cublasSetStream(blas_handle, save_stream);
}

template
struct ForwardPass<int8_t>;
template
struct BackwardPass<int16_t>;
//template
//struct ForwardPass<half>;
template
struct ForwardPass<float>;
template
struct ForwardPass<double>;

}  // namespace gru
