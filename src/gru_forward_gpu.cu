#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstdio>

#include "blas.h"
#include "gru.h"
#include "inline_ops.h"
#include "device_ptr.h"
#include "quantize_ops.cuh"

namespace kernel {

//__global__ void quantize_int32_to_int8(const int32_t *intput,
//                                       int8_t *output,
//                                       int M,
//                                       int N,
//                                       int32_t inv_scale,
//                                       int32_t zero_point) {
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if (row < M && col < N) {
//        const int idx = row * N + col;  // Calculate the index for the element in tmp_Wx_dev
//
//        output[idx] = quantize_to_int8(intput[idx], inv_scale, zero_point);
//    }
//}


//// 转置类型枚举
//enum class GemmTranspose { N, T };
//
//// int8 GEMM + 量化 kernel 支持转置
//template<GemmTranspose transA, GemmTranspose transB>
//__global__ void gemm_int8_requant_kernel(const int8_t *__restrict__ A,
//                                         const int8_t *__restrict__ B,
//                                         int8_t *__restrict__ C,
//                                         int M, int N, int K,
//                                         float scale) {
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if (row < M && col < N) {
//        int32_t sum = 0;
//        for (int k = 0; k < K; ++k) {
//            int a_val = transA == GemmTranspose::N ? A[row * K + k] : A[k * M + row];
//            int b_val = transB == GemmTranspose::N ? B[k * N + col] : B[col * K + k];
//            sum += static_cast<int32_t>(a_val) * static_cast<int32_t>(b_val);
//        }
//
//        // requantization
//        float tmp = sum * scale;
//        tmp = roundf(tmp);
//        tmp = tmp > 127.f ? 127.f : (tmp < -128.f ? -128.f : tmp);
//        C[row * N + col] = static_cast<int8_t>(tmp);
//    }
//}


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
    const T *zoneout_mask // 训练模式用
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
    const int bz_idx = row + 0 * hidden_dim; // 更新门对应索引
    const int br_idx = row + 1 * hidden_dim; // 重置门对应索引
    const int bg_idx = row + 2 * hidden_dim; // 候选状态对应索引

    /* GRU前向计算 */
    RescaleParam rescale_Rh_to_Wx_z, rescale_bx_to_Wx_z, rescale_br_to_Wx_z, rescale_Wx_z_to_z;
    RescaleParam rescale_Rh_to_Wx_r, rescale_bx_to_Wx_r, rescale_br_to_Wx_r, rescale_Wx_r_to_r;
    RescaleParam rescale_Rh_to_Wx_g, rescale_bx_to_Wx_g, rescale_br_to_Wx_g, rescale_r_to_Wx_g, rescale_Wx_g_to_g;

    // step1: 计算 M_r
//    shift_r = 15; // 举例，8~16 比较常用
//    M_r = round((r_scale / acc_scale) * (1 << shift_r));

    T z, r, g; // 三个门控
    if constexpr (std::is_same_v<T, int8_t>) { // int8 量化

        const int32_t Rh_aligned_z = dev::rescale(Rh[z_idx],
                                                  rescale_Rh_to_Wx_z.M,
                                                  rescale_Rh_to_Wx_z.shift); // 对齐到 Wx 的scale
        const int32_t bx_aligned_z = dev::rescale(bx[bz_idx],
                                                  rescale_bx_to_Wx_z.M,
                                                  rescale_bx_to_Wx_z.shift); // 对齐到 Wx 的scale
        const int32_t br_aligned_z = dev::rescale(br[bz_idx],
                                                  rescale_br_to_Wx_z.M,
                                                  rescale_br_to_Wx_z.shift); // 对齐到 Wx 的scale
        const int32_t z_tmp_i32 = Wx[z_idx] + Rh_aligned_z + bx_aligned_z + br_aligned_z;
        const int8_t z_tmp_i8 = dev::quantize_i32_to_i8(z_tmp_i32, rescale_Wx_z_to_z.M, rescale_Wx_z_to_z.shift);
        z = dev::sigmoid_int8_lut(z_tmp_i8); // 更新门z

        const int32_t Rh_aligned_r = dev::rescale(Rh[r_idx],
                                                  rescale_Rh_to_Wx_r.M,
                                                  rescale_Rh_to_Wx_r.shift); // 对齐到 Wx 的scale
        const int32_t bx_aligned_r = dev::rescale(bx[br_idx],
                                                  rescale_bx_to_Wx_r.M,
                                                  rescale_bx_to_Wx_r.shift); // 对齐到 Wx 的scale
        const int32_t br_aligned_r = dev::rescale(br[br_idx],
                                                  rescale_br_to_Wx_r.M,
                                                  rescale_br_to_Wx_r.shift); // 对齐到 Wx 的scale
        const int32_t r_tmp_i32 = Wx[r_idx] + Rh_aligned_r + bx_aligned_r + br_aligned_r;
        const int8_t r_tmp_i8 = dev::quantize_i32_to_i8(r_tmp_i32, rescale_Wx_r_to_r.M, rescale_Wx_r_to_r.shift);
        r = dev::sigmoid_int8_lut(r_tmp_i8); // 重置门r

        const int32_t Rh_aligned_g = dev::rescale(Rh[g_idx],
                                                  rescale_Rh_to_Wx_g.M,
                                                  rescale_Rh_to_Wx_g.shift); // 对齐到 Wx 的scale
        const int32_t bx_aligned_g = dev::rescale(bx[bg_idx],
                                                  rescale_bx_to_Wx_g.M,
                                                  rescale_bx_to_Wx_g.shift); // 对齐到 Wx 的scale
        const int32_t br_aligned_g = dev::rescale(br[bg_idx],
                                                  rescale_br_to_Wx_g.M,
                                                  rescale_br_to_Wx_g.shift); // 对齐到 Wx 的scale
        const int32_t r_aligned_g = dev::rescale((int32_t) r,
                                                 rescale_r_to_Wx_g.M,
                                                 rescale_r_to_Wx_g.shift); // 对齐到 Wx 的scale
        const int32_t g_tmp_i32 = Wx[g_idx] + r_aligned_g * (Rh_aligned_g + br_aligned_g) + bx_aligned_g;
        const int8_t g_tmp_i8 = dev::quantize_i32_to_i8(g_tmp_i32, rescale_Wx_g_to_g.M, rescale_Wx_g_to_g.shift);
        g = dev::tanh_int8_lut(g_tmp_i8); // 候选状态~ht
    } else if constexpr (std::is_same_v<T, int16_t>) { // int16 量化

    } else { // 非量化
        z = dev::sigmoid(Wx[z_idx] + Rh[z_idx] + bx[bz_idx] + br[bz_idx]); // 更新门z
        r = dev::sigmoid(Wx[r_idx] + Rh[r_idx] + bx[br_idx] + br[br_idx]); // 重置门r
        g = dev::tanh(Wx[g_idx] + r * (Rh[g_idx] + br[bg_idx]) + bx[bg_idx]); // 候选状态~ht
    }

    /* 训练模式 */
    // Store internal activations if we're eventually going to backprop.
    if (Training) {
        const int base_v_idx = col * (hidden_dim * 4) + row;
        v[base_v_idx + 0 * hidden_dim] = z;
        v[base_v_idx + 1 * hidden_dim] = r;
        v[base_v_idx + 2 * hidden_dim] = g;
        v[base_v_idx + 3 * hidden_dim] = Rh[g_idx] + br[bg_idx];
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
                                     const T *zoneout_mask) {  // Zoneout mask [N,H]
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
                    zoneout_mask);

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
    const T *zoneout_mask) {  // Zoneout mask [N,H]
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

    // TODO: 计算 Rh_scale, bx_scale, br_scale

    // Compute launch configuration for pointwise operations kernel.
    const dim3 blockDim(32, 16);
    const dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x,
                       (batch_size + blockDim.y - 1) / blockDim.y);

    cudaStreamWaitEvent(stream1, event, 0);

    if (training) { // 训练模式
        if (zoneout_prob && zoneout_mask) { // 启用Zoneout, 对GRU 隐藏状态的随机保留
            kernel::PointwiseOperations<T, AccumT, true, true><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, v,
                    zoneout_prob, zoneout_mask);
        } else {
            kernel::PointwiseOperations<T, AccumT, true, false><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, v, 0.0f,
                    nullptr);
        }
    } else { // 推理模式
        if (zoneout_prob && zoneout_mask) {
            kernel::PointwiseOperations<T, AccumT, false, true><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, nullptr,
                    zoneout_prob, zoneout_mask);
        } else {
            kernel::PointwiseOperations<T, AccumT, false, false><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, nullptr,
                    0.0f, nullptr);
        }
    }
}

//template<>
//void ForwardPass<int8_t>::IterateInternal(
//    // C = input_size(输入维度), H = hidden_size(隐藏层维度),
//    // T = time_steps(时间步), N = batch_size(批量大小)
//    const int8_t *R,   // [H,H*3]
//    const int8_t *bx,  // [H*3]
//    const int8_t *br,  // [H*3]
//    const int8_t *h,   // [N,H]
//    int8_t *h_out,     // [N,H]
//    int8_t *v,         // [N,H*4]
//    int8_t *tmp_Wx,    // [N,H*3]
//    int8_t *tmp_Rh,    // [N,H*3]
//    int32_t *tmp_Rh_i32, // 为了储存cuBLAS的GEMM中int32输出
//    const float zoneout_prob,
//    const int8_t *zoneout_mask) {  // Zoneout mask [N,H]
//    // Constants for GEMM
//
//    static const int alpha = static_cast<int>(1);
//    static const int beta = static_cast<int>(0);
//
//    const bool training = data_->training;
//    const int batch_size = data_->batch_size;
//    const int hidden_size = data_->hidden_size;
//    const cublasHandle_t blas_handle = data_->blas_handle;
//    const cudaStream_t stream1 = data_->stream[0];
//    const cudaEvent_t event = data_->event;
//
//    cublasSetStream(blas_handle, stream1);
//
//
//    blas<int8_t>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3,
//                       batch_size, hidden_size, &alpha, R, hidden_size * 3, h,
//                       hidden_size, &beta, tmp_Rh_i32, hidden_size * 3);
//
////    // Optionally synchronize if needed
////    cudaStreamSynchronize(stream1);
//
//    const int M = data_->hidden_size * 3;
//    const int N = data_->batch_size;
//    // Define block and grid sizes for the kernel launch
//    dim3 block(16, 16);  // Example block size
//    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
//
//    constexpr float scale = 1.0f;
//    // Launch the kernel to quantize tmp_Wx_dev to tmp_Wx
//    kernel::quantize_int32_to_int8<<<grid, block, 0, stream1>>>(tmp_Rh_i32, tmp_Rh, M, N, scale);
//
////    // Optionally synchronize if needed
////    cudaStreamSynchronize(stream1);
//
//    // Compute launch configuration for pointwise operations kernel.
//    const dim3 blockDim(32, 16);
//    const dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x,
//                       (batch_size + blockDim.y - 1) / blockDim.y);
//
//    cudaStreamWaitEvent(stream1, event, 0);
//
//    if (training) { // 训练模式
//        if (zoneout_prob && zoneout_mask) { // 启用Zoneout, 对GRU 隐藏状态的随机保留
//            kernel::PointwiseOperations<int8_t, true, true><<<gridDim, blockDim, 0, stream1>>>(
//                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, v,
//                    zoneout_prob, zoneout_mask);
//        } else {
//            kernel::PointwiseOperations<int8_t, true, false><<<gridDim, blockDim, 0, stream1>>>(
//                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, v, 0.0f,
//                    nullptr);
//        }
//    } else { // 推理模式
//        if (zoneout_prob && zoneout_mask) {
//            kernel::PointwiseOperations<int8_t, false, true><<<gridDim, blockDim, 0, stream1>>>(
//                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, nullptr,
//                    zoneout_prob, zoneout_mask);
//        } else {
//            kernel::PointwiseOperations<int8_t, false, false><<<gridDim, blockDim, 0, stream1>>>(
//                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, nullptr,
//                    0.0f, nullptr);
//        }
//    }
//}

//void gemm_int8_requant(const int8_t *A, const int8_t *B, int8_t *C,
//                       int M, int N, int K, float scale,
//                       cudaStream_t stream = 0) {
//    dim3 block(16, 16);
//    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
//
//    kernel::gemm_int8_requant_kernel<kernel::GemmTranspose::N, kernel::GemmTranspose::N>
//    <<<grid, block, 0, stream>>>(A, B, C, M, N, K, scale);
//    cudaStreamSynchronize(stream);
//}

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
                                 const T *zoneout_mask // Zoneout mask，0/1 矩阵，控制哪些隐藏单元被保留,  // Zoneout mask [N,H]
) {

    using AlphaBetaType = typename std::conditional<std::is_integral<T>::value, int, T>::type;
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

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    cublasSetStream(blas_handle, stream2);
    blas<T>::gemm(blas_handle,  // 提前使用cuBlas计算W * x
                  CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3, steps * batch_size,
                  input_size, &alpha, W, hidden_size * 3, x, input_size, &beta,
                  tmp_Wx, hidden_size * 3);
    cudaEventRecord(event, stream2);

    // TODO: 计算Wx_scale

    printf("cudaError(ForwardPass): %s\n", cudaGetErrorString(cudaGetLastError()));

    const int NH = batch_size * hidden_size;

    for (int i = 0; i < steps; ++i) {
        IterateInternal(R, bx, br, h + i * NH, h + (i + 1) * NH, v + i * NH * 4,
                        tmp_Wx + i * NH * 3, tmp_Rh, zoneout_prob,
                        zoneout_mask ? zoneout_mask + i * NH : nullptr);
    }


    printf("cudaError(ForwardPass): %s\n", cudaGetErrorString(cudaGetLastError()));

    cublasSetStream(blas_handle, save_stream);
}

//template<>
//void ForwardPass<int8_t>::Run(const int steps, // 时间步数, 序列长度T
//                              const int8_t *W,   // [C,H*3], 输入到隐藏状态的权重矩阵（Wx）, 对应 GRU 的三个门（z、r、h）。C 是输入特征维度，H 是隐藏状态维度
//                              const int8_t *R,   // [H,H*3], 隐状态到隐藏状态的权重矩阵（Rh），对应 GRU 的三个门（z、r、h）
//                              const int32_t *bx,  // [H*3], 输入偏置（bias for W），对应 z、r、h 门
//                              const int32_t *br,  // [H*3], 隐状态偏置（bias for R），对应 z、r、h 门
//                              const int8_t *x,   // [N,C], 输入序列，batch_size = N，特征维度 = C
//                              int8_t *h,         // [N,H], 输出隐藏状态，每个时间步保存的 GRU 隐状态
//                              int8_t *v,         // [N,H*4], 临时存储向量/中间计算值，通常保存 z, r, h_tilde, h_new 的中间值，用于后向传播或 zoneout
//                              int8_t *tmp_Wx,    // [N,H*3], W * x 的临时结果
//                              int8_t *tmp_Rh,    // [N,H*3], R * h 的临时结果
//                              const float zoneout_prob, // Zoneout 概率，用于随机丢弃部分隐藏状态
//                              const int8_t *zoneout_mask // Zoneout mask，0/1 矩阵，控制哪些隐藏单元被保留
//) {
//
//    const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
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
//
//    static const int alpha = static_cast<int>(1);
//    static const int beta = static_cast<int>(0);
//
//    blas<int8_t>::gemm(blas_handle,  // 提前使用cuBlas计算W * x
//                       CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3, steps * batch_size,
//                       input_size, &alpha, W, hidden_size * 3, x, input_size, &beta,
//                       tmp_Wx, hidden_size * 3);
//
//    // Optionally synchronize if needed
//    cudaStreamSynchronize(stream2);
//
//    const int M = data_->hidden_size * 3;
//    const int N = data_->batch_size;
//    // Define block and grid sizes for the kernel launch
//    dim3 block(16, 16);  // Example block size
//    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
//
//    constexpr float scale = 1.0f;
//    // Launch the kernel to quantize tmp_Wx_dev to tmp_Wx
//    kernel::quantize_int32_to_int8<<<grid, block, 0, stream2>>>(tmp_Wx_i32, tmp_Wx, M, N, scale);
//
////        // Optionally synchronize if needed
////        cudaStreamSynchronize(stream2);
//
//
//    cudaEventRecord(event, stream2);
//
//    printf("cudaError(ForwardPass): %s\n", cudaGetErrorString(cudaGetLastError()));
//
//    const int NH = batch_size * hidden_size;
//    for (int i = 0; i < steps; ++i) {
//        IterateInternal(R, bx, br, h + i * NH, h + (i + 1) * NH, v + i * NH * 4,
//                        tmp_Wx + i * NH * 3, tmp_Rh, zoneout_prob,
//                        zoneout_mask ? zoneout_mask + i * NH : nullptr);
//    }
//
//
//    printf("cudaError(ForwardPass): %s\n", cudaGetErrorString(cudaGetLastError()));
//
//    cublasSetStream(blas_handle, save_stream);
//}

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
