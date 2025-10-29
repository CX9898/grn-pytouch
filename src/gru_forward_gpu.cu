#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstdio>

#include "blas.h"
#include "gru.h"
#include "inline_ops.h"
#include "device_ptr.h"

namespace kernel {

__global__ void quantize_int32_to_int8(const int32_t *tmp_Wx_dev, int8_t *tmp_Wx, int M, int N, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        const int idx = row * N + col;  // Calculate the index for the element in tmp_Wx_dev

        // Apply quantization: scale and clip to int8 range [-128, 127]
        int32_t value = tmp_Wx_dev[idx];
        float quantized_value = roundf(value * scale);  // Apply the scale and round
        quantized_value = fminf(fmaxf(quantized_value, -128.f), 127.f);  // Clamp to int8 range

        // Store the quantized value in tmp_Wx
        tmp_Wx[idx] = static_cast<int8_t>(quantized_value);
    }
}


// 转置类型枚举
enum class GemmTranspose { N, T };

// int8 GEMM + 量化 kernel 支持转置
template<GemmTranspose transA, GemmTranspose transB>
__global__ void gemm_int8_requant_kernel(const int8_t *__restrict__ A,
                                         const int8_t *__restrict__ B,
                                         int8_t *__restrict__ C,
                                         int M, int N, int K,
                                         float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int32_t sum = 0;
        for (int k = 0; k < K; ++k) {
            int a_val = transA == GemmTranspose::N ? A[row * K + k] : A[k * M + row];
            int b_val = transB == GemmTranspose::N ? B[k * N + col] : B[col * K + k];
            sum += static_cast<int32_t>(a_val) * static_cast<int32_t>(b_val);
        }

        // requantization
        float tmp = sum * scale;
        tmp = roundf(tmp);
        tmp = tmp > 127.f ? 127.f : (tmp < -128.f ? -128.f : tmp);
        C[row * N + col] = static_cast<int8_t>(tmp);
    }
}


template<typename T, bool Training, bool ApplyZoneout>
__global__ void PointwiseOperations(
    const int batch_dim, // 批量大小
    const int hidden_dim, // 隐藏单元数
    const T *Wx, // 前向矩阵乘W * x, 包含Wz, Wr, Wh
    const T *Rh, // 前向矩阵乘R * h, 包含Rz, Rr, Rh
    const T *bx, // 输入偏置, 包含bz, br, bh
    const T *br, // 隐藏偏置, 包含bz, br, bh
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
    const T z = dev::sigmoid(Wx[z_idx] + Rh[z_idx] + bx[bz_idx] + br[bz_idx]); // 更新门z
    const T r = dev::sigmoid(Wx[r_idx] + Rh[r_idx] + bx[br_idx] + br[br_idx]); // 重置门r
    const T g = dev::tanh(Wx[g_idx] + r * (Rh[g_idx] + br[bg_idx]) + bx[bg_idx]); // 候选状态~ht

    /* 训练模式 */
    // Store internal activations if we're eventually going to backprop.
    if (Training) {
        const int base_v_idx = col * (hidden_dim * 4) + row;
        v[base_v_idx + 0 * hidden_dim] = z;
        v[base_v_idx + 1 * hidden_dim] = r;
        v[base_v_idx + 2 * hidden_dim] = g;
        v[base_v_idx + 3 * hidden_dim] = Rh[g_idx] + br[bg_idx];
    }

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

template<typename T>
struct ForwardPass<T>::private_data {
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
ForwardPass<T>::ForwardPass(const bool training, const int batch_size,
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
ForwardPass<T>::~ForwardPass() {
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
void ForwardPass<T>::Iterate(const T *W,   // [C,H*3]
                             const T *R,   // [H,H*3]
                             const T *bx,  // [H*3]
                             const T *br,  // [H*3]
                             const T *x,   // [N,C]
                             const T *h,   // [N,H]
                             T *h_out,     // [N,H]
                             T *v,         // [N,H*4]
                             T *tmp_Wx,    // [N,H*3]
                             T *tmp_Rh,    // [N,H*3]
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

    IterateInternal(R, bx, br, h, h_out, v, tmp_Wx, tmp_Rh, nullptr, zoneout_prob,
                    zoneout_mask);

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
void ForwardPass<T>::IterateInternal(
    // C = input_size(输入维度), H = hidden_size(隐藏层维度),
    // T = time_steps(时间步), N = batch_size(批量大小)
    const T *R,   // [H,H*3]
    const T *bx,  // [H*3]
    const T *br,  // [H*3]
    const T *h,   // [N,H]
    T *h_out,     // [N,H]
    T *v,         // [N,H*4]
    T *tmp_Wx,    // [N,H*3]
    T *tmp_Rh,    // [N,H*3]
    int32_t *tmp_Rh_i32, // 为了储存cuBLAS的GEMM中int32输出
    const float zoneout_prob,
    const T *zoneout_mask) {  // Zoneout mask [N,H]
    // Constants for GEMM
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

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
            kernel::PointwiseOperations<T, true, true><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, v,
                    zoneout_prob, zoneout_mask);
        } else {
            kernel::PointwiseOperations<T, true, false><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, v, 0.0f,
                    nullptr);
        }
    } else { // 推理模式
        if (zoneout_prob && zoneout_mask) {
            kernel::PointwiseOperations<T, false, true><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, nullptr,
                    zoneout_prob, zoneout_mask);
        } else {
            kernel::PointwiseOperations<T, false, false><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, nullptr,
                    0.0f, nullptr);
        }
    }
}

template<>
void ForwardPass<int8_t>::IterateInternal(
    // C = input_size(输入维度), H = hidden_size(隐藏层维度),
    // T = time_steps(时间步), N = batch_size(批量大小)
    const int8_t *R,   // [H,H*3]
    const int8_t *bx,  // [H*3]
    const int8_t *br,  // [H*3]
    const int8_t *h,   // [N,H]
    int8_t *h_out,     // [N,H]
    int8_t *v,         // [N,H*4]
    int8_t *tmp_Wx,    // [N,H*3]
    int8_t *tmp_Rh,    // [N,H*3]
    int32_t *tmp_Rh_i32, // 为了储存cuBLAS的GEMM中int32输出
    const float zoneout_prob,
    const int8_t *zoneout_mask) {  // Zoneout mask [N,H]
    // Constants for GEMM

    static const int alpha = static_cast<int>(1);
    static const int beta = static_cast<int>(0);

    const bool training = data_->training;
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaEvent_t event = data_->event;

    cublasSetStream(blas_handle, stream1);


    blas<int8_t>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3,
                       batch_size, hidden_size, &alpha, R, hidden_size * 3, h,
                       hidden_size, &beta, tmp_Rh_i32, hidden_size * 3);

//    // Optionally synchronize if needed
//    cudaStreamSynchronize(stream1);

    const int M = data_->hidden_size * 3;
    const int N = data_->batch_size;
    // Define block and grid sizes for the kernel launch
    dim3 block(16, 16);  // Example block size
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    constexpr float scale = 1.0f;
    // Launch the kernel to quantize tmp_Wx_dev to tmp_Wx
    kernel::quantize_int32_to_int8<<<grid, block, 0, stream1>>>(tmp_Rh_i32, tmp_Rh, M, N, scale);

//    // Optionally synchronize if needed
//    cudaStreamSynchronize(stream1);

    // Compute launch configuration for pointwise operations kernel.
    const dim3 blockDim(32, 16);
    const dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x,
                       (batch_size + blockDim.y - 1) / blockDim.y);

    cudaStreamWaitEvent(stream1, event, 0);

    if (training) { // 训练模式
        if (zoneout_prob && zoneout_mask) { // 启用Zoneout, 对GRU 隐藏状态的随机保留
            kernel::PointwiseOperations<int8_t, true, true><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, v,
                    zoneout_prob, zoneout_mask);
        } else {
            kernel::PointwiseOperations<int8_t, true, false><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, v, 0.0f,
                    nullptr);
        }
    } else { // 推理模式
        if (zoneout_prob && zoneout_mask) {
            kernel::PointwiseOperations<int8_t, false, true><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, nullptr,
                    zoneout_prob, zoneout_mask);
        } else {
            kernel::PointwiseOperations<int8_t, false, false><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, nullptr,
                    0.0f, nullptr);
        }
    }
}

void gemm_int8_requant(const int8_t *A, const int8_t *B, int8_t *C,
                       int M, int N, int K, float scale,
                       cudaStream_t stream = 0) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    kernel::gemm_int8_requant_kernel<kernel::GemmTranspose::N, kernel::GemmTranspose::N>
    <<<grid, block, 0, stream>>>(A, B, C, M, N, K, scale);
    cudaStreamSynchronize(stream);
}

template<typename T>
void ForwardPass<T>::Run(const int steps,
                         const T *W,   // [C,H*3]
                         const T *R,   // [H,H*3]
                         const T *bx,  // [H*3]
                         const T *br,  // [H*3]
                         const T *x,   // [N,C]
                         T *h,         // [N,H]
                         T *v,         // [N,H*4]
                         T *tmp_Wx,   // [N,H*3]
                         T *tmp_Rh,    // [N,H*3]
                         const float zoneout_prob,
                         const T *zoneout_mask) {  // Zoneout mask [N,H]
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

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

    printf("cudaError(ForwardPass): %s\n", cudaGetErrorString(cudaGetLastError()));

    const int NH = batch_size * hidden_size;
    for (int i = 0; i < steps; ++i) {
        IterateInternal(R, bx, br, h + i * NH, h + (i + 1) * NH, v + i * NH * 4,
                        tmp_Wx + i * NH * 3, tmp_Rh, nullptr, zoneout_prob,
                        zoneout_mask ? zoneout_mask + i * NH : nullptr);
    }


    printf("cudaError(ForwardPass): %s\n", cudaGetErrorString(cudaGetLastError()));

    cublasSetStream(blas_handle, save_stream);
}

template<>
void ForwardPass<int8_t>::Run(const int steps,
                              const int8_t *W,   // [C,H*3]
                              const int8_t *R,   // [H,H*3]
                              const int8_t *bx,  // [H*3]
                              const int8_t *br,  // [H*3]
                              const int8_t *x,   // [N,C]
                              int8_t *h,         // [N,H]
                              int8_t *v,         // [N,H*4]
                              int8_t *tmp_Wx,    // [N,H*3]
                              int8_t *tmp_Rh,    // [N,H*3]
                              const float zoneout_prob,
                              const int8_t *zoneout_mask) {

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

    static const int alpha = static_cast<int>(1);
    static const int beta = static_cast<int>(0);

    int32_t *tmp_Wx_i32;
    int32_t *tmp_Rh_i32;
    cudaMalloc(&tmp_Wx_i32, sizeof(int32_t) * steps * batch_size * hidden_size * 3);
    cudaMalloc(&tmp_Rh_i32, sizeof(int32_t) * batch_size * hidden_size * 3);

    blas<int8_t>::gemm(blas_handle,  // 提前使用cuBlas计算W * x
                       CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3, steps * batch_size,
                       input_size, &alpha, W, hidden_size * 3, x, input_size, &beta,
                       tmp_Wx_i32, hidden_size * 3);

    // Optionally synchronize if needed
    cudaStreamSynchronize(stream2);

    const int M = data_->hidden_size * 3;
    const int N = data_->batch_size;
    // Define block and grid sizes for the kernel launch
    dim3 block(16, 16);  // Example block size
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    constexpr float scale = 1.0f;
    // Launch the kernel to quantize tmp_Wx_dev to tmp_Wx
    kernel::quantize_int32_to_int8<<<grid, block, 0, stream2>>>(tmp_Wx_i32, tmp_Wx, M, N, scale);

//        // Optionally synchronize if needed
//        cudaStreamSynchronize(stream2);


    cudaEventRecord(event, stream2);

    printf("cudaError(ForwardPass): %s\n", cudaGetErrorString(cudaGetLastError()));

    const int NH = batch_size * hidden_size;
    for (int i = 0; i < steps; ++i) {
        IterateInternal(R, bx, br, h + i * NH, h + (i + 1) * NH, v + i * NH * 4,
                        tmp_Wx + i * NH * 3, tmp_Rh, tmp_Rh_i32, zoneout_prob,
                        zoneout_mask ? zoneout_mask + i * NH : nullptr);
    }


    printf("cudaError(ForwardPass): %s\n", cudaGetErrorString(cudaGetLastError()));

    cublasSetStream(blas_handle, save_stream);

    cudaFree(tmp_Wx_i32);
}

template
struct ForwardPass<int8_t>;
//template
//struct BackwardPass<int16_t>;
//template
//struct ForwardPass<half>;
template
struct ForwardPass<float>;
template
struct ForwardPass<double>;

}  // namespace gru
