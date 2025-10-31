#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cstdio>

#include "blas.h"
#include "device_assert.h"
#include "gru.h"
#include "inline_ops.h"

namespace {

template<typename T, typename AccumT, bool ApplyZoneout>
__global__
void PointwiseOperations(const int batch_dim,
                         const int hidden_dim,
                         const T *h,
                         const T *v,
                         const T *dh_new,
                         AccumT *dbx_out,
                         AccumT *dbr_out,
                         T *dh_inout,
                         T *dp_out,
                         T *dq_out,
                         const T *zoneout_mask) {  // Zoneout mask (only used if ApplyZoneout==true)
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= hidden_dim || col >= batch_dim)
        return;

    const int base_idx = col * hidden_dim + row;

    T dh_total = dh_new[base_idx] + dh_inout[base_idx];

    const int stride4_base_idx = col * (hidden_dim * 4) + row;
    const int z_idx = stride4_base_idx + 0 * hidden_dim;
    const int r_idx = stride4_base_idx + 1 * hidden_dim;
    const int g_idx = stride4_base_idx + 2 * hidden_dim;
    const int q_g_idx = stride4_base_idx + 3 * hidden_dim;

    const T z = v[z_idx];
    const T r = v[r_idx];
    const T g = v[g_idx];
    const T q_g = v[q_g_idx];

    if (ApplyZoneout) {
        const T mask = zoneout_mask[base_idx];
        dh_inout[base_idx] = (static_cast<T>(1.0) - mask) * dh_total;
        dh_total = mask * dh_total;
        dh_inout[base_idx] += z * dh_total;
    } else {
        dh_inout[base_idx] = z * dh_total;
    }

    const T dg = (static_cast<T>(1.0) - z) * dh_total;
    const T dz = (h[base_idx] - g) * dh_total;
    const T dp_g = d_tanh(g) * dg;
    const T dq_g = dp_g * r;
    const T dr = dp_g * q_g;
    const T dp_r = d_sigmoid(r) * dr;
    const T dq_r = dp_r;
    const T dp_z = d_sigmoid(z) * dz;
    const T dq_z = dp_z;

    const int idx = col * (hidden_dim * 3) + row;

    dp_out[idx + 0 * hidden_dim] = dp_z;
    dp_out[idx + 1 * hidden_dim] = dp_r;
    dp_out[idx + 2 * hidden_dim] = dp_g;

    dq_out[idx + 0 * hidden_dim] = dq_z;
    dq_out[idx + 1 * hidden_dim] = dq_r;
    dq_out[idx + 2 * hidden_dim] = dq_g;

    atomicAdd(&dbx_out[row + 0 * hidden_dim], dp_z);
    atomicAdd(&dbx_out[row + 1 * hidden_dim], dp_r);
    atomicAdd(&dbx_out[row + 2 * hidden_dim], dp_g);

    atomicAdd(&dbr_out[row + 0 * hidden_dim], dq_z);
    atomicAdd(&dbr_out[row + 1 * hidden_dim], dq_r);
    atomicAdd(&dbr_out[row + 2 * hidden_dim], dq_g);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)

template<typename T, bool ApplyZoneout>
__global__
void PointwiseOperations(const int batch_dim,
                         const int hidden_dim,
                         const half *h,
                         const half *v,
                         const half *dh_new,
                         half *dbx_out,
                         half *dbr_out,
                         half *dh_inout,
                         half *dp_out,
                         half *dq_out,
                         const half *zoneout_mask) {
    device_assert_fail("FP16 is not supported on compute capability < 7.0.");
}

#endif

}  // anonymous namespace

namespace gru {

template<typename T, typename AccumT>
struct BackwardPass<T, AccumT>::private_data {
  int batch_size;
  int input_size;
  int hidden_size;
  cublasHandle_t blas_handle;
  cudaStream_t stream[2];
  cudaEvent_t event;
  cudaStream_t sync_stream;
};

template<typename T, typename AccumT>
BackwardPass<T, AccumT>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t &blas_handle,
    const cudaStream_t &stream) : data_(new private_data) {
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
BackwardPass<T, AccumT>::~BackwardPass() {
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
void BackwardPass<T, AccumT>::Iterate(
    const T *W_t,     // [H*3,C]
    const T *R_t,     // [H*3,H]
    const AccumT *bx,      // [H*3]
    const AccumT *br,      // [H*3]
    const T *x_t,     // [C,N]
    const T *h,       // [N,H]
    const T *v,       // [N,H*4]
    const T *dh_new,  // [N,H]
    T *dx,            // [N,C]
    T *dW,            // [C,H*3]
    T *dR,            // [H,H*3]
    AccumT *dbx,           // [H*3]
    AccumT *dbr,           // [H*3]
    T *dh,            // [N,H]
    T *dp,            // [N,H*3]
    T *dq,            // [N,H*3]
    const T *zoneout_mask) {  // [N,H]
    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    using alpha_beta_t = std::conditional_t<
        std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>,
        int,
        T>;

    static const alpha_beta_t alpha = static_cast<alpha_beta_t>(1.0);
    static const alpha_beta_t beta_sum = static_cast<alpha_beta_t>(1.0);
    static const alpha_beta_t beta_assign = static_cast<alpha_beta_t>(0.0);

    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const int input_size = data_->input_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaStream_t stream2 = data_->stream[1];
    const cudaEvent_t event = data_->event;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    IterateInternal(
        R_t,
        h,
        v,
        dh_new,
        dbx,
        dbr,
        dh,
        dp,
        dq,
        zoneout_mask);

    cublasSetStream(blas_handle, stream1);
    blas<T>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  hidden_size * 3, input_size, batch_size,
                  &alpha,
                  dp, hidden_size * 3,
                  x_t, batch_size,
                  &beta_sum,
                  dW, hidden_size * 3);

    // Wait for pointwise operations to complete since there's a
    // data dependency between its output (`dp`, `dq`) and the following matmuls.
    cudaStreamWaitEvent(stream2, event, 0);

    cublasSetStream(blas_handle, stream2);
    blas<T>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  input_size, batch_size, hidden_size * 3,
                  &alpha,
                  W_t, input_size,
                  dp, hidden_size * 3,
                  &beta_assign,
                  dx, input_size);

    cublasSetStream(blas_handle, stream2);
    blas<T>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_T,
                  hidden_size * 3, hidden_size, batch_size,
                  &alpha,
                  dq, hidden_size * 3,
                  h, hidden_size,
                  &beta_sum,
                  dR, hidden_size * 3);

    cublasSetStream(blas_handle, save_stream);
}

template<typename T, typename AccumT>
void BackwardPass<T, AccumT>::IterateInternal( // 内部迭代
    const T *R_t,     // [H*3,H]
    const T *h,       // [N,H]
    const T *v,       // [N,H*4]
    const T *dh_new,  // [N,H]
    AccumT *dbx,           // [H*3]
    AccumT *dbr,           // [H*3]
    T *dh,            // [N,H]
    T *dp,            // [N,H*3]
    T *dq,            // [N,H*3]
    const T *zoneout_mask) {  // [N,H]

    using alpha_beta_t = std::conditional_t<
        std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>,
        int,
        T>;

    static const alpha_beta_t alpha = static_cast<alpha_beta_t>(1.0);
    static const alpha_beta_t beta_sum = static_cast<alpha_beta_t>(1.0);

    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaEvent_t event = data_->event;

    // Compute launch configuration for pointwise operations kernel.
    const dim3 blockDim(32, 16);
    const dim3 gridDim(
        (hidden_size + blockDim.x - 1) / blockDim.x,
        (batch_size + blockDim.y - 1) / blockDim.y);

    if (zoneout_mask) {
        PointwiseOperations<T, AccumT, true><<<gridDim, blockDim, 0, stream1>>>(
            batch_size,
            hidden_size,
            h,
            v,
            dh_new,
            dbx,
            dbr,
            dh,
            dp,
            dq,
            zoneout_mask
        );
    } else {
        PointwiseOperations<T, AccumT, false><<<gridDim, blockDim, 0, stream1>>>(
            batch_size,
            hidden_size,
            h,
            v,
            dh_new,
            dbx,
            dbr,
            dh,
            dp,
            dq,
            nullptr
        );
    }
    cudaEventRecord(event, stream1);

    printf("cudaError(BackwardPass:IterateInternal:gemm before): %s\n", cudaGetErrorString(cudaGetLastError()));

    cublasSetStream(blas_handle, stream1);
    blas<T>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  hidden_size, batch_size, hidden_size * 3,
                  &alpha,
                  R_t, hidden_size,
                  dq, hidden_size * 3,
                  &beta_sum,
                  dh, hidden_size);

    printf("cudaError(BackwardPass:IterateInternal:gemm after): %s\n", cudaGetErrorString(cudaGetLastError()));
}

template<typename T, typename AccumT>
void BackwardPass<T, AccumT>::Run(
    const int steps,
    const T *W_t, // 输入权重转置
    const T *R_t, // 循环权重转置
    const AccumT *bx, // 输入bias
    const AccumT *br, // 循环bias
    const T *x_t, // 当前步输入
    const T *h, // 前一时刻隐藏状态
    const T *v, // 前向传播中间结果(含z, r,g,h_out)
    const T *dh_new, // 从后面时间步传来的梯度
    T *dx, //
    T *dW,
    T *dR,
    AccumT *dbx,
    AccumT *dbr,
    T *dh,
    T *dp,
    T *dq,
    const T *zoneout_mask) {
    const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    using alpha_beta_t = std::conditional_t<
        std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>,
        int,
        T>;

    const alpha_beta_t alpha = static_cast<alpha_beta_t>(1.0);
    const alpha_beta_t beta_sum = static_cast<alpha_beta_t>(1.0);
    const alpha_beta_t beta_assign = static_cast<alpha_beta_t>(0.0);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaStream_t stream2 = data_->stream[1];
    const cudaEvent_t event = data_->event;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    const int NH = batch_size * hidden_size;
    for (int i = steps - 1; i >= 0; --i) { // 反向迭代
        IterateInternal(
            R_t,
            h + i * NH,
            v + i * NH * 4,
            dh_new + (i + 1) * NH,
            dbx,
            dbr,
            dh,
            dp + i * NH * 3,
            dq + i * NH * 3,
            zoneout_mask ? zoneout_mask + i * NH : nullptr);
    }

    // Wait for pointwise operations to complete since there's a
    // data dependency between its output (`dp`, `dq`) and the following matmuls.
    cudaStreamWaitEvent(stream2, event, 0);

    printf("cudaError(BackwardPass): %s\n", cudaGetErrorString(cudaGetLastError()));

    cublasSetStream(blas_handle, stream2);
    blas<T>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  input_size, batch_size * steps, hidden_size * 3,
                  &alpha,
                  W_t, input_size,
                  dp, hidden_size * 3,
                  &beta_assign,
                  dx, input_size);

    cublasSetStream(blas_handle, stream2);
    blas<T>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_T,
                  hidden_size * 3, hidden_size, batch_size * steps,
                  &alpha,
                  dq, hidden_size * 3,
                  h, hidden_size,
                  &beta_sum,
                  dR, hidden_size * 3);

    cublasSetStream(blas_handle, stream1);
    blas<T>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  hidden_size * 3, input_size, batch_size * steps,
                  &alpha,
                  dp, hidden_size * 3,
                  x_t, batch_size * steps,
                  &beta_sum,
                  dW, hidden_size * 3);

    cublasSetStream(blas_handle, save_stream);
}

template
struct BackwardPass<int8_t>;
template
struct BackwardPass<int16_t>;
//template
//struct BackwardPass<half>;
template
struct BackwardPass<float>;
template
struct BackwardPass<double>;

}  // namespace gru


