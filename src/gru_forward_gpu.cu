#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "blas.h"
#include "device_assert.h"
#include "gru.h"
#include "inline_ops.h"

namespace {

template<typename T, bool Training, bool ApplyZoneout, bool Calibration = false>
__global__
void PointwiseOperations(const int batch_dim,
                         const int hidden_dim,
                         const T *Wx,
                         const T *Rh,
                         const T *bx,
                         const T *br,
                         const T *h,
                         T *h_out,
                         T *v,
                         const T zoneout_prob,
                         const T *zoneout_mask
//                         ,
//                         T *z_pres,
//                         T *r_pres,
//                         T *g_pres
) {  // Zoneout mask (only used if ApplyZoneout==true)
    const int row = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程对应的隐藏单元
    const int col = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程对应的batch样本

    if (row >= hidden_dim || col >= batch_dim)
        return;

    const int weight_idx = col * (hidden_dim * 3) + row;

    // Index into the `h` and `h_out` vectors (they have a stride of `hidden_dim`).
    const int output_idx = col * hidden_dim + row;

    // Indicies into the Wx and Rh matrices (for each of the u, r, and e components).
    const int z_idx = weight_idx + 0 * hidden_dim;
    const int r_idx = weight_idx + 1 * hidden_dim;
    const int g_idx = weight_idx + 2 * hidden_dim;

    // Indices into the bias vectors (for each of the u, r, and e components).
    const int bz_idx = row + 0 * hidden_dim;
    const int br_idx = row + 1 * hidden_dim;
    const int bg_idx = row + 2 * hidden_dim;

    const T z_pre = Wx[z_idx] + Rh[z_idx] + bx[bz_idx] + br[bz_idx];
    const T z = sigmoid(z_pre);

    const T r_pre = Wx[r_idx] + Rh[r_idx] + bx[br_idx] + br[br_idx];
    const T r = sigmoid(r_pre);

    const T g_pre = Wx[g_idx] + r * (Rh[g_idx] + br[bg_idx]) + bx[bg_idx];
    const T g = tanh(g_pre);

    if (Calibration) {
//        z_pres;
    }

    // Store internal activations if we're eventually going to backprop.
    if (Training) {
        const int base_v_idx = col * (hidden_dim * 4) + row;
        v[base_v_idx + 0 * hidden_dim] = z;
        v[base_v_idx + 1 * hidden_dim] = r;
        v[base_v_idx + 2 * hidden_dim] = g;
        v[base_v_idx + 3 * hidden_dim] = Rh[g_idx] + br[bg_idx];
    }

    T cur_h_value = z * h[output_idx] + (static_cast<T>(1.0) - z) * g;

    if (ApplyZoneout) {
        if (Training) {
            cur_h_value = (cur_h_value - h[output_idx]) * zoneout_mask[output_idx] + h[output_idx];
        } else {
            cur_h_value = (zoneout_prob * h[output_idx]) + ((static_cast<T>(1.0) - zoneout_prob) * cur_h_value);
        }
    }

    h_out[output_idx] = cur_h_value;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)

template<typename T, bool Training, bool ApplyZoneout, bool Calibration = false>
__global__
void PointwiseOperations(const int batch_dim,
                         const int hidden_dim,
                         const half *Wx,
                         const half *Rh,
                         const half *bx,
                         const half *br,
                         const half *h,
                         half *h_out,
                         half *v,
                         const half zoneout_prob,
                         const half *zoneout_mask) {
    device_assert_fail("FP16 is not supported on compute capability < 7.0.");
}

#endif

}  // anonymous namespace

namespace gru {

template<typename T>
struct ForwardPass<T>::private_data {
  bool calibration;
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
ForwardPass<T>::ForwardPass(
    const bool calibration,
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t &blas_handle,
    const cudaStream_t &stream) : data_(new private_data) {
    data_->calibration = calibration;
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
void ForwardPass<T>::Iterate(
    const T *W,  // [C,H*3]
    const T *R,  // [H,H*3]
    const T *bx, // [H*3]
    const T *br, // [H*3]
    const T *x,  // [N,C]
    const T *h,  // [N,H]
    T *h_out,    // [N,H]
    T *v,        // [N,H*4]
    T *tmp_Wx,   // [N,H*3]
    T *tmp_Rh,   // [N,H*3]
    const float zoneout_prob,
    const T *zoneout_mask) { // Zoneout mask [N,H]
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

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
    blas<T>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  hidden_size * 3, batch_size, input_size,
                  &alpha,
                  W, hidden_size * 3,
                  x, input_size,
                  &beta,
                  tmp_Wx, hidden_size * 3);
    cudaEventRecord(event, stream2);

    IterateInternal(
        R,
        bx,
        br,
        h,
        h_out,
        v,
        tmp_Wx,
        tmp_Rh,
        zoneout_prob,
        zoneout_mask);

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
void ForwardPass<T>::IterateInternal(
    const T *R,  // [H,H*3]
    const T *bx, // [H*3]
    const T *br, // [H*3]
    const T *h,  // [N,H]
    T *h_out,    // [N,H]
    T *v,        // [N,H*4]
    T *tmp_Wx,   // [N,H*3]
    T *tmp_Rh,   // [N,H*3]
    const float zoneout_prob,
    const T *zoneout_mask) { // Zoneout mask [N,H]
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
    blas<T>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  hidden_size * 3, batch_size, hidden_size,
                  &alpha,
                  R, hidden_size * 3,
                  h, hidden_size,
                  &beta,
                  tmp_Rh, hidden_size * 3);

    // Compute launch configuration for pointwise operations kernel.
    const dim3 blockDim(32, 16);
    const dim3 gridDim(
        (hidden_size + blockDim.x - 1) / blockDim.x,
        (batch_size + blockDim.y - 1) / blockDim.y);

    cudaStreamWaitEvent(stream1, event, 0);

    if (training) {
        if (zoneout_prob && zoneout_mask) {
            PointwiseOperations<T, true, true><<<gridDim, blockDim, 0, stream1>>>(
                batch_size,
                hidden_size,
                tmp_Wx,
                tmp_Rh,
                bx,
                br,
                h,
                h_out,
                v,
                zoneout_prob,
                zoneout_mask);
        } else {
            PointwiseOperations<T, true, false><<<gridDim, blockDim, 0, stream1>>>(
                batch_size,
                hidden_size,
                tmp_Wx,
                tmp_Rh,
                bx,
                br,
                h,
                h_out,
                v,
                0.0f,
                nullptr);
        }
    } else {
        if (zoneout_prob && zoneout_mask) {
            PointwiseOperations<T, false, true><<<gridDim, blockDim, 0, stream1>>>(
                batch_size,
                hidden_size,
                tmp_Wx,
                tmp_Rh,
                bx,
                br,
                h,
                h_out,
                nullptr,
                zoneout_prob,
                zoneout_mask);
        } else {
            PointwiseOperations<T, false, false><<<gridDim, blockDim, 0, stream1>>>(
                batch_size,
                hidden_size,
                tmp_Wx,
                tmp_Rh,
                bx,
                br,
                h,
                h_out,
                nullptr,
                0.0f,
                nullptr);
        }
    }
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T *W,  // [C,H*3]
    const T *R,  // [H,H*3]
    const T *bx, // [H*3]
    const T *br, // [H*3]
    const T *x,  // [N,C]
    T *h,        // [N,H]
    T *v,        // [N,H*4]
    T *tmp_Wx,   // [N,H*3]
    T *tmp_Rh,   // [N,H*3]
    const float zoneout_prob,
    const T *zoneout_mask) { // Zoneout mask [N,H]
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
    blas<T>::gemm(blas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  hidden_size * 3, steps * batch_size, input_size,
                  &alpha,
                  W, hidden_size * 3,
                  x, input_size,
                  &beta,
                  tmp_Wx, hidden_size * 3);
    cudaEventRecord(event, stream2);

    const int NH = batch_size * hidden_size;
    for (int i = 0; i < steps; ++i) {
        const int Rh_offset = data_->calibration ? i * NH * 3 : 0;
        IterateInternal(
            R,
            bx,
            br,
            h + i * NH,
            h + (i + 1) * NH,
            v + i * NH * 4,
            tmp_Wx + i * NH * 3,
            tmp_Rh + Rh_offset,
            zoneout_prob,
            zoneout_mask ? zoneout_mask + i * NH : nullptr);
    }

    cublasSetStream(blas_handle, save_stream);
}

template
struct ForwardPass<half>;
template
struct ForwardPass<float>;
template
struct ForwardPass<double>;

}  // namespace gru
