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


__device__ __forceinline__ int8_t computeZ( // 更新门z
    const int channel_idx,
    const int32_t Wx_val,   // Wx 对应门的值
    const int32_t Rh_val,   // Rh 对应门的值
    const int32_t W_sum_mul_x_zp,
    const int32_t R_sum_mul_h_zp,
    const int32_t bx_val,   // bx 对应门的bias
    const int32_t br_val,   // br 对应门的bias
    const QuantGRUReScale &rescale_params
) {
    // z = sigmoid(Wx[z_idx] + Rh[z_idx] + bx[bz_idx] + br[bz_idx]);

    const int32_t Wx = rshift_round(Wx_val, rescale_params.n_Wx_[channel_idx]) - W_sum_mul_x_zp + rescale_params.zp_Wx_;
    const int32_t Rh = rshift_round(Rh_val, rescale_params.n_Rh_[channel_idx]) - R_sum_mul_h_zp + rescale_params.zp_Rh_;

    // scale_z_pre是通过效验阶段得到的; 通过sigmoid函数入口前的各项相加:Wx_val+Rh_val+bx_val+br_val的结果的的最大最小值计算得到
    const int8_t z_pre_i8 = dev::clamp<int8_t>( // clamp: 截断到int8的范围
        rshift_round(Wx, rescale_params.n_Wx_to_z_) + // n为: scale_Wx / scale_z_pre ≈ 2^-n
        rshift_round(Rh, rescale_params.n_Rh_to_z_) + // n为: scale_Rh / scale_z_pre ≈ 2^-n
        rshift_round(bx_val, rescale_params.n_bx_to_z_[channel_idx]) + // n为: scale_bx / scale_z_pre ≈ 2^-n; bx为X的偏置
        rshift_round(br_val, rescale_params.n_br_to_z_[channel_idx]) + // n为: scale_br / scale_z_pre ≈ 2^-n; br为R的偏置
        rescale_params.zp_z_pre_);
    return dev::sigmoid_int8_lut(z_pre_i8, d_sigmoid_int8_z_lut);
}

__device__ __forceinline__ int8_t computeR( // 重置门r
    const int channel_idx,
    const int32_t Wx_val,   // Wx 对应门的值
    const int32_t Rh_val,   // Rh 对应门的值
    const int32_t W_sum_mul_x_zp,
    const int32_t R_sum_mul_h_zp,
    const int32_t bx_val,   // bx 对应门的bias
    const int32_t br_val,   // br 对应门的bias
    const QuantGRUReScale &rescale_params
) {
    // r = sigmoid(Wx[r_idx] + Rh[r_idx] + bx[br_idx] + br[br_idx]);

    const int32_t Wx = rshift_round(Wx_val, rescale_params.n_Wx_[channel_idx]) - W_sum_mul_x_zp + rescale_params.zp_Wx_;
    const int32_t Rh = rshift_round(Rh_val, rescale_params.n_Rh_[channel_idx]) - R_sum_mul_h_zp + rescale_params.zp_Rh_;

    // scale_z_pre是通过效验阶段得到的; 通过sigmoid函数入口前的各项相加:Wx_val+Rh_val+bx_val+br_val的结果的的最大最小值计算得到
    const int8_t r_pre_i8 = dev::clamp<int8_t>( // clamp: 截断到int8的范围
        rshift_round(Wx, rescale_params.n_Wx_to_r_) +
        // n为: (scale_W * scale_x) / scale_r_pre ≈ 2^-n
        rshift_round(Rh, rescale_params.n_Rh_to_r_) +
        // n为: (scale_R * scale_h) / scale_r_pre ≈ 2^-n
        rshift_round(bx_val, rescale_params.n_bx_to_r_[channel_idx]) + // n为: scale_bx / scale_r_pre ≈ 2^-n; bx为X的偏置
        rshift_round(br_val, rescale_params.n_br_to_r_[channel_idx]) + // n为: scale_br / scale_r_pre ≈ 2^-n; br为R的偏置
        rescale_params.zp_r_pre_);
    return dev::sigmoid_int8_lut(r_pre_i8, d_sigmoid_int8_r_lut);
}

__device__ __forceinline__ int8_t computeG( // New Gate
    const int channel_idx,
    const int32_t Wx_val,   // Wx 对应门的值
    const int32_t Rh_val,   // Rh 对应门的值
    const int32_t W_sum_mul_x_zp,
    const int32_t R_sum_mul_h_zp,
    const int32_t bx_val,   // bx 对应门的bias
    const int32_t br_val,   // br 对应门的bias
    const int32_t r,
    const QuantGRUReScale &rescale_params
) {
    //  g = tanh (Wx[g_idx] + r * (Rh[g_idx] + br[bg_idx]) + bx[bg_idx]);

    const int32_t Wx = rshift_round(Wx_val, rescale_params.n_Wx_[channel_idx]) - W_sum_mul_x_zp + rescale_params.zp_Wx_;
    const int32_t Rh = rshift_round(Rh_val, rescale_params.n_Rh_[channel_idx]) - R_sum_mul_h_zp + rescale_params.zp_Rh_;
    const int32_t Rh_add_br = rshift_round(Rh, rescale_params.n_Rh_to_Rh_add_br_) +
                              rshift_round(bx_val, rescale_params.n_br_to_Rh_add_br_[channel_idx])
    /*+ rescale_params.zp_Rh_add_br_*/;

    const int32_t rRh = (r - rescale_params.zp_r_out_) *
                        (Rh_add_br/* - rescale_params.zp_Rh_add_br_*/)
                        /*+ rescale_params.zp_r_pre_*/;

    // 累加求和
    const int8_t g_pre_i8 = dev::clamp<int8_t>(
        rshift_round(Wx, rescale_params.n_Wx_to_g_[channel_idx]) +
        rshift_round(rRh/* - rescale_params.zp_r_pre_*/, rescale_params.n_rRh_to_g_[channel_idx]) +
        rshift_round(bx_val, rescale_params.n_bx_to_g_[channel_idx]) +
        rescale_params.zp_g_pre_);

    return dev::tanh_int8_lut(g_pre_i8, d_tanh_int8_g_lut);
}

__device__ __forceinline__ int8_t computeH( // 最终h
    int8_t z,
    int8_t g,
    int8_t h_old,
    const QuantGRUReScale &rescale_params
) {
    // T cur_h_value = z * h[output_idx] + (static_cast<T>(1.0) - z) * g;

    const int32_t old_contrib =
        rshift_round((z - rescale_params.zp_z_out_) * (h_old - rescale_params.zp_h_), rescale_params.n14_) +
        rescale_params.zp_old_contrib_;

    const int32_t one_minus_update = rescale_params.c12_ - rshift_round(z, rescale_params.n_one_minus_update_);
    const int32_t new_contrib =
        rshift_round(one_minus_update * (g - rescale_params.zp_g_out_), rescale_params.n13_) +
        rescale_params.zp_new_contrib_;

    return dev::clamp<int8_t>(
        rshift_round(new_contrib, rescale_params.n15_) +
        rshift_round(old_contrib, rescale_params.n16_) +
        rescale_params.c15_);
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
    const int32_t *W_sum_mul_x_zp, // hidden_size * 3
    const int32_t *R_sum_mul_h_zp, // hidden_size * 3
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
        z = computeZ(row,
                     Wx[z_idx],
                     Rh[z_idx],
                     W_sum_mul_x_zp[b_z_idx],
                     R_sum_mul_h_zp[b_z_idx],
                     bx[b_z_idx],
                     br[b_z_idx],
                     re_scale_param); // 更新门z

        r = computeR(row,
                     Wx[r_idx],
                     Rh[r_idx],
                     W_sum_mul_x_zp[b_r_idx],
                     R_sum_mul_h_zp[b_r_idx],
                     bx[b_r_idx],
                     br[b_r_idx],
                     re_scale_param); // 重置门r

        g = computeG(row,
                     Wx[g_idx],
                     Rh[g_idx],
                     W_sum_mul_x_zp[b_g_idx],
                     R_sum_mul_h_zp[b_g_idx],
                     bx[b_g_idx],
                     br[b_g_idx],
                     r,
                     re_scale_param);
        // 候选状态~ht
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

    T cur_h_value = computeH(z, g, h[output_idx], re_scale_param);

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
    const T *R,   // [H,H*3]
    const int32_t *bx,  // [H*3]
    const int32_t *br,  // [H*3]
    const T *h,   // [N,H]
    T *h_out,     // [N,H]
    T *v,         // [N,H*4]
    const int32_t *tmp_Wx,    // [N,H*3]
    int32_t *tmp_Rh,    // [N,H*3]
    const int *W_sum_mul_x_zp, // hidden_size * 3
    const int *R_sum_mul_h_zp, // hidden_size * 3
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
                batch_size, hidden_size, tmp_Wx, tmp_Rh, W_sum_mul_x_zp, R_sum_mul_h_zp, bx, br, h, h_out, v,
                    zoneout_prob, zoneout_mask, rescale_param_);
        } else {
            kernel::PointwiseOperationsQuant<T, true, false><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, W_sum_mul_x_zp, R_sum_mul_h_zp, bx, br, h, h_out, v, 0.0f,
                    nullptr, rescale_param_);
        }
    } else { // 推理模式
        if (zoneout_prob && zoneout_mask) {
            kernel::PointwiseOperationsQuant<T, false, true><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, W_sum_mul_x_zp, R_sum_mul_h_zp, bx, br, h, h_out, nullptr,
                    zoneout_prob, zoneout_mask, rescale_param_);
        } else {
            kernel::PointwiseOperationsQuant<T, false, false><<<gridDim, blockDim, 0, stream1>>>(
                batch_size, hidden_size, tmp_Wx, tmp_Rh, W_sum_mul_x_zp, R_sum_mul_h_zp, bx, br, h, h_out, nullptr,
                    0.0f, nullptr, rescale_param_);
        }
    }
}

template<typename T>
void ForwardPassQuant<T>::setRescaleParam(const GRUQuantitativeParameters &parms) {
    const int hidden = parms.hidden_;

    // z门
    std::vector<int32_t> n_Wx_to_z(hidden); // size = hidden (channel)
    std::vector<int32_t> n_Rh_to_z(hidden);
    std::vector<int32_t> n_bx_to_z(hidden);
    std::vector<int32_t> n_br_to_z(hidden);

    // r门
    std::vector<int32_t> n_Wx_to_r(hidden);
    std::vector<int32_t> n_Rh_to_r(hidden);
    std::vector<int32_t> n_bx_to_r(hidden);
    std::vector<int32_t> n_br_to_r(hidden);

    // n门
    std::vector<int32_t> n_Rh_to_Rh_add_br(hidden);
    std::vector<int32_t> n_br_to_Rh_add_br(hidden);
    std::vector<int32_t> n_Wx_to_g(hidden);
    std::vector<int32_t> n_rRh_to_g(hidden);
    std::vector<int32_t> n_bx_to_g(hidden);

    for (int idx = 0; idx < hidden; ++idx) { // per-channel
        const float scale_Wx = parms.scale_Wx_;
        const float scale_Rh = parms.scale_Rh_;
        // z门
        n_Wx_to_z[idx] = calculate_right_shift_bits(scale_Wx / parms.scale_z_pre_);
        n_Rh_to_z[idx] = calculate_right_shift_bits(scale_Rh / parms.scale_z_pre_);
        n_bx_to_z[idx] = calculate_right_shift_bits(parms.scale_bx_[idx] / parms.scale_z_pre_);
        n_br_to_z[idx] = calculate_right_shift_bits(parms.scale_br_[idx] / parms.scale_z_pre_);

        // r门
        n_Wx_to_r[idx] = calculate_right_shift_bits(scale_Wx / parms.scale_r_pre_);
        n_Rh_to_r[idx] = calculate_right_shift_bits(scale_Rh / parms.scale_r_pre_);
        n_bx_to_r[idx] = calculate_right_shift_bits(parms.scale_bx_[idx] / parms.scale_r_pre_);
        n_br_to_r[idx] = calculate_right_shift_bits(parms.scale_br_[idx] / parms.scale_r_pre_);

        // n门
        n_Rh_to_Rh_add_br[idx] = calculate_right_shift_bits(scale_Rh / parms.scale_Rh_add_br_[idx]);
        n_br_to_Rh_add_br[idx] = calculate_right_shift_bits(parms.scale_br_[idx] / parms.scale_Rh_add_br_[idx]);
        n_Wx_to_g[idx] = calculate_right_shift_bits(scale_Wx / parms.scale_g_pre_);
        n_rRh_to_g[idx] = calculate_right_shift_bits(parms.scale_rRh_[idx] / parms.scale_g_pre_);
        n_bx_to_g[idx] = calculate_right_shift_bits(parms.scale_bx_[idx] / parms.scale_g_pre_);
    }

    /* init */
    rescale_param_.zp_x_ = parms.zp_x_;
    rescale_param_.zp_h_ = parms.zp_h_;

    // z门
    rescale_param_.zp_z_pre_ = parms.zp_z_pre_;
    rescale_param_.zp_z_out_ = parms.zp_z_out_;
    h2d(rescale_param_.n_Wx_to_z_, n_Wx_to_z);
    h2d(rescale_param_.n_Rh_to_z_, n_Rh_to_z);
    h2d(rescale_param_.n_bx_to_z_, n_bx_to_z);
    h2d(rescale_param_.n_br_to_z_, n_br_to_z);

    // r门
    rescale_param_.zp_r_pre_ = parms.zp_r_pre_;
    rescale_param_.zp_r_out_ = parms.zp_r_out_;
    h2d(rescale_param_.n_Wx_to_r_, n_Wx_to_r);
    h2d(rescale_param_.n_Rh_to_r_, n_Rh_to_r);
    h2d(rescale_param_.n_bx_to_r_, n_bx_to_r);
    h2d(rescale_param_.n_br_to_r_, n_br_to_r);

    // n门
    rescale_param_.zp_g_pre_ = parms.zp_g_pre_;
    rescale_param_.zp_g_out_ = parms.zp_g_out_;
    h2d(rescale_param_.n_Rh_to_Rh_add_br_, n_Rh_to_Rh_add_br);
    h2d(rescale_param_.n_br_to_Rh_add_br_, n_br_to_Rh_add_br);
    h2d(rescale_param_.n_Wx_to_g_, n_Wx_to_g);
    h2d(rescale_param_.n_rRh_to_g_, n_rRh_to_g);
    h2d(rescale_param_.n_bx_to_g_, n_bx_to_g);

    // h_new
    rescale_param_.n_one_minus_update_ = calculate_right_shift_bits(parms.scale_one_minus_update_);
    rescale_param_.c12_ =
        calculate_one_over_Somu(parms.scale_one_minus_update_) +
        rshift_round(parms.zp_z_out_,
                     calculate_right_shift_bits(parms.scale_z_out_ / parms.scale_one_minus_update_)); // 不需要加zp_omu
    rescale_param_.zp_new_contrib_ = parms.zp_new_contrib_;
    rescale_param_.n13_ = calculate_right_shift_bits(
        (parms.scale_one_minus_update_ * parms.scale_g_out_) / parms.scale_new_contrib_);
    rescale_param_.n14_ = calculate_right_shift_bits((parms.scale_z_out_ * parms.scale_h_) / parms.scale_old_contrib_);
    rescale_param_.zp_old_contrib_ = parms.zp_old_contrib_;
    rescale_param_.n15_ = calculate_right_shift_bits(parms.scale_new_contrib_ / parms.scale_h_);
    rescale_param_.n16_ = calculate_right_shift_bits(parms.scale_old_contrib_ / parms.scale_h_);
    rescale_param_.c15_ = parms.zp_h_ - (rshift_round(parms.zp_new_contrib_, rescale_param_.n15_) +
                                         rshift_round(parms.zp_old_contrib_, rescale_param_.n16_));
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
    const cudaStream_t stream1 = data_->stream[1];
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

    // 计算W_sum_mul_zp用于补偿x_zp
    dev::vector<int32_t> W_sum_mul_x_zp(hidden_size * 3);
    computeWeightSumMulzp(W, W_sum_mul_x_zp.data(), rescale_param_.zp_x_, W_sum_mul_x_zp.size(), input_size, stream2);

    // Rh的gemm需要补偿h_zp, 所以提前计算 h_zp * R_sum * h_zp, stream1
    dev::vector<int32_t> R_sum_mul_h_zp(hidden_size * 3);
    computeWeightSumMulzp(R, R_sum_mul_h_zp.data(), rescale_param_.zp_h_, R_sum_mul_h_zp.size(), input_size, stream1);

    // 同步Wx计算
    cudaEventRecord(event, stream2);

    // 同步R_sum_mul_h_zp计算
    cudaEventRecord(event, stream1);

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

        IterateInternal(R, bx, br, h + i * NH, h + (i + 1) * NH, v + i * NH * 4,
                        tmp_Wx + i * NH * 3, tmp_Rh, W_sum_mul_x_zp.data(), R_sum_mul_h_zp.data(), zoneout_prob,
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
