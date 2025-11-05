#include <Eigen/Dense>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

#include "device_ptr.h"
#include "gru.h"

using Tensor1f = Eigen::Tensor<float, 1>;
using Tensor2f = Eigen::Tensor<float, 2>;
using Tensor3f = Eigen::Tensor<float, 3>;
using Tensor1i8 = Eigen::Tensor<int8_t, 1>;
using Tensor2i8 = Eigen::Tensor<int8_t, 2>;
using Tensor3i8 = Eigen::Tensor<int8_t, 3>;
using Tensor1i16 = Eigen::Tensor<int16_t, 1>;
using Tensor2i16 = Eigen::Tensor<int16_t, 2>;
using Tensor3i16 = Eigen::Tensor<int16_t, 3>;
using Tensor1i32 = Eigen::Tensor<int32_t, 1>;
using Tensor2i32 = Eigen::Tensor<int32_t, 2>;
using Tensor3i32 = Eigen::Tensor<int32_t, 3>;

constexpr int BATCH_SIZE = 64; // 批大小
constexpr int SEQUENCE_LEN = 1000; // 序列长度(T), 每个样本有T个时间步
constexpr int HIDDEN_DIMS = 512; // 隐藏层维度(H), h_t的维度
constexpr int INPUT_DIMS = 512; // 输入维度(I), x_t的维度

static cublasHandle_t g_blas_handle;

class ScopeTimer { // 测量时间类
 public:
  ScopeTimer(const std::string &msg) : msg_(msg) {
      cudaEventCreate(&start_);
      cudaEventCreate(&stop_);
      cudaDeviceSynchronize();
      cudaEventRecord(start_);
  }

  ~ScopeTimer() {
      float elapsed_ms;
      cudaEventRecord(stop_);
      cudaEventSynchronize(stop_);
      cudaEventElapsedTime(&elapsed_ms, start_, stop_);
      printf("%s %fms\n", msg_.c_str(), elapsed_ms);
      cudaEventDestroy(start_);
      cudaEventDestroy(stop_);
  }

 private:
  std::string msg_;
  cudaEvent_t start_, stop_;
};

void GruInference(
    const Tensor2f &W,
    const Tensor2f &R,
    const Tensor1f &bx,
    const Tensor1f &br,
    const Tensor3f &x) {
    const int time_steps = x.dimension(2);
    const int batch_size = x.dimension(1);
    const int input_size = x.dimension(0);
    const int hidden_size = R.dimension(1);

    // Copy weights over to GPU.
    device_ptr<Tensor2f> W_dev(W);
    device_ptr<Tensor2f> R_dev(R);
    device_ptr<Tensor1f> bx_dev(bx);
    device_ptr<Tensor1f> br_dev(br);
    device_ptr<Tensor3f> x_dev(x);

    device_ptr<Tensor2f> h_dev((time_steps + 1) * batch_size * hidden_size);
    device_ptr<Tensor3f> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3); // 用于存放W * x的中间结果
    device_ptr<Tensor2f> tmp_Rh_dev(batch_size * hidden_size * 3); // 用于存放R * h的中间结果

    h_dev.zero(); // h初始化为0

    ScopeTimer t("Inference:");

    gru::ForwardPass<float> forward = gru::ForwardPass<float>(
        false,  // training
        batch_size,
        input_size,
        hidden_size,
        g_blas_handle);

    // TODO: 计算scale和zero point
//    QuantParams quantParams;
//    calculateScaleZeroPoint(W_dev.data, W_dev.size,quantParams.s_x,zp_w);

    forward.Run(
        time_steps,
        W_dev.data,
        R_dev.data,
        bx_dev.data,
        br_dev.data,
        x_dev.data,
        h_dev.data,
        nullptr,
        tmp_Wx_dev.data,
        tmp_Rh_dev.data,
        0.0f,
        nullptr);
}

void GruTrain(
    const Tensor2f &W, // 输入到隐藏层的权重矩阵. [input_size, hidden_size * 3] 对应三个门
    const Tensor2f &R, // 隐藏层到隐藏层的循环权重矩阵
    const Tensor1f &bx, // 输入偏置项（input bias），来自输入路径
    const Tensor1f &br, // 循环偏置项（recurrent bias），来自循环路径
    const Tensor3f &x, // 输入序列张量
    const Tensor3f &dh_new, // 来自上层网络或损失函数的反向梯度. [hidden_size, batch_size, time_steps]
    bool enable_quantitative = false, // 是否启用量化推理模式
    bool use_int16 = false // 控制量化精度位宽
) {
    const int time_steps = x.dimension(2);
    const int batch_size = x.dimension(1);
    const int input_size = x.dimension(0);
    const int hidden_size = R.dimension(1);
    if (!enable_quantitative) { // 非量化
        // Copy weights over to GPU.
        device_ptr<Tensor2f> W_dev(W);
        device_ptr<Tensor2f> R_dev(R);
        device_ptr<Tensor1f> bx_dev(bx);
        device_ptr<Tensor1f> br_dev(br);
        device_ptr<Tensor3f> x_dev(x);
        device_ptr<Tensor3f> dh_new_dev(dh_new);

        device_ptr<Tensor2f> h_dev((time_steps + 1) * batch_size * hidden_size);
        device_ptr<Tensor3f> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
        device_ptr<Tensor2f> tmp_Rh_dev(batch_size * hidden_size * 3);
        device_ptr<Tensor3f> v_dev(time_steps * batch_size * hidden_size * 4);

        h_dev.zero();

        {
            ScopeTimer t("Train forward:");
            gru::ForwardPass<float> forward = gru::ForwardPass<float>(
                true,  // training
                batch_size,
                input_size,
                hidden_size,
                g_blas_handle);

            forward.Run(
                time_steps,
                W_dev.data,
                R_dev.data,
                bx_dev.data,
                br_dev.data,
                x_dev.data,
                h_dev.data,
                v_dev.data,
                tmp_Wx_dev.data,
                tmp_Rh_dev.data,
                0.0f,
                nullptr);
        }

        device_ptr<Tensor3f> dx_dev(time_steps * batch_size * input_size); // 输入序列梯度
        device_ptr<Tensor2f> dW_dev(input_size * hidden_size * 3); // 对输入权重的梯度
        device_ptr<Tensor2f> dR_dev(hidden_size * hidden_size * 3); // 对循环权重的梯度
        device_ptr<Tensor1f> dbx_dev(hidden_size * 3); // 对输入偏置的梯度
        device_ptr<Tensor1f> dbr_dev(hidden_size * 3); // 对循环偏置的梯度
        device_ptr<Tensor2f> dh_dev(batch_size * hidden_size); // 对最后隐藏状态的梯度
        device_ptr<Tensor3f> dp_dev(time_steps * batch_size * hidden_size * 3); // 临时缓存梯度（内部结构用）
        device_ptr<Tensor3f> dq_dev(time_steps * batch_size * hidden_size * 3); // 临时缓存梯度（内部结构用）

        {
            ScopeTimer t("Train backward:");
            gru::BackwardPass<float> backward(
                batch_size,
                input_size,
                hidden_size,
                g_blas_handle);

            backward.Run(
                time_steps,
                W_dev.data,
                R_dev.data,
                bx_dev.data,
                br_dev.data,
                x_dev.data,
                h_dev.data,
                v_dev.data,
                dh_new_dev.data,
                dx_dev.data,
                dW_dev.data,
                dR_dev.data,
                dbx_dev.data,
                dbr_dev.data,
                dh_dev.data,
                dp_dev.data,
                dq_dev.data,
                nullptr);
        }
    } else {
        if (!use_int16) { // int8量化
            GruQuantScales gruQuantScales = computeGruQuantParams<int8_t>(x.data(),
                                                                          time_steps,
                                                                          batch_size,
                                                                          input_size,
                                                                          W.data(),
                                                                          hidden_size,
                                                                          R.data(),
                                                                          bx.data(),
                                                                          br.data());

            std::vector<RescaleParam3> rescale_Wx(time_steps);
            computeWxRescaleParamsFixedShift(time_steps,
                                             gruQuantScales.x,
                                             gruQuantScales.W.gate[0].scale,
                                             gruQuantScales.W.gate[1].scale,
                                             gruQuantScales.W.gate[2].scale,
                                             rescale_Wx);

            // Copy weights over to GPU.
            device_ptr<Tensor2f> W_tmp_dev(W);
            device_ptr<Tensor2f> R_tmp_dev(R);
            device_ptr<Tensor1f> bx_tmp_dev(bx);
            device_ptr<Tensor1f> br_tmp_dev(br);
            device_ptr<Tensor3f> x_tmp_dev(x);
            device_ptr<Tensor3f> dh_new_tmp_dev(dh_new);

            device_ptr<Tensor2i8> W_dev(W.size());
            device_ptr<Tensor2i8> R_dev(R.size());
            device_ptr<Tensor1i32> bx_dev(bx.size());
            device_ptr<Tensor1i32> br_dev(br.size());
            device_ptr<Tensor3i8> x_dev(x.size());
            device_ptr<Tensor3i8> dh_new_dev(dh_new.size());


// -----------------------------
// 3. 量化 W
// -----------------------------
            quantizeFloatToInt<int8_t, true, true>(
                W_tmp_dev.data,
                W_dev.data,
                HIDDEN_DIMS * INPUT_DIMS,
                1.0f / gruQuantScales.W.gate[0].scale
            );

            quantizeFloatToInt<int8_t, true, true>(
                W_tmp_dev.data + HIDDEN_DIMS * INPUT_DIMS,
                W_dev.data + HIDDEN_DIMS * INPUT_DIMS,
                HIDDEN_DIMS * INPUT_DIMS,
                1.0f / gruQuantScales.W.gate[1].scale
            );

            quantizeFloatToInt<int8_t, true, true>(
                W_tmp_dev.data + 2 * HIDDEN_DIMS * INPUT_DIMS,
                W_dev.data + 2 * HIDDEN_DIMS * INPUT_DIMS,
                HIDDEN_DIMS * INPUT_DIMS,
                1.0f / gruQuantScales.W.gate[2].scale
            );

// -----------------------------
// 4. 量化 R
// -----------------------------
            quantizeFloatToInt<int8_t, true, true>(
                R_tmp_dev.data,
                R_dev.data,
                HIDDEN_DIMS * HIDDEN_DIMS,
                1.0f / gruQuantScales.R.gate[0].scale
            );

            quantizeFloatToInt<int8_t, true, true>(
                R_tmp_dev.data + HIDDEN_DIMS * HIDDEN_DIMS,
                R_dev.data + HIDDEN_DIMS * HIDDEN_DIMS,
                HIDDEN_DIMS * HIDDEN_DIMS,
                1.0f / gruQuantScales.R.gate[1].scale
            );

            quantizeFloatToInt<int8_t, true, true>(
                R_tmp_dev.data + 2 * HIDDEN_DIMS * HIDDEN_DIMS,
                R_dev.data + 2 * HIDDEN_DIMS * HIDDEN_DIMS,
                HIDDEN_DIMS * HIDDEN_DIMS,
                1.0f / gruQuantScales.R.gate[2].scale
            );

// -----------------------------
// 5. 量化 bx
// -----------------------------
            quantizeFloatToInt<int32_t, true, true, false>(
                bx_tmp_dev.data,
                bx_dev.data,
                HIDDEN_DIMS,
                1.0f / gruQuantScales.bx.gate[0].scale
            );

            quantizeFloatToInt<int32_t, true, true, false>(
                bx_tmp_dev.data + HIDDEN_DIMS,
                bx_dev.data + HIDDEN_DIMS,
                HIDDEN_DIMS,
                1.0f / gruQuantScales.bx.gate[1].scale
            );

            quantizeFloatToInt<int32_t, true, true, false>(
                bx_tmp_dev.data + 2 * HIDDEN_DIMS,
                bx_dev.data + 2 * HIDDEN_DIMS,
                HIDDEN_DIMS,
                1.0f / gruQuantScales.bx.gate[2].scale
            );

// -----------------------------
// 6. 量化 br
// -----------------------------
            quantizeFloatToInt<int32_t, true, true, false>(
                br_tmp_dev.data,
                br_dev.data,
                HIDDEN_DIMS,
                1.0f / gruQuantScales.br.gate[0].scale
            );

            quantizeFloatToInt<int32_t, true, true, false>(
                br_tmp_dev.data + HIDDEN_DIMS,
                br_dev.data + HIDDEN_DIMS,
                HIDDEN_DIMS,
                1.0f / gruQuantScales.br.gate[1].scale
            );

            quantizeFloatToInt<int32_t, true, true, false>(
                br_tmp_dev.data + 2 * HIDDEN_DIMS,
                br_dev.data + 2 * HIDDEN_DIMS,
                HIDDEN_DIMS,
                1.0f / gruQuantScales.br.gate[2].scale
            );

// -----------------------------
// 7. 量化 x
// -----------------------------
            // TODO: x分时间步不同
            quantizeFloatToInt<int8_t, true, true>(
                x_tmp_dev.data,
                x_dev.data,
                x_tmp_dev.size,
                1.0f / gruQuantScales.x.scale
            );

// -----------------------------
// 8. 量化 dh_new
// -----------------------------
//            quantizeFloatToInt<int8_t, true, true>(
//                dh_new_tmp_dev.data,
//                dh_new_dev.data,
//                dh_new_tmp_dev.size,
//                1.0f / gruQuantScales.x.scale
//            );

            device_ptr<Tensor1i32> tmp_W_sum_dev(hidden_size * 3);

            computeWeightSum(W_dev.data, tmp_W_sum_dev.data, hidden_size * 3, INPUT_DIMS);

            device_ptr<Tensor2i8> h_dev((time_steps + 1) * batch_size * hidden_size);
            device_ptr<Tensor3i32> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
            device_ptr<Tensor2i32> tmp_Rh_dev(batch_size * hidden_size * 3);
            device_ptr<Tensor3i8> v_dev(time_steps * batch_size * hidden_size * 4);

            h_dev.zero();

            {
                ScopeTimer t("Train forward:");
                gru::ForwardPass<int8_t> forward = gru::ForwardPass<int8_t>(
                    true,  // training
                    batch_size,
                    input_size,
                    hidden_size,
                    g_blas_handle);

                forward.Run(
                    time_steps,
                    W_dev.data,
                    R_dev.data,
                    bx_dev.data,
                    br_dev.data,
                    x_dev.data,
                    h_dev.data,
                    v_dev.data,
                    tmp_Wx_dev.data,
                    tmp_Rh_dev.data,
                    0.0f,
                    nullptr,
                    gruQuantScales);
            }


//            device_ptr<Tensor3f> dx_dev(time_steps * batch_size * input_size);
//            device_ptr<Tensor2f> dW_dev(input_size * hidden_size * 3);
//            device_ptr<Tensor2f> dR_dev(hidden_size * hidden_size * 3);
//            device_ptr<Tensor1f> dbx_dev(hidden_size * 3);
//            device_ptr<Tensor1f> dbr_dev(hidden_size * 3);
//            device_ptr<Tensor2f> dh_dev(batch_size * hidden_size);
//            device_ptr<Tensor3f> dp_dev(time_steps * batch_size * hidden_size * 3);
//            device_ptr<Tensor3f> dq_dev(time_steps * batch_size * hidden_size * 3);
//
//            {
//                ScopeTimer t("Train backward:");
//                gru::BackwardPass<float> backward(
//                    batch_size,
//                    input_size,
//                    hidden_size,
//                    g_blas_handle);
//
//                backward.Run(
//                    time_steps,
//                    W_dev.data,
//                    R_dev.data,
//                    bx_dev.data,
//                    br_dev.data,
//                    x_dev.data,
//                    h_dev.data,
//                    v_dev.data,
//                    dh_new_dev.data,
//                    dx_dev.data,
//                    dW_dev.data,
//                    dR_dev.data,
//                    dbx_dev.data,
//                    dbr_dev.data,
//                    dh_dev.data,
//                    dp_dev.data,
//                    dq_dev.data,
//                    nullptr);
//            }

        } else { // int16量化

        }
    }
}


int main() {
    srand(time(0));

    cublasCreate(&g_blas_handle);

    // Weights.
    Tensor2f W(HIDDEN_DIMS * 3, INPUT_DIMS); // 对应W_z/W_r/W_h的合并
    Tensor2f R(HIDDEN_DIMS * 3, HIDDEN_DIMS); // 对应R_z/R_r/R_h的合并
    Tensor1f bx(HIDDEN_DIMS * 3); // 对应b_z/b_r/b_h的合并. bx 负责给 “输入 x_t 到门控的线性变换” 加偏置
    Tensor1f br(HIDDEN_DIMS * 3); // br: 3H(部分实现中偏置分输出\隐藏层. br 负责给 “隐藏状态 h_{t-1} 到门控的线性变换” 加偏置

    // Input.
    Tensor3f x(INPUT_DIMS, BATCH_SIZE, SEQUENCE_LEN);

    // Gradients from upstream layers.
    Tensor3f dh(HIDDEN_DIMS, BATCH_SIZE, SEQUENCE_LEN + 1);

    W.setRandom();
    R.setRandom();
    bx.setRandom();
    br.setRandom();
    x.setRandom();
    dh.setRandom();

    GruInference(W, R, bx, br, x);

    printf("cudaError(GruInference finish): %s\n", cudaGetErrorString(cudaGetLastError()));

    GruTrain(W, R, bx, br, x, dh, true, false);

    printf("cudaError(GruTrain finish): %s\n", cudaGetErrorString(cudaGetLastError()));


    cublasDestroy(g_blas_handle);

    return 0;
}
