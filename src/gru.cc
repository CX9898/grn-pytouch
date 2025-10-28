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

using INPUT_TPYE = int8_t;

using Tensor1 = Eigen::Tensor<INPUT_TPYE, 1>;
using Tensor2 = Eigen::Tensor<INPUT_TPYE, 2>;
using Tensor3 = Eigen::Tensor<INPUT_TPYE, 3>;

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
    const Tensor2 &W,
    const Tensor2 &R,
    const Tensor1 &bx,
    const Tensor1 &br,
    const Tensor3 &x) {
    const int time_steps = x.dimension(2);
    const int batch_size = x.dimension(1);
    const int input_size = x.dimension(0);
    const int hidden_size = R.dimension(1);

    // Copy weights over to GPU.
    device_ptr<Tensor2> W_dev(W);
    device_ptr<Tensor2> R_dev(R);
    device_ptr<Tensor1> bx_dev(bx);
    device_ptr<Tensor1> br_dev(br);
    device_ptr<Tensor3> x_dev(x);

    device_ptr<Tensor2> h_dev((time_steps + 1) * batch_size * hidden_size);
    device_ptr<Tensor3> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    device_ptr<Tensor2> tmp_Rh_dev(batch_size * hidden_size * 3);

    h_dev.zero();

    ScopeTimer t("Inference:");

    gru::ForwardPass<INPUT_TPYE> forward = gru::ForwardPass<INPUT_TPYE>(
        false,  // training
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
        nullptr,
        tmp_Wx_dev.data,
        tmp_Rh_dev.data,
        0.0f,
        nullptr);
}

void GruTrain(
    const Tensor2 &W,
    const Tensor2 &R,
    const Tensor1 &bx,
    const Tensor1 &br,
    const Tensor3 &x,
    const Tensor3 &dh_new) {
    const int time_steps = x.dimension(2);
    const int batch_size = x.dimension(1);
    const int input_size = x.dimension(0);
    const int hidden_size = R.dimension(1);

    // Copy weights over to GPU.
    device_ptr<Tensor2> W_dev(W);
    device_ptr<Tensor2> R_dev(R);
    device_ptr<Tensor1> bx_dev(bx);
    device_ptr<Tensor1> br_dev(br);
    device_ptr<Tensor3> x_dev(x);
    device_ptr<Tensor3> dh_new_dev(dh_new);

    device_ptr<Tensor2> h_dev((time_steps + 1) * batch_size * hidden_size);
    device_ptr<Tensor3> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    device_ptr<Tensor2> tmp_Rh_dev(batch_size * hidden_size * 3);
    device_ptr<Tensor3> v_dev(time_steps * batch_size * hidden_size * 4);

    h_dev.zero();

    {
        ScopeTimer t("Train forward:");
        gru::ForwardPass<INPUT_TPYE> forward = gru::ForwardPass<INPUT_TPYE>(
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

    device_ptr<Tensor3> dx_dev(time_steps * batch_size * input_size);
    device_ptr<Tensor2> dW_dev(input_size * hidden_size * 3);
    device_ptr<Tensor2> dR_dev(hidden_size * hidden_size * 3);
    device_ptr<Tensor1> dbx_dev(hidden_size * 3);
    device_ptr<Tensor1> dbr_dev(hidden_size * 3);
    device_ptr<Tensor2> dh_dev(batch_size * hidden_size);
    device_ptr<Tensor3> dp_dev(time_steps * batch_size * hidden_size * 3);
    device_ptr<Tensor3> dq_dev(time_steps * batch_size * hidden_size * 3);

    {
        ScopeTimer t("Train backward:");
        gru::BackwardPass<INPUT_TPYE> backward(
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
}

int main() {
    srand(time(0));

    cublasCreate(&g_blas_handle);

    // Weights.
    Tensor2 W(HIDDEN_DIMS * 3, INPUT_DIMS); // 对应W_z/W_r/W_h的合并
    Tensor2 R(HIDDEN_DIMS * 3, HIDDEN_DIMS); // 对应R_z/R_r/R_h的合并
    Tensor1 bx(HIDDEN_DIMS * 3); // 对应b_z/b_r/b_h的合并. bx 负责给 “输入 x_t 到门控的线性变换” 加偏置
    Tensor1 br(HIDDEN_DIMS * 3); // br: 3H(部分实现中偏置分输出\隐藏层. br 负责给 “隐藏状态 h_{t-1} 到门控的线性变换” 加偏置

    // Input.
    Tensor3 x(INPUT_DIMS, BATCH_SIZE, SEQUENCE_LEN);

    // Gradients from upstream layers.
    Tensor3 dh(HIDDEN_DIMS, BATCH_SIZE, SEQUENCE_LEN + 1);

    W.setRandom();
    R.setRandom();
    bx.setRandom();
    br.setRandom();
    x.setRandom();
    dh.setRandom();

    GruInference(W, R, bx, br, x);


    GruTrain(W, R, bx, br, x, dh);


    cublasDestroy(g_blas_handle);

    return 0;
}
