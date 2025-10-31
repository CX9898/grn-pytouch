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
#include "quantize_ops.cuh"

using INPUT_TPYE = int8_t;
using ACCUMULATION_TYPE = int32_t;

using Tensor1 = Eigen::Tensor<INPUT_TPYE, 1>;
using Tensor2 = Eigen::Tensor<INPUT_TPYE, 2>;
using Tensor3 = Eigen::Tensor<INPUT_TPYE, 3>;

using AccTensor1 = Eigen::Tensor<ACCUMULATION_TYPE, 1>;
using AccTensor2 = Eigen::Tensor<ACCUMULATION_TYPE, 2>;
using AccTensor3 = Eigen::Tensor<ACCUMULATION_TYPE, 3>;

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

struct QuantParams {
  // 示例

  // ==========================
  // 输入 / 权重 / 隐状态 scale & zero_point
  // ==========================
  float s_x;       // 输入 x 的 scale，用于 float->int8 或对齐累加
  int32_t zp_x;    // 输入 x 的 zero_point(非对称量化时使用)

  float s_w;       // 输入到隐藏权重 Wx 的 scale，用于矩阵乘法
  int32_t zp_w;    // 权重 Wx 的 zero_point

  float s_r;       // 隐状态到隐藏权重 Rh 的 scale，用于矩阵乘法
  int32_t zp_r;    // 权重 Rh 的 zero_point

  float s_h;       // 隐状态 h 的 scale，用于下一时间步累加或 zoneout
  int32_t zp_h;    // 隐状态 h 的 zero_point

  // ==========================
  // 门控 scale & zero_point(最终量化目标)
  // ==========================
  float s_z;       // 更新门 z 的目标 scale，用于 int32->int8
  int32_t zp_z;    // 更新门 z 的 zero_point

  float s_r_gate;  // 重置门 r 的目标 scale
  int32_t zp_r_gate; // 重置门 r 的 zero_point

  float s_g;       // 候选门 g 的目标 scale
  int32_t zp_g;    // 候选门 g 的 zero_point

  // ==========================
  // M / shift (整数缩放系数)
  // 用于对齐累加值到目标 scale(通常 Wx 的 scale)
  // ==========================
  int32_t M_Rh_z;     // Rh 对齐到 Wx scale 的整数乘法系数，z 门
  int32_t shift_Rh_z; // Rh 对齐 z 门的右移位数

  int32_t M_Rh_r;     // Rh 对齐 r 门
  int32_t shift_Rh_r;

  int32_t M_Rh_g;     // Rh 对齐 g 门
  int32_t shift_Rh_g;

  int32_t M_br_z;     // br 对齐 z 门
  int32_t shift_br_z;

  int32_t M_br_r;     // br 对齐 r 门
  int32_t shift_br_r;

  int32_t M_br_g;     // br 对齐 g 门
  int32_t shift_br_g;

  // ==========================
  // 使用规则总结：
  // 1. s_x/s_w/s_r/s_h：用于输入/权重/隐藏状态量化
  // 2. s_z/s_r_gate/s_g：累加后量化到门控 int8 的目标 scale
  // 3. zp_*：非对称量化偏移
  // 4. M_* / shift_*：host 端计算，用于 GPU kernel 整数量化对齐
  // ==========================
};

void GruInference(
    const Tensor2 &W,
    const Tensor2 &R,
    const AccTensor1 &bx,
    const AccTensor1 &br,
    const Tensor3 &x) {
    const int time_steps = x.dimension(2);
    const int batch_size = x.dimension(1);
    const int input_size = x.dimension(0);
    const int hidden_size = R.dimension(1);

    // Copy weights over to GPU.
    device_ptr<Tensor2> W_dev(W);
    device_ptr<Tensor2> R_dev(R);
    device_ptr<AccTensor1> bx_dev(bx);
    device_ptr<AccTensor1> br_dev(br);
    device_ptr<Tensor3> x_dev(x);

    device_ptr<Tensor2> h_dev((time_steps + 1) * batch_size * hidden_size);
    device_ptr<AccTensor3> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3); // 用于存放W * x的中间结果
    device_ptr<AccTensor2> tmp_Rh_dev(batch_size * hidden_size * 3); // 用于存放R * h的中间结果

    h_dev.zero(); // h初始化为0

    ScopeTimer t("Inference:");

    gru::ForwardPass<INPUT_TPYE> forward = gru::ForwardPass<INPUT_TPYE>(
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
    const Tensor2 &W,
    const Tensor2 &R,
    const AccTensor1 &bx,
    const AccTensor1 &br,
    const Tensor3 &x,
    const Tensor3 &dh_new) {
    const int time_steps = x.dimension(2);
    const int batch_size = x.dimension(1);
    const int input_size = x.dimension(0);
    const int hidden_size = R.dimension(1);

    // Copy weights over to GPU.
    device_ptr<Tensor2> W_dev(W);
    device_ptr<Tensor2> R_dev(R);
    device_ptr<AccTensor1> bx_dev(bx);
    device_ptr<AccTensor1> br_dev(br);
    device_ptr<Tensor3> x_dev(x);
    device_ptr<Tensor3> dh_new_dev(dh_new);

    device_ptr<Tensor2> h_dev((time_steps + 1) * batch_size * hidden_size);
    device_ptr<AccTensor3> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    device_ptr<AccTensor2> tmp_Rh_dev(batch_size * hidden_size * 3);
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
    device_ptr<AccTensor1> dbx_dev(hidden_size * 3);
    device_ptr<AccTensor1> dbr_dev(hidden_size * 3);
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
    AccTensor1 bx(HIDDEN_DIMS * 3); // 对应b_z/b_r/b_h的合并. bx 负责给 “输入 x_t 到门控的线性变换” 加偏置
    AccTensor1 br(HIDDEN_DIMS * 3); // br: 3H(部分实现中偏置分输出\隐藏层. br 负责给 “隐藏状态 h_{t-1} 到门控的线性变换” 加偏置

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
