#include <Eigen/Dense>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "devVector.h"
#include "device_ptr.h"
#include "gru.h"
#include "gru_quant.h"

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

constexpr int BATCH_SIZE = 64;     // 批大小
constexpr int SEQUENCE_LEN = 1000; // 序列长度(T), 每个样本有T个时间步
constexpr int HIDDEN_DIMS = 512; // 隐藏层维度(H), h_t的维度
constexpr int INPUT_DIMS = 512;  // 输入维度(I), x_t的维度

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

template<bool use_int16 = false>
// 控制量化精度位宽
void GruQuantInit(
    const Tensor2f &W, // 输入到隐藏层的权重矩阵. [input_size, hidden_size * 3] 对应三个门
    const Tensor2f &R,  // 隐藏层到隐藏层的循环权重矩阵
    const Tensor1f &bx, // 输入偏置项（input bias），来自输入路径
    const Tensor1f &br, // 循环偏置项（recurrent bias），来自循环路径
    const Tensor3f &x, // 输入序列张量
    const Tensor3f &dh_new, // 来自上层网络或损失函数的反向梯度. [hidden_size, batch_size, time_steps]
    Tensor2i8 &W_quant,
    Tensor2i8 &R_quant,
    Tensor1i32 &bx_quant,
    Tensor1i32 &br_quant,
    Tensor3i8 &x_quant,
    Tensor3i8 &dh_new_quant,
    GruQuantScales &gruQuantScales,
    std::vector<RescaleParamsPerStep> &gruRescaleParams
) {
    const int time_steps = x.dimension(2);
    const int batch_size = x.dimension(1);
    const int input_size = x.dimension(0);
    const int hidden_size = R.dimension(1);

    // N : batch_size
    // C : input_size
    if (!use_int16) { // int8量化
        gruQuantScales = computeGruQuantParams<int8_t>(
            x.data(), time_steps, batch_size, input_size, W.data(), hidden_size,
            R.data(), bx.data(), br.data());

        computeGruRescaleParamsPerStep(time_steps, gruQuantScales, gruRescaleParams, 15);

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
            W_tmp_dev.data, W_dev.data, HIDDEN_DIMS * INPUT_DIMS,
            1.0f / gruQuantScales.W.gate[0].scale);

        quantizeFloatToInt<int8_t, true, true>(
            W_tmp_dev.data + HIDDEN_DIMS * INPUT_DIMS,
            W_dev.data + HIDDEN_DIMS * INPUT_DIMS, HIDDEN_DIMS * INPUT_DIMS,
            1.0f / gruQuantScales.W.gate[1].scale);

        quantizeFloatToInt<int8_t, true, true>(
            W_tmp_dev.data + 2 * HIDDEN_DIMS * INPUT_DIMS,
            W_dev.data + 2 * HIDDEN_DIMS * INPUT_DIMS, HIDDEN_DIMS * INPUT_DIMS,
            1.0f / gruQuantScales.W.gate[2].scale);

        // -----------------------------
        // 4. 量化 R
        // -----------------------------
        quantizeFloatToInt<int8_t, true, true>(
            R_tmp_dev.data, R_dev.data, HIDDEN_DIMS * HIDDEN_DIMS,
            1.0f / gruQuantScales.R.gate[0].scale);

        quantizeFloatToInt<int8_t, true, true>(
            R_tmp_dev.data + HIDDEN_DIMS * HIDDEN_DIMS,
            R_dev.data + HIDDEN_DIMS * HIDDEN_DIMS, HIDDEN_DIMS * HIDDEN_DIMS,
            1.0f / gruQuantScales.R.gate[1].scale);

        quantizeFloatToInt<int8_t, true, true>(
            R_tmp_dev.data + 2 * HIDDEN_DIMS * HIDDEN_DIMS,
            R_dev.data + 2 * HIDDEN_DIMS * HIDDEN_DIMS, HIDDEN_DIMS * HIDDEN_DIMS,
            1.0f / gruQuantScales.R.gate[2].scale);

        // -----------------------------
        // 5. 量化 bx
        // -----------------------------
        quantizeFloatToInt<int32_t, true, true, false>(
            bx_tmp_dev.data, bx_dev.data, HIDDEN_DIMS,
            1.0f / gruQuantScales.bx.gate[0].scale);

        quantizeFloatToInt<int32_t, true, true, false>(
            bx_tmp_dev.data + HIDDEN_DIMS, bx_dev.data + HIDDEN_DIMS, HIDDEN_DIMS,
            1.0f / gruQuantScales.bx.gate[1].scale);

        quantizeFloatToInt<int32_t, true, true, false>(
            bx_tmp_dev.data + 2 * HIDDEN_DIMS, bx_dev.data + 2 * HIDDEN_DIMS,
            HIDDEN_DIMS, 1.0f / gruQuantScales.bx.gate[2].scale);

        // -----------------------------
        // 6. 量化 br
        // -----------------------------
        quantizeFloatToInt<int32_t, true, true, false>(
            br_tmp_dev.data, br_dev.data, HIDDEN_DIMS,
            1.0f / gruQuantScales.br.gate[0].scale);

        quantizeFloatToInt<int32_t, true, true, false>(
            br_tmp_dev.data + HIDDEN_DIMS, br_dev.data + HIDDEN_DIMS, HIDDEN_DIMS,
            1.0f / gruQuantScales.br.gate[1].scale);

        quantizeFloatToInt<int32_t, true, true, false>(
            br_tmp_dev.data + 2 * HIDDEN_DIMS, br_dev.data + 2 * HIDDEN_DIMS,
            HIDDEN_DIMS, 1.0f / gruQuantScales.br.gate[2].scale);

        // -----------------------------
        // 7. 量化 x. 分时间步不同
        // -----------------------------
        dev::vector<float> x_scale_dev(gruQuantScales.x_scale);
        dev::vector<int32_t> x_zp_dev(gruQuantScales.x_zp);
        quantizeFloatToIntPerStep<int8_t, false, false>(
            x_tmp_dev.data, x_dev.data, x_tmp_dev.size, x_scale_dev.data(),
            x_zp_dev.data(), time_steps);

        // -----------------------------
        // 8. 量化 dh_new
        // -----------------------------
        //            quantizeFloatToInt<int8_t, true, true>(
        //                dh_new_tmp_dev.data,
        //                dh_new_dev.data,
        //                dh_new_tmp_dev.size,
        //                1.0f / gruQuantScales.x.scale
        //            );

        W_dev.ToHost(W_quant);
        R_dev.ToHost(R_quant);
        bx_dev.ToHost(bx_quant);
        br_dev.ToHost(br_quant);
        x_dev.ToHost(x_quant);
        dh_new_dev.ToHost(dh_new_quant);

    } else {
        // int16量化
    }


}

template<bool use_int16_quant = false>
void GruInferenceQuant(const Tensor2i8 &W,
                       const Tensor2i8 &R,
                       const Tensor1i32 &bx,
                       const Tensor1i32 &br,
                       const Tensor3i8 &x,
                       const std::vector<RescaleParamsPerStep> &rescaleParams,
                       const GruQuantScalesFixed &gruQuantScalesFixed,
                       Tensor3i8 &h // (time_steps + 1) * batch_size * hidden_size
) {
    const int time_steps = x.dimension(2);
    const int batch_size = x.dimension(1);
    const int input_size = x.dimension(0);
    const int hidden_size = R.dimension(1);

    // Copy weights over to GPU.
    device_ptr<Tensor2i8> W_dev(W);
    device_ptr<Tensor2i8> R_dev(R);
    device_ptr<Tensor1i32> bx_dev(bx);
    device_ptr<Tensor1i32> br_dev(br);
    device_ptr<Tensor3i8> x_dev(x);

    device_ptr<Tensor3i32> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3); // 用于存放W * x的中间结果
    device_ptr<Tensor2i32> tmp_Rh_dev(batch_size * hidden_size * 3); // 用于存放R * h的中间结果

    device_ptr<Tensor3i8> h_dev(h);
//    h_dev.zero(); // h初始化为0

    {
        ScopeTimer t("Inference Quant:");

        gru::ForwardPassQuant<int8_t> forward = gru::ForwardPassQuant<int8_t>(
            false, // training
            batch_size, input_size, hidden_size, g_blas_handle);

        // TODO: 得到

        forward.Run(time_steps, W_dev.data, R_dev.data, bx_dev.data, br_dev.data,
                    x_dev.data, h_dev.data, nullptr, tmp_Wx_dev.data, tmp_Rh_dev.data,
                    0.0f, nullptr);
    }

    h_dev.ToHost(h);
}

void GruInference(const Tensor2f &W,
                  const Tensor2f &R,
                  const Tensor1f &bx,
                  const Tensor1f &br,
                  const Tensor3f &x,
                  Tensor3f h) {
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

    device_ptr<Tensor3f> h_dev(h);
    device_ptr<Tensor3f> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3); // 用于存放W * x的中间结果
    device_ptr<Tensor2f> tmp_Rh_dev(batch_size * hidden_size * 3); // 用于存放R * h的中间结果

//    h_dev.zero(); // h初始化为0

    {
        ScopeTimer t("Inference:");

        gru::ForwardPass<float> forward = gru::ForwardPass<float>(
            false, // training
            batch_size, input_size, hidden_size, g_blas_handle);

        forward.Run(time_steps, W_dev.data, R_dev.data, bx_dev.data, br_dev.data,
                    x_dev.data, h_dev.data, nullptr, tmp_Wx_dev.data, tmp_Rh_dev.data,
                    0.0f, nullptr);
    }

    h_dev.ToHost(h);
}

void GruTrain(const Tensor2f &W, // 输入到隐藏层的权重矩阵. [input_size,
    // hidden_size * 3] 对应三个门
              const Tensor2f &R, // 隐藏层到隐藏层的循环权重矩阵
              const Tensor1f &bx, // 输入偏置项（input bias），来自输入路径
              const Tensor1f &br, // 循环偏置项（recurrent bias），来自循环路径
              const Tensor3f &x, // 输入序列张量
              const Tensor3f &dh_new, // 来自上层网络或损失函数的反向梯度.
    // [hidden_size, batch_size, time_steps]
              bool enable_quantitative = false, // 是否启用量化推理模式
              bool use_int16 = false            // 控制量化精度位宽
) {
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

    device_ptr<Tensor3f> dx_dev(time_steps * batch_size *
                                input_size); // 输入序列梯度
    device_ptr<Tensor2f> dW_dev(input_size * hidden_size *
                                3); // 对输入权重的梯度
    device_ptr<Tensor2f> dR_dev(hidden_size * hidden_size *
                                3);                // 对循环权重的梯度
    device_ptr<Tensor1f> dbx_dev(hidden_size * 3); // 对输入偏置的梯度
    device_ptr<Tensor1f> dbr_dev(hidden_size * 3); // 对循环偏置的梯度
    device_ptr<Tensor2f> dh_dev(batch_size *
                                hidden_size); // 对最后隐藏状态的梯度
    device_ptr<Tensor3f> dp_dev(time_steps * batch_size * hidden_size *
                                3); // 临时缓存梯度（内部结构用）
    device_ptr<Tensor3f> dq_dev(time_steps * batch_size * hidden_size * 3); // 临时缓存梯度（内部结构用）

    {
        ScopeTimer t("Train backward:");
        gru::BackwardPass<float> backward(batch_size, input_size, hidden_size,
                                          g_blas_handle);

        backward.Run(time_steps, W_dev.data, R_dev.data, bx_dev.data, br_dev.data,
                     x_dev.data, h_dev.data, v_dev.data, dh_new_dev.data,
                     dx_dev.data, dW_dev.data, dR_dev.data, dbx_dev.data,
                     dbr_dev.data, dh_dev.data, dp_dev.data, dq_dev.data,
                     nullptr);
    }

}

// 计算余弦相似度
float cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-8f); // 防止除零
}

void checkHQuantizationWithCosine(
    const std::vector<float> &h_inference,          // 浮点 h, size = (time_steps+1) * batch_size * hidden_size
    const std::vector<int8_t> &h_quant_inference,  // 量化 h, size 同上
    int time_steps,
    int batch_size,
    int hidden_size,
    const std::vector<ScaleParam> &scaleParam,                  // 每步 scale M, 每步 shift
    float threshold = 1.0f                          // 超阈值
) {

    float max_h = 0.0f;
    for (float v : h_inference) max_h = std::max(max_h, std::abs(v));
    float step = max_h / 127.0f;  // int8
    threshold = 0.5f * step;

    const int size_per_step = batch_size * hidden_size;

    std::vector<float> h_float_step(size_per_step);
    std::vector<float> h_quant_step(size_per_step);

    for (int t = 1; t <= time_steps; ++t) {
        // 拷贝浮点 h
        std::copy(h_inference.begin() + t * size_per_step,
                  h_inference.begin() + (t + 1) * size_per_step,
                  h_float_step.begin());

        // 反量化
        dequantizeTensorFixedPoint(h_quant_inference.data() + t * size_per_step,
                                   size_per_step,
                                   scaleParam[t].M,
                                   scaleParam[t].shift,
                                   h_quant_step.data());

        // 差值统计
        float max_diff = 0.0f;
        float sum_diff = 0.0f;

        int count = 0;
        for (int idx = 0; idx < size_per_step; ++idx) {
            float diff = std::abs(h_float_step[idx] - h_quant_step[idx]);
            sum_diff += diff;
            if (diff > max_diff) max_diff = diff;

            if (diff > threshold) {
                count++;
                if (count < 5) {
                    printf("[Warning] t=%d idx=%d diff=%f h_float=%f h_quant=%f\n",
                           t, idx, diff, h_float_step[idx], h_quant_step[idx]);
                }
            }
        }
        const float baifenbi = static_cast<float>(count) / static_cast<float>(size_per_step);

        float mean_diff = sum_diff / size_per_step;
        float cos_sim = cosineSimilarity(h_float_step, h_quant_step);

        printf("Time step %d: max_diff=%f, mean_diff=%f, cosine_sim=%f, baifenbi = %f\n",
               t, max_diff, mean_diff, cos_sim, baifenbi);
    }
}

template<typename QuantT>
void calibrateGruScales(const Tensor2f &W,
                        const Tensor2f &R,
                        const Tensor1f &bx,
                        const Tensor1f &br,
                        const Tensor3f &x,
                        GruQuantScales &gruQuantScales
) {
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
//    device_ptr<Tensor3f> dh_new_dev(dh_new);

    device_ptr<Tensor2f> h_dev((time_steps + 1) * batch_size * hidden_size);
    device_ptr<Tensor3f> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    device_ptr<Tensor2f> tmp_Rh_dev(batch_size * hidden_size * 3);
    device_ptr<Tensor3f> v_dev(time_steps * batch_size * hidden_size * 4);

    h_dev.zero();

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

    // TODO 校准得到量化参数

    for (int t = 0; t < time_steps; ++t) {
//        const int wx_size_per_step = batch_size * hidden_size * 3;
//        const int Rh_size_per_step =;
    }
//    GruQuantScales
}

int main() {
    srand(time(0));

    cublasCreate(&g_blas_handle);

    // Weights.
    Tensor2f W(HIDDEN_DIMS * 3, INPUT_DIMS);  // 对应W_z/W_r/W_h的合并
    Tensor2f R(HIDDEN_DIMS * 3, HIDDEN_DIMS); // 对应R_z/R_r/R_h的合并
    Tensor1f bx(HIDDEN_DIMS * 3); // 对应b_z/b_r/b_h的合并. bx 负责给 “输入 x_t 到门控的线性变换” 加偏置
    Tensor1f br(HIDDEN_DIMS * 3); // br: 3H(部分实现中偏置分输出\隐藏层. br 负责给“隐藏状态 h_{t-1} 到门控的线性变换” 加偏置

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

    const int time_steps = x.dimension(2);
    const int batch_size = x.dimension(1);
    const int input_size = x.dimension(0);
    const int hidden_size = R.dimension(1);

    Tensor3f h_inference(hidden_size, batch_size, (time_steps + 1));
    h_inference.setRandom();
    GruInference(W, R, bx, br, x, h_inference);

    printf("cudaError(GruInference finish): %s\n", cudaGetErrorString(cudaGetLastError()));

    GruTrain(W, R, bx, br, x, dh, false, false);

    printf("cudaError(GruTrain finish): %s\n", cudaGetErrorString(cudaGetLastError()));

    // Quant



    Tensor2i8 W_quant(HIDDEN_DIMS * 3, INPUT_DIMS);  // 对应W_z/W_r/W_h的合并
    Tensor2i8 R_quant(HIDDEN_DIMS * 3, HIDDEN_DIMS); // 对应R_z/R_r/R_h的合并
    Tensor1i32 bx_quant(HIDDEN_DIMS * 3); // 对应b_z/b_r/b_h的合并. bx 负责给 “输入 x_t 到门控的线性变换” 加偏置
    Tensor1i32 br_quant(HIDDEN_DIMS * 3); // br: 3H(部分实现中偏置分输出\隐藏层. br 负责给“隐藏状态 h_{t-1} 到门控的线性变换” 加偏置
    Tensor3i8 x_quant(INPUT_DIMS, BATCH_SIZE, SEQUENCE_LEN);
    Tensor3i8 dh_new_quant(HIDDEN_DIMS, BATCH_SIZE, SEQUENCE_LEN + 1);

    GruQuantScales gruQuantScales;
    std::vector<RescaleParamsPerStep> rescaleParams(SEQUENCE_LEN);
    GruQuantInit<false>(W,
                        R,
                        bx,
                        br,
                        x,
                        dh,
                        W_quant,
                        R_quant,
                        bx_quant,
                        br_quant,
                        x_quant,
                        dh_new_quant,
                        gruQuantScales,
                        rescaleParams);

    Tensor3i8 h_quant_inference(hidden_size, batch_size, (time_steps + 1));
    {
        // N : batch_size
        // C : input_size
        QuantParams h_0_quantParams = calculateQuantParams<int8_t>(h_inference.data(),
                                                                   batch_size * hidden_size);
        gruQuantScales.h_scale[0] = h_0_quantParams.scale;

        device_ptr<Tensor3f> h_dev(h_inference);
        device_ptr<Tensor3i8> h_quant_dev((time_steps + 1) * batch_size * hidden_size);
        quantizeFloatToInt<int8_t, true, true>(
            h_dev.data, h_quant_dev.data, batch_size * hidden_size,
            1.0f / gruQuantScales.h_scale[0]);
        h_quant_dev.ToHost(h_quant_inference);
    }

    GruQuantScalesFixed gruQuantScalesFixed;
    gruQuantScalesFixed.initialize(gruQuantScales);
    gruQuantScalesFixed.h[0] = floatScaleToFixed(gruQuantScales.h_scale[0]);

    GruInferenceQuant(W_quant,
                      R_quant,
                      bx_quant,
                      br_quant,
                      x_quant,
                      rescaleParams,
                      gruQuantScalesFixed,
                      h_quant_inference);

    { // Test
        std::vector<float> h_inference_tmp(h_inference.data(), h_inference.data() + h_inference.size());
        std::vector<int8_t> h_quant_inference_tmp(h_quant_inference.data(),
                                                  h_quant_inference.data() + h_quant_inference.size());

        checkHQuantizationWithCosine(h_inference_tmp,
                                     h_quant_inference_tmp,
                                     time_steps,
                                     batch_size,
                                     hidden_size,
                                     gruQuantScalesFixed.h);
    }

    printf("cudaError(GruInferenceQuant finish): %s\n", cudaGetErrorString(cudaGetLastError()));

    cublasDestroy(g_blas_handle);

    return 0;
}
