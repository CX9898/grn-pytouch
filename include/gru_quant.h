#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "quantize_ops_helper.h"
#include "scale_encoding.h"  // encodeMShift / toFixedScale(s) / makeRescale*（setRescaleParam 用）

namespace gru {

// 量化 GRU 前向传播类
// 所有量化值统一使用 int32_t 存储，实际值通过 clamp_by_bitwidth 限制到对应位宽
class ForwardPassQuant {
   public:
    // training: `true` if the caller intends to perform a backward pass to compute gradients.
    // batch_size: the number of training/inference inputs provided in each tensor.
    // input_size: the dimension of each input vector.
    // hidden_size: the expected dimension of each output vector.
    // blas_handle: an initialized cuBLAS handle (see `cublasCreate`).
    ForwardPassQuant(const bool training, const int batch_size, const int input_size,
                     const int hidden_size, const cublasHandle_t &blas_handle,
                     const cudaStream_t &stream = 0);

    // Releases internal resources.
    // Blocks until all iterations have completed executing on the GPU.
    ~ForwardPassQuant();

    void setRescaleParam(const GRUQuantParams &parms);

    // 简化的 GRU 前向接口（内部管理临时缓冲区）
    // 所有输入输出数据使用 int32_t 存储，实际值限制在配置的位宽范围内
    //
    // W: [C,H*3] 输入权重矩阵（量化后，int32_t 存储）
    // R: [H,H*3] 循环权重矩阵（量化后，int32_t 存储）
    // bw: [H*3] 输入偏置（量化后）
    // br: [H*3] 循环偏置（量化后）
    // x: [N*T,C] 输入序列（量化后，int32_t 存储）
    // h: [(T+1)*N,H] 初始和输出隐藏状态（量化后，int32_t 存储）
    // v: [T*N,H*4] 中间激活值（训练模式需要）
    // zoneout_prob: Zoneout 概率
    // zoneout_mask: [T*N,H] Zoneout mask（int32_t 存储）
    // weight_ih_linear_mask: [T*N, H*3] weight_ih_linear clamp mask（外部分配，nullptr=不保存）
    // weight_hh_linear_mask: [T*N, H*3] weight_hh_linear clamp mask（外部分配，nullptr=不保存）
    // gate_input_mask: [T*N, H*3] gate input clamp mask（外部分配，nullptr=不保存）
    // gate_output_mask: [T*N, H*3] gate output clamp mask（外部分配，nullptr=不保存）
    // h_mask: [T*N, H] hidden state clamp mask（外部分配，nullptr=不保存）
    void Run(const int steps, const int32_t *W, const int32_t *R, const int32_t *bw,
             const int32_t *br, const int32_t *x, int32_t *h, int32_t *v,
             const float zoneout_prob, const int32_t *zoneout_mask,
             uint8_t *weight_ih_linear_mask = nullptr,
             uint8_t *weight_hh_linear_mask = nullptr,
             uint8_t *gate_input_mask = nullptr,
             uint8_t *gate_output_mask = nullptr,
             uint8_t *h_mask = nullptr);

    /// @brief 获取中间值（用于调试和比较）
    /// @param weight_ih_linear_out 输出 weight_ih_linear (W*x + bw) 的量化值
    /// @param weight_hh_linear_out 输出 weight_hh_linear (R*h + br) 的量化值（最后一个时间步）
    void GetIntermediateValues(dev::vector<int32_t>* weight_ih_linear_out,
                               dev::vector<int32_t>* weight_hh_linear_out) const;

   private:
    // 按 rescale 表示 R 模板化的执行实现（R = Pot2Rescale / FixedPointScale）。
    // Run() 根据 use_pot2_ 选择实例化，编译期生成两套 kernel，逻辑一套。
    template <class R>
    void RunImpl(const LinearQuantParamsGPUT<R> &lp, const GateQuantParamsT<R> &gp,
                 int steps, const int32_t *W, const int32_t *R_w, const int32_t *bw,
                 const int32_t *br, const int32_t *x, int32_t *h, int32_t *v,
                 float zoneout_prob, const int32_t *zoneout_mask,
                 uint8_t *weight_ih_linear_mask, uint8_t *weight_hh_linear_mask,
                 uint8_t *gate_input_mask, uint8_t *gate_output_mask, uint8_t *h_mask);

    // 内部迭代函数 (Linear 融合版本)
    template <class R>
    void IterateInternal(const LinearQuantParamsGPUT<R> &lp, const GateQuantParamsT<R> &gp,
                         const int32_t *R_w, const int32_t *br,
                         const int32_t *h, int32_t *h_out, int32_t *v,
                         const int32_t *cur_linear_x, const float zoneout_prob,
                         const int32_t *zoneout_mask,
                         uint8_t *weight_hh_linear_mask = nullptr,
                         uint8_t *gate_input_mask = nullptr,
                         uint8_t *gate_output_mask = nullptr,
                         uint8_t *h_mask = nullptr);

    // 计算输入 Linear 变换: W*x + bw（输出到 tmp_linear_x_）
    template <class R>
    void ComputeLinearX(const LinearQuantParamsGPUT<R> &lp, const GateQuantParamsT<R> &gp,
                        const int32_t *W, const int32_t *x, const int32_t *bw, int steps,
                        uint8_t *weight_ih_linear_mask = nullptr);

    // 计算隐状态 Linear 变换: R*h + br（输出到 tmp_linear_h_）
    template <class R>
    void ComputeLinearH(const LinearQuantParamsGPUT<R> &lp, const GateQuantParamsT<R> &gp,
                        const int32_t *R_w, const int32_t *h, const int32_t *br,
                        uint8_t *weight_hh_linear_mask = nullptr);

    // 预分配内存缓冲区
    void EnsureBuffersAllocated(int steps);

    // 预计算权重相关的常量
    void PrecomputeWeightSums(const int32_t *W, const int32_t *R);

    struct private_data;
    private_data *data_;

    // -------------------- 量化参数（类型驱动派发，只填充生效的那套）--------------------
    bool use_pot2_ = false;                ///< setRescaleParam 时确定的运行时模式
    OperatorQuantConfig bitwidth_config_;  ///< 位宽配置（缓冲区分配等通用逻辑使用）
    GateQuantParamsPot2 gate_params_pot2_;
    GateQuantParamsMShift gate_params_mshift_;
    LinearQuantParamsGPUT<Pot2Rescale> linear_params_pot2_;
    LinearQuantParamsGPUT<FixedPointScale> linear_params_mshift_;

    // 预分配的内部缓冲区（使用 dev::vector 自动管理内存）
    int max_steps_ = 0;

    // Linear 变换结果（int32，供 gate 计算使用）
    dev::vector<int32_t> tmp_weight_ih_linear_;  // [hidden*3 * max_steps * batch] W*x + bw
    dev::vector<int32_t> tmp_weight_hh_linear_;  // [hidden*3 * batch] R*h + br

    // 权重和常量（预计算）
    dev::vector<int64_t> W_sum_mul_x_zp_;  // [hidden*3]
    dev::vector<int64_t> R_sum_mul_h_zp_;  // [hidden*3]
    bool weight_sums_computed_ = false;

    // 缓存的权重指针（用于检测权重是否变化）
    const int32_t *cached_W_ = nullptr;
    const int32_t *cached_R_ = nullptr;

    // INT8 GEMM 优化：临时缓冲区（当位宽 <= 8 时使用 cuBLAS INT8 GEMM）
    dev::vector<int8_t> tmp_W_i8_;   // [hidden*3 * input] 权重 int8 缓存
    dev::vector<int8_t> tmp_R_i8_;   // [hidden*3 * hidden] 递归权重 int8 缓存
    dev::vector<int8_t> tmp_x_i8_;   // [input * steps * batch] 输入 int8 缓存
    dev::vector<int8_t> tmp_h_i8_;   // [hidden * batch] 隐藏状态 int8 缓存
    
    // cuBLAS INT8 GEMM 的 N 维度填充
    int N_padded_Rh_ = 0;
    dev::vector<int8_t> h_padded_i8_;  // [hidden * N_padded_Rh]
};

// ============================================================================
// 浮点存储版量化 GRU 前向传播类
// ============================================================================

/**
 * @brief 量化 GRU 前向传播类（浮点存储版本）
 *
 * 与 ForwardPassQuant 的区别：
 *   - 所有量化值使用 float 存储（值仍是定点整数）
 *   - 使用 cuBLAS SGEMM + 单独的 bias/rescale kernel
 *   - 只使用 real_sigmoid/real_tanh，不用 LUT
 *
 * 适用场景：
 *   - 需要浮点存储格式进行后续处理
 *   - 调试和验证量化精度
 */
class ForwardPassQuantFP {
public:
    /// @brief 构造函数
    /// @param training 是否训练模式（需要保存中间值）
    /// @param batch_size 批大小
    /// @param input_size 输入维度
    /// @param hidden_size 隐藏层维度
    /// @param blas_handle cuBLAS 句柄
    /// @param stream CUDA 流（可选）
    ForwardPassQuantFP(bool training, int batch_size, int input_size,
                       int hidden_size, const cublasHandle_t &blas_handle,
                       const cudaStream_t &stream = 0);

    ~ForwardPassQuantFP();

    /// @brief 设置量化参数（从整数版参数转换）
    void setRescaleParam(const GRUQuantParams &params);

    /// @brief 前向传播
    /// @param steps 时间步数
    /// @param W [C, H*3] 输入权重（float 存储的定点值）
    /// @param R [H, H*3] 循环权重
    /// @param bw [H*3] 输入偏置
    /// @param br [H*3] 循环偏置
    /// @param x [N*T, C] 输入序列
    /// @param h [(T+1)*N, H] 隐状态（输入输出）
    /// @param v [T*N, H*4] 中间值（训练模式）
    /// @param zoneout_prob Zoneout 概率
    /// @param zoneout_mask Zoneout mask
    /// @param weight_ih_linear_mask [T*N, H*3] weight_ih_linear clamp mask（外部分配，nullptr=不保存）
    /// @param weight_hh_linear_mask [T*N, H*3] weight_hh_linear clamp mask（外部分配，nullptr=不保存）
    /// @param gate_input_mask [T*N, H*3] gate input clamp mask（外部分配，nullptr=不保存）
    /// @param gate_output_mask [T*N, H*3] gate output clamp mask（外部分配，nullptr=不保存）
    /// @param h_mask [T*N, H] hidden state clamp mask（外部分配，nullptr=不保存）
    void Run(int steps,
             const float *W, const float *R,
             const float *bw, const float *br,
             const float *x, float *h, float *v,
             float zoneout_prob, const float *zoneout_mask,
             uint8_t *weight_ih_linear_mask = nullptr,
             uint8_t *weight_hh_linear_mask = nullptr,
             uint8_t *gate_input_mask = nullptr,
             uint8_t *gate_output_mask = nullptr,
             uint8_t *h_mask = nullptr);

private:
    void ComputeLinearX(const float *W, const float *x, const float *bw, int steps);
    void ComputeLinearH(const float *R, const float *h, const float *br,
                        uint8_t *weight_hh_linear_mask = nullptr);
    void IterateInternal(const float *R, const float *bw, const float *br,
                         const float *h, float *h_out, float *v,
                         const float *cur_weight_ih_linear,
                         float zoneout_prob, const float *zoneout_mask,
                         uint8_t *weight_ih_linear_mask = nullptr,
                         uint8_t *weight_hh_linear_mask = nullptr,
                         uint8_t *gate_input_mask = nullptr,
                         uint8_t *gate_output_mask = nullptr,
                         uint8_t *h_mask = nullptr);
    void EnsureBuffersAllocated(int steps);
    void PrecomputeWeightSums(const float *W, const float *R);

    struct private_data;
    private_data *data_;

    GateQuantParamsFP gate_params_;
    LinearRescaleParamsFP linear_params_;  ///< Linear 层量化参数（统一管理，直接传递给 kernel）
    OperatorQuantConfig bitwidth_config_;  ///< 位宽配置（单独存储，作为 kernel 参数传递）

    int max_steps_ = 0;

    // Linear 变换结果（cuBLAS GEMM 直接写入，然后原地 rescale）
    dev::vector<float> tmp_weight_ih_linear_;  // [hidden*3 * max_steps * batch] W*x + bw
    dev::vector<float> tmp_weight_hh_linear_;  // [hidden*3 * batch] R*h + br

    // 权重和常量（预计算，用于零点补偿）
    // 使用 float（权重总是 8bit，K < 1024 且 zp < 100 时 float 足够精确）
    dev::vector<float> W_sum_mul_x_zp_;  // [hidden*3]
    dev::vector<float> R_sum_mul_h_zp_;  // [hidden*3]
    bool weight_sums_computed_ = false;

    // Per-channel 数组存储（PER_CHANNEL 粒度时使用，指针存储在 linear_params_ 中）
    dev::vector<float> inv_div_gemm_x_to_weight_ih_linear_;  // [3*hidden] PER_CHANNEL 时使用
    dev::vector<float> inv_div_bw_to_gemm_x_;              // [3*hidden] PER_CHANNEL 时使用
    dev::vector<float> inv_div_gemm_h_to_weight_hh_linear_;  // [3*hidden] PER_CHANNEL 时使用
    dev::vector<float> inv_div_br_to_gemm_h_;              // [3*hidden] PER_CHANNEL 时使用

    const float *cached_W_ = nullptr;
    const float *cached_R_ = nullptr;
};

// ============================================================================
// 量化 GRU 反向传播类（支持 QAT mask）
// ============================================================================

/**
 * @brief 量化 GRU 反向传播类
 *
 * 基于原始 BackwardPass 实现，增加 QAT mask 支持：
 *   - 在反向传播中应用 clamp mask（Straight-Through Estimator）
 *   - 被 clamp 的值（mask=1）梯度置零
 *   - 未被 clamp 的值（mask=0）梯度正常传播
 *
 * QAT Mask 对应关系（前向 → 反向）：
 *   - x_mask [T*N, C] → dx
 *   - h0_mask [N, H] → dh（最终传回初始状态的梯度）
 *   - W_mask [C, H*3] → dW
 *   - R_mask [H, H*3] → dR
 *   - bw_mask [H*3] → dbw
 *   - br_mask [H*3] → dbr
 *   - weight_ih_linear_mask [T*N, H*3] → dp
 *   - weight_hh_linear_mask [T*N, H*3] → dq
 *   - gate_mask [T*N, H*3] → 门梯度（在 pointwise kernel 中处理）
 *   - h_mask [T*N, H] → 隐状态梯度（在 pointwise kernel 中处理）
 *
 * 注意：
 *   - 反向传播使用的是反量化后的浮点值，梯度计算已经是正确的
 *   - 不需要rescale补偿，因为前向传播中的rescale已经在反量化时处理了
 *
 * 模板参数 T: 数据类型（float 或 double）
 */
template <typename T>
class BackwardPassQuant {
public:
    /// @brief 构造函数
    /// @param batch_size 批大小
    /// @param input_size 输入维度
    /// @param hidden_size 隐藏层维度
    /// @param blas_handle cuBLAS 句柄
    /// @param stream CUDA 流（可选）
    BackwardPassQuant(int batch_size, int input_size, int hidden_size,
                      const cublasHandle_t &blas_handle,
                      const cudaStream_t &stream = 0);

    ~BackwardPassQuant();

    /// @brief 多步反向传播
    /// @param steps 时间步数
    /// @param W_t [H*3, C] 输入权重转置
    /// @param R_t [H*3, H] 循环权重转置
    /// @param bw [H*3] 输入偏置
    /// @param br [H*3] 循环偏置
    /// @param x_t [C, N*T] 输入转置（所有时间步）
    /// @param h [N*(T+1), H] 所有隐状态（包含初始状态）
    /// @param v [N*T, H*4] 所有中间值
    /// @param dh_new [N*(T+1), H] 所有隐状态梯度
    /// @param dx [N*T, C] 输入梯度（输出）
    /// @param dW [C, H*3] 输入权重梯度（累加）
    /// @param dR [H, H*3] 循环权重梯度（累加）
    /// @param dbw [H*3] 输入偏置梯度（累加）
    /// @param dbr [H*3] 循环偏置梯度（累加）
    /// @param dh [N, H] 传递到初始状态的梯度（输出）
    /// @param dp [N*T, H*3] dp 中间梯度
    /// @param dq [N*T, H*3] dq 中间梯度
    /// @param zoneout_mask [N*T, H] Zoneout mask（可选）
    /// @param x_mask [N*T, C] 输入量化 clamp mask
    /// @param h0_mask [N, H] 初始隐状态量化 clamp mask
    /// @param W_mask [C, H*3] 权重 W 量化 clamp mask
    /// @param R_mask [H, H*3] 权重 R 量化 clamp mask
    /// @param bw_mask [H*3] 偏置 bw 量化 clamp mask
    /// @param br_mask [H*3] 偏置 br 量化 clamp mask
    /// @param weight_ih_linear_mask [N*T, H*3] W*x+bw 输出 clamp mask
    /// @param weight_hh_linear_mask [N*T, H*3] R*h+br 输出 clamp mask
    /// @param gate_input_mask [N*T, H*3] 门输入 clamp mask
    /// @param gate_output_mask [N*T, H*3] 门输出 clamp mask
    /// @param h_mask [N*T, H] 隐状态输出 clamp mask
    void Run(int steps, const T *W_t, const T *R_t, const T *bw, const T *br,
             const T *x_t, const T *h, const T *v, const T *dh_new,
             T *dx, T *dW, T *dR, T *dbw, T *dbr, T *dh, T *dp, T *dq,
             const T *zoneout_mask = nullptr,
             // QAT masks
             const uint8_t *x_mask = nullptr,
             const uint8_t *h0_mask = nullptr,
             const uint8_t *W_mask = nullptr,
             const uint8_t *R_mask = nullptr,
             const uint8_t *bw_mask = nullptr,
             const uint8_t *br_mask = nullptr,
             const uint8_t *weight_ih_linear_mask = nullptr,
             const uint8_t *weight_hh_linear_mask = nullptr,
             const uint8_t *gate_input_mask = nullptr,
             const uint8_t *gate_output_mask = nullptr,
             const uint8_t *h_mask = nullptr);

private:
    void IterateInternal(const T *R_t, const T *h, const T *v, const T *dh_new,
                         T *dbw, T *dbr, T *dh, T *dp, T *dq,
                         const T *zoneout_mask,
                         const uint8_t *weight_hh_linear_mask,
                         const uint8_t *gate_input_mask,
                         const uint8_t *gate_output_mask,
                         const uint8_t *h_mask,
                         const uint8_t *bw_mask,
                         const uint8_t *br_mask);

    struct private_data;
    private_data *data_;
};

}  // namespace gru
