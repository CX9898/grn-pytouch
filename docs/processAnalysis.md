# Haste GRU int8 量化 定点算法

## GruInference

### 创建 ForwardPass 类对象

```C++
ForwardPass(
const bool training,
const int batch_size,
const int input_size,
const int hidden_size,
const cublasHandle_t &blas_handle,
const cudaStream_t &stream = 0);

// training: `true` if the caller intends to perform a backward pass to compute gradients.
// batch_size: the number of training/inference inputs provided in each tensor.
// input_size: the dimension of each input vector.
// hidden_size: the expected dimension of each output vector.
// blas_handle: an initialized cuBLAS handle (see `cublasCreate`).
```

### ForwardPass Run方法

```C++
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
                                 const T *zoneout_mask, // Zoneout mask，0/1 矩阵，控制哪些隐藏单元被保留,  // Zoneout mask [N,H]
                                 const float *scale, // 形式是怎么样的?
                                 const float *zero_point // 形式是怎么样的?
                                 )
```

1. 调用cuBLAS::gemm<int8>计算 (int8)W * (int8)x, 输出(int32)Wx // int8 * int8 -> int32
2. (int32)Wx 量化为 (int8)Wx // 是否需要? (可以手写GEMM在内部做量化)
3. for i in 0..steps-1: 遍历每个时间步
    1. 调用cuBLAS::gemm<int8>计算 (int8)R * (int8)h, 输出(int32)Rh // int8 * int8 -> int32
    2. (int32)Rh 量化为 (int8)Rh // 是否需要? (可以手写GEMM在内部做量化)
    3. 执行逐元素运算(CUDA Kernel)
        1. 计算索引
        2. GRU前向计算(三个门)
            - int32 z_tmp_i32 = (int32)Wx[z_idx] + (int32)Rh[z_idx] + (int32)bx[bz_idx] + (int32)br[bz_idx];
            - int8 z_tmp_i8 = quantize_i32_to_i8(z_tmp_i32);
            - const int8 z = dev::sigmoid(z_tmp_i8); // 更新门z
            - int8 r_tmp_i8 = quantize_i32_to_i8(Wx[r_idx] + Rh[r_idx] + bx[br_idx] + br[br_idx]);
            - const int8 r = dev::sigmoid(r_tmp_i8); // 重置门r
            - int8 r_tmp_i8 = quantize_i32_to_i8(Wx[g_idx] + r * (Rh[g_idx] + br[bg_idx]) + bx[bg_idx])
            - const int8 g = dev::tanh(r_tmp_i8); // 候选状态~ht
            - int32 h_t_tmp_i32 = (int8)z * (int32)h[output_idx] + (static_cast<int32>(1.0) - (int8)z) * (int8)g;
            - int8 h_t_tmp_i32 = quantize_i32_to_i8(h_t_tmp_i32); // 当前时间步最终隐藏状态ht

> 需要做的:
> 构建quantize_i32_to_i8
> 构建sigmoid的定点化方法
> 构建tanh的定点化方法

## GruTrain


