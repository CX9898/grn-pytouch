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

> 输入x类型为float时则先经过量化预处理float -> int8

### ForwardPass Run方法

```C++
template<typename T -> int8, typename AccumT -> int32>
void ForwardPass<T, AccumT>::Run(const int steps, // 时间步数, 序列长度T
                                 const T *W,   // [C,H*3], 输入到隐藏状态的权重矩阵(Wx), 对应 GRU 的三个门(z、r、h)。C 是输入特征维度，H 是隐藏状态维度
                                 const T *R,   // [H,H*3], 隐状态到隐藏状态的权重矩阵(Rh)，对应 GRU 的三个门(z、r、h)
                                 const AccumT *bx,  // [H*3], 输入偏置(bias for W)，对应 z、r、h 门
                                 const AccumT *br,  // [H*3], 隐状态偏置(bias for R)，对应 z、r、h 门
                                 const T *x,   // [N,C], 输入序列，batch_size = N，特征维度 = C
                                 T *h,         // [N,H], 输出隐藏状态，每个时间步保存的 GRU 隐状态
                                 T *v,         // [N,H*4], 临时存储向量/中间计算值，通常保存 z, r, h_tilde, h_new 的中间值，用于后向传播或 zoneout
                                 AccumT *tmp_Wx,    // [N,H*3], W * x 的临时结果
                                 AccumT *tmp_Rh,    // [N,H*3], R * h 的临时结果
                                 const float zoneout_prob, // Zoneout 概率，用于随机丢弃部分隐藏状态
                                 const T *zoneout_mask, // Zoneout mask，0/1 矩阵，控制哪些隐藏单元被保留,  // Zoneout mask [N,H]
                                 const QuantParams &quantParams // 形式是怎么样的? 如果内部自己算就不需要传入
                                 ){}

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
```

**输入:**

- x输入进入的是int8还是float? (float)x的话则先进行 -量化-> (int8)x
- 其他参数是怎么得到? 例如W, R, bx, br. 使用训练阶段得到的float, 然后部署前量化成int8

**流程:**

1. 如果不是在外部传入各个scale的话, 先查找x, W, R, h的最大最小值用于计算各个scale和zero_point.
2. 调用cuBLAS::gemm<int8>计算 (int8)W * (int8)x, 输出(int32)Wx // int8 * int8 -> int32
3. (int32)Wx的scale_Wx = s_w * s_x. (parameter scaling)
4. for i in 0..steps-1: 循环每个时间步 (每个时间步都要更新scale, 除了W)
    1. 先查找x, W, R, h的最大最小值用于计算各个scale和zero_point. (上一步最大最小值占比90%, 当前最大最小值占比10%)
    2. 调用cuBLAS::gemm<int8>计算 (int8)R * (int8)h, 输出(int32)Rh // int8 * int8 -> int32
    3. (int32)Rh 的 scale_Rh = scale_R * scale_h.  (parameter scaling)
    4. 执行逐元素运算(CUDA Kernel)
        1. 计算索引
        2. GRU前向计算(三个门)
            - 各个门计算前保证各个变量的scale一致. 例如:Wx: scale_Wx, Rh: scale_Rh, bx: scale_bx(通常等于 scale_Wx), br:
              scale_br(通常等于 scale_Rh). 统一选择对其到scale_Wx
            -
            - **更新门z:**
            - int32_t Rh_aligned = ((int32)Rh[z_idx] * M_Rh_z + (1 << (shift_Rh - 1))) >> shift_Rh_z; // 对齐到 Wx 的
              scale
            - int32_t br_aligned = ((int32)br[bz_idx] * M_br_z + (1 << (shift_br - 1))) >> shift_br_z; // 对齐到 Wx 的
              scale
            - int32 z_tmp_i32 = (int32)Wx[z_idx] + (int32)Rh_aligned + (int32)bx[bz_idx] + (int32)br_aligned; // 更新门计算公式
            - int8 z_tmp_i8 = quantize_i32_to_int8(z_tmp_i32, ); // 量化为int8
            - const int8 z = dev::sigmoid_int8_lut(z_tmp_i8); // 更新门z
            -
            - **重置门r**
            - // todo: scale对齐
            - int32 r_tmp_i32 = (int32)Wx[r_idx] + (int32)Rh[r_idx] + (int32)bx[br_idx] + (int32)br[br_idx]; // 重置门计算公式
            - int8 r_tmp_i8 = quantize_i32_to_int8(r_tmp_i32, scale_r, zero_point_r); // 量化为int8
            - const int8 r = dev::sigmoid_int8_lut(r_tmp_i8); // 重置门r
            -
            - **候选状态~ht**
            - // todo: scale对齐
            - int64_t tmp_r_Rh_bx = (int64_t)r_int32 * (int64_t)((int32)Rh[g_idx] + (int32)br[bg_idx]);
            - int32_t scaled_r_Rh_bx = (int32_t)(((tmp_r_Rh_bx * M_rRh_to_g) + rounding) >> shift_rRh_to_g);
            - int32 g_tmp_i32 = (int32)Wx[g_idx] + (int32)scaled_r_Rh_bx + (int32)bx[bg_idx]; // 候选状态计算公式
            - int8 g_tmp_i8 = quantize_i32_to_int8(g_tmp_i32, scale_g, zero_point_g); // 量化为int8
            - const int8 g = dev::tanh_int8_lut(g_tmp_i8); // 候选状态~ht
            -
            - 如果开启训练模式: 将中间值 z, r, g 保存到v
            -
            - int8 h_t = (int8)z * (int8)h[output_idx] + (static_cast<int8>(1.0) - (int8)z) * (int8)g;// 当前时间步最终隐藏状态ht
            - 如果启用Zoneout: 对GRU 隐藏状态随机保留

```C++
template<bool use_inv_scale>
__device__ __forceinline__ int8_t quantize_float_to_int8(
    const float value,
    const float scale_param,
    const int32_t zero_point
) {
    // 编译期分支：根据use_inv_scale选择计算方式(无运行时开销)
    const float scaled = [value, scale_param]() {
      if constexpr (use_inv_scale) {
          // 分支1：用inv_scale，乘法(编译期确定，仅当use_inv_scale=true时保留)
          return value * scale_param;
      } else {
          // 分支2：用scale，除法(编译期确定，仅当use_inv_scale=false时保留)
          return value / scale_param;
      }
    }();

    const float shifted = scaled + static_cast<float>(zero_point);
    const int32_t rounded = __float2int_rn(shifted); // 四舍五入
    const int32_t clamped = max(-128, min(127, rounded)); // 范围截断
    return static_cast<int8_t>(clamped);
}

host:
{
    float scale_Wx = ...;
    float scale_Rh = ...;
    float ratio = scale_Rh / scale_Wx;
    int32_t N = 15;                   // 推荐经验值, int8 通常右移 15~20 bits, int16 可右移 22~24 bits
    int32_t M_r = int32_t(round(ratio * (1 << N)));
    int32_t shift_r = N;
    int32_t zp_r = 0;                  // 对称量化
}

__device__ __forceinline__ int8_t quantize_i32_to_int8(
    const int32_t value,      // int32 累加结果
    const int32_t M,          // host 端预先计算好的整数缩放系数
    const int32_t shift,      // host 端计算好的右移位数
    const int32_t zero_point  // zero_point
) {
    int32_t tmp = (value * M + (1 << (shift - 1))) >> shift; // 四舍五入
    tmp += zero_point;
    tmp = max(-128, min(127, tmp)); // clamp 到 int8
    return static_cast<int8_t>(tmp);
}

__device__ __forceinline__ int16_t quantize_i32_to_int16(
    const int32_t value,
    const int32_t M,
    const int32_t shift,
    const int32_t zero_point)
{
    int32_t tmp = (value * M + (1 << (shift - 1))) >> shift;
    tmp += zero_point;
    tmp = max(-32768, min(32767, tmp));
    return static_cast<int16_t>(tmp);
}

host:
{
    // 全局作用域声明
    __constant__ int8_t d_sigmoid_lut[256]; // 全局常量
    __constant__ int8_t d_tanh_lut[256]; // 全局常量
    
    int8_t h_sigmoid_lut[256];
    int8_t h_tanh_lut[256];
    
    for(int i = 0; i < 256; i++){
        int8_t x = i - 128;         // [-128,127]
        float fx = x / 128.0f;      // 转 float [-1,1]
        float s = 1.f / (1.f + expf(-fx));
        float t = tanhf(fx);
    
        h_sigmoid_lut[i] = static_cast<int8_t>(roundf(s * 127.f));
        h_tanh_lut[i] = static_cast<int8_t>(roundf(t * 127.f));
    }
    cudaMemcpyToSymbol(d_sigmoid_lut, h_sigmoid_lut, sizeof(int8_t) * 256); // 从host端拷贝到device端中编译期固定的地址
    cudaMemcpyToSymbol(d_tanh_lut,    h_tanh_lut,    sizeof(int8_t) * 256); // 从host端拷贝到device端中编译期固定的地址
  
}


__device__ __forceinline__ int8_t sigmoid_int8_lut(int8_t x, const int8_t* lut) {
    // x in [-128,127], lut 长度 = 256
    const int8_t x_clamped = max(-128, min(127, x));
    return lut[static_cast<uint8_t>(x_clamped)]; // uint8_t 转索引 [0,255]
}

__device__ __forceinline__ int8_t tanh_int8_lut(int8_t x, const int8_t* lut) {
    const int8_t x_clamped = max(-128, min(127, x));
    return lut[static_cast<uint8_t>(x_clamped)];
}

__device__ __forceinline__ int8_t sigmoid_int16_lut(int16_t x, const int8_t* lut) { (TODO: 二项式拟合查表方式)
    // 将 int16_t 范围 [-32768, 32767] 映射到 int8_t 范围 [-128, 127]
    // 公式：idx = round( (x + 32768) * (255.0f / 65535.0f) ) - 128
    // 整数优化：避免浮点运算，用移位实现近似缩放
    int32_t tmp = static_cast<int32_t>(x) + 32768; // 转为 [0, 65535]
    tmp = (tmp * 255 + 65535 / 2) / 65535; // 四舍五入缩放到 [0, 255]
    int8_t idx = static_cast<int8_t>(tmp - 128); // 转为 [-128, 127]
    return lut[static_cast<uint8_t>(idx)];
}

__device__ __forceinline__ int8_t tanh_int16_lut(int16_t x, const int8_t* lut) { (TODO: 二项式拟合查表方式)
    // 与 sigmoid 完全相同的索引映射逻辑
    int32_t tmp = static_cast<int32_t>(x) + 32768; // int16_t [-32768, 32767] → [0, 65535]
    tmp = (tmp * 255 + 65535 / 2) / 65535; // 缩放到 [0, 255]（四舍五入）
    int8_t idx = static_cast<int8_t>(tmp - 128); // → [-128, 127]
    return lut[static_cast<uint8_t>(idx)]; // 用索引访问 tanh LUT
}
```

> 疑问:
> - 量化方式选择? 对称量化(权重全用对称); 非对称量化; 混合量化(V). 给定参数选择其他选择是否对称量化
> - > 选择混合量化. 权重使用对称量化, 其他参数则给定参数选择是否对称量化.
> - scale是传入进来还是内部计算得到? scale = (max_float - min_float) / (max_int - min_int)
> - > scale是内部计算
> - scale是第一次计算好之后一直使用还是每个时间步开始前计算?
> - > scale每个时间步都需要重新计算, 并且每个时间步的scale计算方式为: (上一步最大最小值占比90%, 当前最大最小值占比10%)
> - 门控计算的量化步骤中, scale和zero_point的值是相同的还是根据每个门控都不同?
> - > 每个门控的都不同

## GruTrain

### **初始化**

首先根据是否在内部做量化的来分支.

- 如果开启内部量化: 传进来的参数则是float类型, 第一步先将每个参数进行量化(int8/int16), 计算得到Q值或scale和zp并转换成M和shift.
- 如果不开启: 传进来的参数则是int8/int16类型, 并且还需要传入各个参数的量化参数(Q值或scale,zp或M,shift)

> - x 是非对称量化.
> - x 是每个时间步的scale不同, 也就是 x_scale[steps], x_zp[steps]. 这其中需要用(上一时间步最大最小值占比90%,
    当前最大最小值占比10%)的做法吗?
> - W, R, bx(偏置), br(偏置) 是对称量化
> - W, R 中分为三个门的scale, 也就是 W_scale[3], R_scale[3]. // z, r, g

### **前向传播:**

1. 首先调用 cuBlas::GEMM 提前计算好 W * x.
2. 计算 Wx_scale[steps * 3个门] 并转换成 M 和shift的形式 // 可以在初始化的阶段做, 或外部传入
3. 由于 x 是非对称, Wx 需要进行零点补偿.

   > - 首先要计算 W_sum, 也就是 W 每行的和 // 是否可以初始化的时候做, 或者外部传入
   > - 然后遍历 Wx的所有元素, 每一行的值都要减去当前行的 x_zp * W_sum[当前行]

4. for i in 0..steps-1: 循环每个时间步
    1. 调用 cuBlas::GEMM 算好 R * h.
    2. 执行逐元素并行运算(CUDA Kernel)
        1. Rh_aligned_z = rescale(Rh_z); bx_aligned_z = rescale(bx_z); br_aligned_z = rescale(br_z);
        2. const int32_t z_tmp_i32 = Wx[z_idx] + Rh_aligned_z + bx_aligned_z + br_aligned_z;
        3. const int8_t z_tmp_i8 = dev::quantize_i32_to_i8(z_tmp_i32, rescale_Wx_z_to_z.M, rescale_Wx_z_to_z.shift);
        4. z = dev::sigmoid_int8_lut(z_tmp_i8); // 更新门z

        5.
    3. 计算当前时间步的h的scale.
   > - h 是对称量化?
   > - 是不是只需要计算h的scale