# 更新日志 (Changelog)

本文档记录 Quant-GRU-PyTorch 项目的所有重要更新和变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [Unreleased]

### 新增 (Added)

1. **统一 POT scale 编码入口 `encodeScaleResult(ContinuousScaleResult, PotScaleMethod)`**
   - 新增 `PotScaleMethod`：`Round` / `CoverRange`（默认）/ `Floor`，挂在 `OperatorQuantConfig.pot_scale_method_`（容差 `pot_scale_tolerance_=0.02`）。
   - 核心取整 `scaleToPowerOfTwo` 与 `PotScaleMethod` 收敛到底层头 `pot_scale_encode.h`（零项目内依赖），与 AIMET `find_closest_power_of_2_scale(cover_range)` 对齐：`real_range≈2^k` 则 round，否则 floor 覆盖。`quantize_bitwidth_config.h` / `histogram_collector.h` 改为包含该头，消除重复定义。
   - MinMax（`encodeFromRange`）、CPU/GPU 直方图、以及 `calibrateQuantParams` 均走同一编码路径，不再按调用方硬编码 round vs floor。
   - Python 绑定暴露 `pot_scale_method_` / `pot_scale_tolerance_`。

### 变更 (Changed)

1. **头文件职责按依赖分层，量化 scale 编码从算子层 / 校准层剥离**
   - `pot_scale_encode.h`（底层，零项目内依赖）：`PotScaleMethod` + `checkedShiftToInt8` / `isNearPowerOfTwo` / `scaleToPowerOfTwo`。`quantize_bitwidth_config.h`、`histogram_collector.h` 改为包含此头，消除此前 `scaleToPowerOfTwo` 在 `histogram_collector.h` 与 `pot_scale_encode.h` 的重复定义。
   - 新增 `scale_encoding.h`（编码层，位于 `quantize_param_types.h` 之上）：迁入 `encodeMShift` / `decodeMShift` / `pot2Shift` / `toFixedScale(s)` / `makeRescale*`（原 `quantize_ops_helper.h`）与 `ContinuousScaleResult` / `EncodedScaleResult` / `convertToPot` / `encodeScaleResult` / `printParms`（原 `pot_sqnr_calibrator.h`）。
   - `quantize_ops_helper.h` 回归纯运行时量化算子；`pot_sqnr_calibrator.h` 回归纯 SQNR / Percentile / MinMax range 连续 scale 校准。`gru_quant.h` / `gru_quant_cpu.h` 默认包含 `scale_encoding.h`，使用方无需改动。

### 修复 (Fixed)

### 性能 (Performance)

### 破坏性变更 (Breaking Changes)

1. **直方图 POT2 默认由纯 Round 改为 CoverRange**：在 `usePOT2_=true` 时，原先可能把 scale 四舍五入调小（裁剪极端值）的算子，现默认改为覆盖优先（与 AIMET `apply_power_of_2_workflow(method="cover_range")` 一致）。若需旧行为，设 `pot_scale_method_=0`（Round）。

### 说明 (Notes)

---

## [1.0.10] - 2026-06-30

### 变更 (Changed)

1. **`GRUQuantParams` 收敛为单一权威量化种子（内部重构，无 Python API 破坏）**
   - **动机**：同一意义的量化数值此前以多套表示并存（`shift_*`、`raw_scale_*`、`fixed_scale_*` 及 per-tensor/per-gate/per-channel 多份副本），极易出现"用错版本"或"更新一处漏更新另一处"的隐藏 BUG，缺乏数据唯一性。
   - **权威种子**：新增 `QuantParam`（仅 `scale` + `zero_point`）与 per-channel 的 `ChannelQuantParam`。`include/quantize_param_types.h` 类型层回归纯数据，不再内嵌任何算子逻辑；执行表示改为强类型 `Pot2Rescale` / `FixedPointScale` / `FloatRescale`。
   - **派生算子族（自由函数，归入 `include/quantize_ops_helper.h`）**：`pot2Shift`、`encodeMShift`、`toFixedScale(s)`、`makeRescale*`。POT2 用整数 shift 差；仿射用**原始连续 scale 比值** + `encodeMShift`（绝不经 `fixed_scale` 往返，避免精度损失）；FP 用 `1/ratio`。新增重载原语 `applyRescale` 统一三种 rescale 的应用。
   - **设备层模板化 + per-channel 统一**：设备结构体按 rescale 类型模板化（`GateQuantParamsT<R>`、`LinearQuantParamsGPUT/CPUT<R>`），device 侧线性参数统一为 per-channel；kernel 模板化、`Run()` 按 `R` 编译期派发，取代运行时 `usePOT2` 分支。FP 路径保留独立 kernel，仅复用 `applyRescale`。
   - **范围**：校准（MINMAX / 直方图 / GPU SQNR）、权重量化、pybind 绑定（`from_cpp` / `to_cpp`）全部对齐到 `QuantParam` / `ChannelQuantParam`。
   - **验证**：容器 `quant-gru-cuda128` 内 `gru_quant_shared`、`gru_example` 均编译通过；POT2 输出 bit 级一致，仿射不劣于浮点参考。

2. **`GRUQuantParamsPy`（pybind 绑定）权重/偏置 scale 收敛为单一 per-channel 数组**
   - **动机**：绑定结构体此前为 W/R/bw/br 各存了三套 scale（per-channel `scale_W_` + per-tensor `scale_W_tensor_` + per-gate `scale_W_gate_`），与上次 C++ 侧消除的多版本冗余是同一问题：多个可变字段易出现"用错版本/漏更新"。
   - **收敛**：仅保留 per-channel 权威数组 `scale_{W,R,bw,br}_`（始终 3*hidden，按粒度广播）；tensor 视图 = `scale_W_[0]`、gate 视图 = `scale_W_[g*hidden]`，纯索引还原。删除 8 个 `_tensor_` / `_gate_` 字段及其 `def_readwrite` 注册。
   - **`from_cpp`**：只读 per-channel 数组，不再派生 tensor/gate 副本；**`to_cpp`**：一律从 per-channel 数组展开，`granularity` 仅作元数据写回——顺带修掉 "granularity=PER_TENSOR 且未设 tensor 副本时灌入 0 scale" 的潜在 bug。
   - **清理**：删除已无调用方的死代码 `encode_runtime_scale` 及其 `encode_per_tensor_scale` / `encode_per_gate_scale` / `encode_per_channel_scale` 辅助函数（旧多粒度 runtime-encode 逻辑）。
   - Python 消费方仅读 per-channel 数组（如 `quantize_per_channel(..., list(params.scale_W_), ...)`），无需改动。

3. **POT2 量化模式默认关闭**：`use_pot2_scale` / `usePOT2_` 的默认值由 `true` 改为 `false`，未显式指定时量化默认走 affine（`M + shift`）编码而非 POT2（`M=1`、仅 shift）。
   - 涉及 C++ 头文件 `include/quantize_bitwidth_config.h`、CPU-only 头文件 `quant-gru-cpu-only/include/quantize_bitwidth_config.h`、pybind 绑定结构体 `pytorch/lib/gru_interface_binding.cc`、默认配置 `pytorch/config/gru_quant_bitwidth_config.json`。
   - Python 端 `QuantGRU` 构造函数、反序列化（`__setstate__`）、`use_pot2_scale` property getter，以及加载旧导出文件时 `model_info.get("use_pot2_scale", ...)` 的兜底默认值统一改为 `False`，保证各入口默认行为一致。

### 修复 (Fixed)

1. **修复配置变更后导出过期量化参数（stale scale）的问题**
   - **影响**：`QuantGRU` 采用懒同步，改配置（`use_pot2_scale` / 位宽 / granularity / 重新收集校准数据等）只标记 `_quant_params_dirty=True`，真正重算 `quant_params` 推迟到下一次 `forward`；而导出路径不经过 `forward`，会静默导出与当前配置不一致的旧 `scale`。
   - 新增 `_ensure_quant_params_fresh_for_export()`，在 AIMET encodings 导出与 `_export_quant_params_impl` 两处导出前对齐 `quant_params`，语义与 `forward` 中的 dirty 处理一致（锁定且已有参数时仅清脏不覆盖）。
   - 校准数据已丢失（如 pickle/deepcopy 后）且参数已过期时，明确抛 `RuntimeError` 而非静默导出错误结果，并提示重新校准或在改配置后导出前调用 `finalize_calibration()`。

### 破坏性变更 (Breaking Changes)

1. **默认量化编码由 POT2 改为 affine**
   - **影响**：此前未显式设置 `use_pot2_scale` 即默认走 POT2；改动后默认走 affine（`M + shift`）。依赖旧默认行为的调用方/示例脚本/配置若未显式指定 `use_pot2_scale=True`（或 JSON `use_pot2_scale: true`），量化编码方式会静默切换。需要 POT2 的场景请显式开启。

### 说明 (Notes)

1. **Python 对外 API**：标量/激活类 `scale_*` / `zp_*` 与 per-channel 数组 `scale_{W,R,bw,br}_` 字段语义保持不变；仅移除了从未被消费方使用的冗余绑定属性 `scale_*_tensor_` / `scale_*_gate_`（仓库内无任何读写方）。本次主体为 C++/绑定内部存储结构的去冗余重构，故按补丁号递增。

---

## [1.0.9] - 2026-06-26

### 修复 (Fixed)

1. **修复 POT2 模式下导出/落盘 scale 仍为连续校准值的问题**
   - **影响**：POT2 模式下 `export_quant_params` 导出的 `scale` 不是 `2^n`，与内核定点 scale 及编译器 POT2 假设（`M=1`、仅 shift）不一致。
   - 校准阶段 `convertToPot` 已计算出 `po2_scale = 2^{-shift}`（`M=1`），但 `raw_scale_*` 此前统一写入 `continuous_scale`。
   - 新增 `storedScaleForMode()`：POT2 落盘 `effective_scale`（`po2_scale`），Affine 仍保留 `continuous_scale`（由编译器侧 `encodeMShift` 推导 `M+shift`）。
   - `gru_interface.cc` 中 MINMAX / CPU 直方图 / GPU SQNR 全部校准路径统一使用该 helper 写入 `raw_scale_*`。
   - 更新 `quantize_param_types.h` 中 `raw_scale_*` 字段注释，明确其为导出/运行时 scale 语义。
   - 新增测试 `pytorch/test_export_pot2_scales.py`：校验 POT2 导出 scale 全部为精确 `2^n`，Affine 对照组仍保留连续 float scale。

2. **修复 `qmax_auto_scale()` 导致的 scale 与 AIMET（及下一层）不一致**
   - **影响**：本层输出 scale 与下一层输入 scale 对不齐，影响 AIMET 链路一致性。
   - 标定阶段统一以 `num_steps = qmax_auto_scale() - qmin_auto_scale()` 计算量化级数与 scale，但 `qmax_auto_scale()` 历史上为追求"对称性"对有符号数返回 `1 << (bits_ - 1)`（int8: 128）、无符号数返回 `1 << bits_`（uint8: 256），使得 `num_steps = 256`，scale = `range/256`。而 AIMET 在所有路径均使用 `num_steps = 2^bits - 1 = 255`（int8 量化范围为 **-128~127**），scale = `range/255`。
   - `quantize_bitwidth_config.h::qmax_auto_scale()`：有符号数改为 `(1 << (bits_ - 1)) - 1`、无符号数改为 `(1 << bits_) - 1`，与 `qmax()` 及 AIMET 完全一致（int8: 127, uint8: 255）。
   - 该修正自动传播至 `calibrateQuantParams`（MINMAX）、`pot_sqnr_calibrator.h`、`gru_interface.cc`、`calibration_gpu.cu` 等所有以 `num_steps` 推导 scale 的路径；对称 INT 路径此后 `num_pos_steps=127`、`num_neg_steps=128`、`offset=-128`，与 AIMET 对齐。

---

## [1.0.8] - 2026-06-23

### 修复 (Fixed)

1. **支持 32bit bias 量化（修复标定阶段 `num_steps` 整型溢出）**
   - **影响**：`bias_ih`/`bias_hh` 设为 32bit 时校准直接报错 `num_steps must be > 0`。
   - 标定阶段以 `int32` 计算量化级数 `num_steps = qmax_auto_scale() - qmin_auto_scale()`，在 32bit 下 `2147483647 - (-2147483648)` 发生 int32 溢出变为负数。同时多处下游又把 `int64` 的 `num_steps` 截断回 `int`（如 `get_minimum_scale`），会得到负的 `minimum_scale` 并静默产生错误 scale。
   - `get_minimum_scale`：入参由 `int` 改为 `int64_t`，消除截断。
   - `pot_sqnr_calibrator.h` / `gru_interface.cc` / `calibration_gpu.cu`：减法前显式 `static_cast<int64_t>`，并去掉所有 `static_cast<int>(num_steps)`。
   - `quantize_ops_helper.h::calibrateQuantParams`（MINMAX 路径）：`num_steps`、`num_pos_steps`、`num_neg_steps` 统一改为 `int64_t`。

### 变更 (Changed)

1. **POT shift 越界防御断言**: 新增 `checkedShiftToInt8()`，在 POT scale 编码（`roundScaleToPowerOfTwo` / `convertToPot` / `calibrateQuantParams`）以及 bias 的 per-tensor/per-gate/per-channel rescale shift 计算处，对收窄到 `int8` 的 shift 做范围校验，超界时直接抛错，避免静默截断成错误 scale。

### 说明 (Notes)

1. **forward 路径未改逻辑**: bias 前向通路本就是 `int32` 存储 + `int64` 累加，`qmin()/qmax()` 对 `bits>=32` 已特判返回 int32 极值，clamp 到 int32 全域为无操作，故 forward 无需为 32bit bias 改动；本次 forward 侧仅新增上述防御断言，不改变正确路径数值结果。

---

## [1.0.7] - 2026-06-22

### 变更 (Changed)

1. **量化参数导出字段形态优化**: 导出量化参数时，`scale`、`zero_point`、`real_min`、`real_max` 等量化字段仅在存在多个元素时输出为列表；per-tensor 等单元素量化参数输出为标量，减少下游解析时对单元素列表的特殊处理。

### 破坏性变更 (Breaking Changes)

1. **ONNX 导出 opset 须显式注册**
   - **影响**：`QuantGRU` 的 ONNX forward 路径不再自行写死注册 opset 18；导出前必须调用 `ensure_quant_gru_onnx_registered(opset=...)`，且与 `torch.onnx.export(opset_version=...)` 一致。
   - `ensure_quant_gru_onnx_registered(opset=...)` 支持 `opset>=13`，并按 opset 分别注册 symbolic。

---

## [1.0.6] - 2026-06-12

### 新增 (Added)

1. **量化值存储类型开关 `quant_storage_dtype`**: `QuantGRU` 新增 `quant_storage_dtype` 参数，可在量化前向中切换量化值的存储类型：
   - `"float32"`（默认，原行为）：量化值以 float32 存储，走 `forward_quant_float_storage`；
   - `"int32"`：量化值以 int32 存储，走纯定点核心 `forward_quant_int_storage`。
   反向传播按存储类型自动选择对应实现（int32 路径内部转 float 后复用原 backward），训练/推理两端均支持。
2. **纯定点 int 进 int 出推理接口（AIMET INT16_FIXED_EVAL 打通）**:
   - C++ 新增 `quantGRUForwardIntIO`：输入 `x_q` / `h0_q` 为上游已量化的 int32 值（同 `scale_x/zp_x`、`scale_h/zp_h` 网格），内部仅量化权重、跳过输入量化，直出 int32 `h_q`（不反量化）。
   - 新增 binding `forward_quantized_int_io`（int32 输入 → int32 `output_q`/`h_n_q`）。
   - `QuantGRU` 新增 `forward_quantized(input, hx=None)`：纯定点推理边界，仅接受已量化的 int32 输入（非整数输入抛 `TypeError`），输出恒为 int32 量化隐藏状态，满足 `(output_q - zp) * scale == 部署核 fp 输出`。
3. **AIMET 黑盒接入契约方法**: `QuantGRU` 新增 `aimet_capabilities()`、`aimet_configure(mode)`、`get_io_quant_meta()`，用于 AIMET 适配层的能力上报、模式配置与输入/输出/隐藏态量化元数据查询。
4. **有效定点 scale 解码**: 新增 binding `decode_effective_scale(scale, usePOT2)`，返回内核实际使用的（POT2/M16 编码后）有效 scale，保证 `get_io_quant_meta` 上报的 scale 与定点输出 bit-exact 一致。

### 变更 (Changed)

1. **输入/隐藏态校准改用全局 min/max（与 AIMET 对齐）**: 输入 `x` 与隐藏态 `h` 的 MinMax 校准范围统计由原先的「分时间步 EMA（decay=0.9，顺序相关）」改为顺序无关的全局 min/max。
   - 修复双向 GRU 正/反向因 EMA 顺序相关导致 `scale_x/zp_x` 不一致的问题：正/反向现共享同一输入网格，双向纯定点 int 进 int 出可达 bit-exact。
   - 与 AIMET 的全局 MinMax 校准语义一致，便于上游量化器对齐。

### 说明 (Notes)

1. **校准跳过初始隐藏状态 h0**: MinMax 与直方图两条校准路径均跳过 `h[0]`（假设无状态 GRU 的 h0 恒为零，避免零值污染 `min_h_/max_h_`）。rxmet 实际接入路径（校准/推理均只传 input、h0 走零初始化）满足该前提。若未来出现有状态/流式 GRU 在校准阶段传入非零 h0，则需改为按需将 h0 纳入范围统计（已在代码注释中标注修法）。

---

## [1.0.5] - 2026-06-04

### 新增 (Added)

1. **普通 scale 支持**: 量化 scale 从原先仅支持 POT2（2 的幂次）形式扩展为支持普通 scale，可使用任意 `M + shift` 组合表示量化缩放参数，提升量化参数配置的灵活性。

### 变更 (Changed)

1. **量化参数导入/导出格式更新**: 导出改为 **scale-only**，算子级仅保留 `scale` 与 `zero_point`（及 `real_min`/`real_max`、`enc_type` 等语义字段），不再输出 `multiplier`、`shift`、`scale_encoding`。`scale` 字段统一表示运行时语义 scale（避免使用 `decode(fixed_scale)` 近似值）。导入侧优先解析 scale-only，同时继续兼容旧 `multiplier/shift` 与 `n/exp2_inv` 格式。
   ```json
   {
     "operators": {
       "input": {
         "dtype": "INT8",
         "symmetric": false,
         "scale": [0.010382890701293945],
         "zero_point": [-5],
         "enc_type": "PER_TENSOR",
         "real_min": [-1.2770955562591553],
         "real_max": [1.3705415725708008]
       }
     }
   }
   ```
2. **ONNX 导出路径更新**: 新增 `export_mode` 单节点 GRU 导出路径，通过 `ensure_quant_gru_onnx_registered(opset=18)` 注册 `custom_gru::quant_gru` / `custom_gru::quant_bigru` symbolic，并在 legacy exporter (`dynamo=False`) 下导出为标准 ONNX `GRU` 节点；支持单向/双向 GRU、`hx=None` 与显式 `hx` 输入。新增 `get_quant_gru_custom_opsets()`、`normalize_quant_gru_onnx_to_optimized_baseline()` 和 `prune_quant_gru_raw_l0_param_encodings()`，用于统一 custom opset 配置、规范化 ONNX 局部命名并清理 AIMET 原生 `*_l0` 参数 encoding。

### 修复 (Fixed)

1. 修复 affine 模式下边界量化导致的精度回退问题。
2. 修复 LUT 表使用错误 scale 的问题。
3. 修复 `encodeMShift` 中 Q mantissa 缺少 16-bit offset 的问题。
4. 修复双向 GRU 导出 AIMET encodings 时，`weight_ih.weight` 和 `weight_hh.weight` 的 `zero_point` 未随反向分支拼接的问题；对称量化权重现在会输出与拼接后 `scale` 等长的全 0 `zero_point` 列表。
5. 修复"对于一开始没有开启POT2模式, 中间开启了POT2模式调用了对应的接口, C++中进行了量化参数的调整, 但没有反应到python端"数据流链路断开的 bug。修复点：所有会改变量化参数生成语义的配置变更，都必须标记 `_quant_params_dirty=True`。

### 破坏性变更 (Breaking Changes)

1. **量化参数导出改为 scale-only**
   - **影响**：依赖 `multiplier`/`shift`/`scale_encoding` 字段的下游解析器需改为读取 `scale`/`zero_point`；导入侧仍兼容旧格式。
2. **ONNX 单节点 GRU 导出新路径**
   - **影响**：导出前须调用 `ensure_quant_gru_onnx_registered(opset=18)` 并完成 symbolic 注册；与旧导出流程不兼容。

### 工程 (Infrastructure)

1. **GitHub Actions 工作流完善**: 新增并完善 Linux CUDA 构建、Docker CI 镜像构建和 Python 包发布流程。
2. **Docker 环境更新**: 更新 Docker 构建环境及配套说明文档，改善 CI 和本地构建的一致性。
3. **安装打包流程优化**: 更新 PyTorch 扩展的安装与打包流程，提升安装脚本的可维护性。

---

## [1.0.4] - 2026-03-24

### 修复 (Fixed)

1. 修复双向 GRU 导入量化参数时反向的输入量化参数没有正确初始化的 bug。

---

## [1.0.3] - 2026-03-19

### 新增 (Added)

1. **双向 GRU 的 AIMET encodings 导出/导入支持**:
   - 导出侧：`param_encodings` 中的权重/偏置量化参数按 **正向在前、反向追加在后** 的规则拼接（例如原本单向 128 个参数 -> 双向 256 个）
   - 导出侧：新增 `activation_encodings[module].internal_ops_reverse` 字段，用于承载反向方向的中间算子量化参数
   - 导入侧：支持从上述 AIMET encodings 结构回读并解析 `quant_params_reverse`

---

## [1.0.2] - 2026-03-13

### 变更 (Changed)

1. **新增 quant_params 锁机制**: 在 `QuantGRU` 中新增 `set_quant_params_locked(locked: bool)` 接口，用于显式控制量化参数是否允许被后续校准覆盖。

---

## [1.0.1] - 2026-02-14

### 变更 (Changed)

1. **安装脚本优化**: 改进 `setup.py` 安装脚本，现在在安装时会自动将 C++ binding 共享库文件 (`libgru_quant_shared.so`) 复制到安装目录的 `lib` 子目录中
   - 支持正常安装模式 (`pip install`) 和开发模式 (`pip install -e`) 下的自动复制
   - 通过 rpath 机制 (`$ORIGIN/lib`) 确保扩展模块在运行时能够正确找到共享库
   - 简化了安装流程，无需手动复制共享库文件

---

## [1.0.0] - 2026-02-13

### 破坏性变更 (Breaking Changes)

1. **算子名称重构**
   - **影响**：旧的 JSON 配置文件和代码中的算子名称须全部更新，否则无法正常工作。
   - `x` → `input` (输入序列)
   - `h` → `output` (隐藏状态输出)
   - `W` → `weight_ih` (输入权重矩阵)
   - `R` → `weight_hh` (循环权重矩阵)
   - `bw` → `bias_ih` (输入偏置)
   - `br` → `bias_hh` (循环偏置)
   - 其他算子名称保持不变（如 `weight_ih_linear`, `update_gate_output` 等）
   - 代码中所有使用旧算子名称的地方（如 `adjust_quant_config("x", ...)`）需更新为新名称
   - 请参考 `pytorch/config/gru_quant_bitwidth_config.json` 查看新的配置格式

### 变更 (Changed)

1. **取消梯度缩放优化**: 取消通过应用内部缩放梯度来解决 TF32 精度问题的优化，该优化原本用于在使用 TensorCore 加速训练时提升数值稳定性。

---

## 开发期记录（< 1.0.0）

> 以下为 1.0.0 正式版之前的开发迭代记录，未单独发版。

### 2026-02-11

#### 变更 (Changed)
1. **量化参数导入导出与 AIMET 适配**: 量化参数导入导出功能与 AIMET 框架适配。

### 2026-02-03

#### 性能 (Performance)
1. **Backward 开启 TensorCore 加速**: 对反向传播过程开启 TensorCore 加速，提升梯度计算性能。

#### 变更 (Changed)
1. **精度阈值动态缩放**: 在反向传播过程中，内部根据精度阈值进行动态缩放，确保计算精度和数值稳定性。

### 2026-01-30

#### 新增 (Added)
1. **权重和 bias 支持 per-tensor 量化**: 支持权重和 bias 使用 per-tensor 量化和计算方式。
2. **权重和 bias 支持 per-gate 量化**: 支持权重和 bias 使用 per-gate 量化和计算方式。

### 2026-01-29

#### 修复 (Fixed)
1. 修复反向时对输入和权重没有进行量化反量化的量化误差感知的 bug。

### 2026-01-27

#### 变更 (Changed)
1. **调整移位顺序**: 优化移位操作的顺序，提升后量化精度。

### 2026-01-26

#### 修复 (Fixed)
1. 修复梯度计算时使用了 TF32 导致梯度精度不足的 bug。
2. 修复梯度计算时，被截断的值也传出了梯度的 bug。

#### 变更 (Changed)
1. 通过将 `roundf` 函数改为和 PyTorch 使用四舍五入逻辑相同的 `rintf` 函数来提升 PTQ 精度。
2. **乘法 scale 融合优化**：将乘法操作（如 `reset_gate * weight_hh_linear`、`update_gate * h_old` 等）的结果直接对齐到目标量化 scale，省略了中间的量化步骤。该优化减少了量化误差累积，提高了数值精度，同时减少了计算开销和参数管理复杂度。

### 2026-01-13

#### 变更 (Changed)
1. **量化计算流程重构**: 从原来的在一个大 kernel 中进行 bias 的累加，改为在线性层同时进行 GEMM 和 bias 的相加，以保持与编译器和硬件实现一致。
2. **统一量化参数命名格式**: 规范化量化参数的命名规则，提高代码可读性和维护性。

### 2026-01-08

#### 新增 (Added)
1. **量化参数导入导出**: 支持量化参数的导入和导出功能。
2. **手动调整导出位宽**: 支持手动调整导出的量化参数位宽。
3. **外部控制符号类型**: 支持算子位宽是 UINT 还是 INT 可以在外部 (JSON) 控制。

#### 修复 (Fixed)
1. 修复校验时是否对称量化与是否有符号量化绑定的 bug。

### 2026-01-07

#### 新增 (Added)
1. **任意位宽支持**: 所有算子支持任意 bit 位宽设置。

#### 修复 (Fixed)
1. 修复分段拟合表全局共享导致多个 GRU 情况下 PTQ 精度下降的 bug。
2. 修复校验时第一个时间步没有正确初始化的 bug。
3. 修复直方图在边缘情况下的处理情况与 AIMET 实现不一致的 bug。

#### 性能 (Performance)
1. 优化各个校验方法的速度。

### 2025-12-31

#### 新增 (Added)
1. **灵活位宽配置**:
   - 支持激活函数输入和输出位宽不一致
   - 支持权重和输入位宽不一致
2. **ONNX 浮点导出**: 支持 ONNX 浮点格式导出。
3. **校准截断控制**: 支持校准时用截断比例控制。
4. **C++ 定点 GRU 实现**: 实现 C++ 版本的纯定点 GRU 用于 Reference Model 验证。

#### 变更 (Changed)
1. **校验流程优化**: 原来不能在 Forward 的时候同时进行校验，必须单独跑校验函数；现已优化支持同时进行。

#### 修复 (Fixed)
1. 修复混合精度配置下中间结果移除导致精度下降的 bug。
2. 修复使用双向 GRU 时分段拟合表不匹配的 bug。
3. 修复 QDQ 导出时，算子位宽与配置不匹配的 bug。

---

## 版本说明

- 标题格式: `[版本号] - YYYY-MM-DD`
- 发版流程: 开发中变更写入 `[Unreleased]`，发版时将对应内容移至新版本标题下，并同步更新 `pytorch/_version.py`
- 更新类型:
  - **新增 (Added)**: 新增的功能特性
  - **变更 (Changed)**: 对现有功能的改进和优化
  - **修复 (Fixed)**: 修复的问题和错误
  - **性能 (Performance)**: 性能相关的改进
  - **工程 (Infrastructure)**: CI、Docker、打包等工程化改进
  - **破坏性变更 (Breaking Changes)**: 不向后兼容的变更；须说明影响与迁移方式
  - **说明 (Notes)**: 已知限制、前提假设等补充说明
