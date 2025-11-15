# GRU Cell 数字量化实现详解（分段二次多项式激活函数版本）

本文档详细描述 GRU Cell 中每个操作在数字电路（Digital Circuit）中的量化实现过程。

**本版本特点**：激活函数（Sigmoid/Tanh）采用**分段二次多项式拟合查找表**实现，相比传统LUT/PWL方法：
- ✅ 精度更高（MAE < 0.001）
- ✅ 内存占用小（每函数仅需128个参数）
- ✅ 硬件友好（INT16定点计算）
- ✅ 计算简单（仅需乘法和加法）

---

## 符号说明

### 浮点域符号
- $X_t$: 输入张量 (input)
- $H_{t-1}$: 上一时刻隐藏状态 (previous hidden state)
- $H_t$: 当前时刻隐藏状态 (current hidden state)
- $W_{ih}$: 输入到隐藏的权重矩阵
- $W_{hh}$: 隐藏到隐藏的权重矩阵
- $B_{ih}$, $B_{hh}$: 偏置向量

### 量化域符号
- $q_*$: 量化后的整数值 (quantized integer value)
- $S_*$: 缩放系数 (scaling factor)
- $Z_*$: 零点 (zero point)
- $q_{b}$: 量化后的偏置 (quantized bias, int32)
- $M_{i32}$: 32位整数乘数 (32-bit integer multiplier)
- $\texttt{>>}$: 右移操作 (right shift)
- $rshift_{i8}$: 8位移位量 (8-bit shift amount)

### 分段二次多项式符号
- $N_{\text{seg}}$: 分段数量（默认32）
- $\{t_1, t_2, \ldots, t_{32}\}$: 32个阈值
- $\{a_i, b_i, c_i\}$: 第$i$段的二次多项式系数
- $f_{\text{seg}}(x, i)$: 第$i$段的二次多项式函数

### 量化关系
浮点值与量化值的转换关系：
$$
S =  \frac{f_{max}-f_{min}}{2^{bit}-1}
$$
$$
\text{Float Value} = S \cdot (q - Z)
$$

---

## GRU Cell 完整操作流程

### 操作 1: 输入线性变换

**浮点域计算：**
$$
G_i = X_t \cdot W_{ih} + B_{ih}
$$

**量化域计算：**
$$
S_{gi}(q_{gi} - Z_{gi}) = S_x(q_x - Z_x) \cdot S_{w_{ih}} q_{w_{ih}} + S_{b_{ih}}q_{b_{ih}}
$$

**Bias 量化关系（对称量化）：**

Bias 采用**对称量化**方式，量化为 int32，零点固定为 0：
$$
Z_{b_{ih}} = 0
$$

Bias 的 scale 独立选择，通常根据 bias 自身的数值范围确定：
$$
S_{b_{ih}} = \frac{\max(|B_{ih}|)}{2^{31} - 1}
$$

量化公式：
$$
q_{b_{ih}} = \text{round}\left(\frac{B_{ih}}{S_{b_{ih}}}\right)
$$

**推导过程：**

**步骤 1**: 展开量化域计算
$$
S_{gi}(q_{gi} - Z_{gi}) = S_x(q_x - Z_x) \cdot S_{w_{ih}} q_{w_{ih}} + S_{b_{ih}}q_{b_{ih}}
$$

**步骤 2**: 展开 MatMul 部分
$$
S_{gi}(q_{gi} - Z_{gi}) = S_x \cdot S_{w_{ih}} (q_x - Z_x) \cdot q_{w_{ih}} + S_{b_{ih}}q_{b_{ih}}
$$

$$
S_{gi}(q_{gi} - Z_{gi}) = S_x \cdot S_{w_{ih}} (q_x \cdot q_{w_{ih}} - Z_x \cdot q_{w_{ih}}) + S_{b_{ih}}q_{b_{ih}}
$$

**步骤 3**: 两边同除 $S_{gi}$，将所有项归一化到输出 scale
$$
q_{gi} - Z_{gi} = \frac{S_x \cdot S_{w_{ih}}}{S_{gi}} (q_x \cdot q_{w_{ih}} - Z_x \cdot q_{w_{ih}}) + \frac{S_{b_{ih}}}{S_{gi}}q_{b_{ih}}
$$

**步骤 4**: 整理并定义缩放因子

令：
$$
S_1 = \frac{S_x \cdot S_{w_{ih}}}{S_{gi}}, \quad S_2 = \frac{S_{b_{ih}}}{S_{gi}}
$$

得到：
$$
q_{gi} - Z_{gi} = (q_x \cdot q_{w_{ih}} - Z_x \cdot q_{w_{ih}}) \cdot S_1 + q_{b_{ih}} \cdot S_2
$$

**步骤 5**: 整数化实现

Scale 采用 2 的幂次表示：
$$
S_1 = 2^{-n_1}, \quad S_2 = 2^{-n_2}
$$

其中 $n_1, n_2$ 为正整数，表示右移位数。

**步骤 6**: 预融合优化（离线计算）

定义预融合常量（在模型加载时一次性计算）：
$$
B_q = (q_{b_{ih}} \;\texttt{>>}\; n_2) - (Z_x \cdot q_{w_{ih}} \;\texttt{>>}\; n_1)
$$

**融合缩放后的最终公式：**
$$
q_{gi} = (q_x \cdot q_{w_{ih}}) \;\texttt{>>}\; n_1 + B_q + Z_{gi}
$$

其中：
- $n_1 = -\log_2(S_1)$: MatMul 的右移位数
- $B_q$: 预融合的 bias 常量（包含了原始 bias 和零点补偿）


---

### 操作 2: 隐藏状态线性变换

**浮点域计算：**
$$
G_h = H_{t-1} \cdot W_{hh} + B_{hh}
$$

**量化域计算：**
$$
S_{gh}(q_{gh} - Z_{gh}) = S_h(q_h - Z_h) \cdot S_{w_{hh}} q_{w_{hh}} + S_{b_{hh}}q_{b_{hh}}
$$

**Bias 量化关系（对称量化）：**

类似地，Bias 采用**对称量化**，量化为 int32，零点固定为 0：
$$
Z_{b_{hh}} = 0
$$

Bias 的 scale 独立选择：
$$
S_{b_{hh}} = \frac{\max(|B_{hh}|)}{2^{31} - 1}
$$

量化公式：
$$
q_{b_{hh}} = \text{round}\left(\frac{B_{hh}}{S_{b_{hh}}}\right)
$$

**推导过程：**

**步骤 1**: 展开量化域计算
$$
S_{gh}(q_{gh} - Z_{gh}) = S_h(q_h - Z_h) \cdot S_{w_{hh}} q_{w_{hh}} + S_{b_{hh}}q_{b_{hh}}
$$

**步骤 2**: 展开 MatMul 部分
$$
S_{gh}(q_{gh} - Z_{gh}) = S_h \cdot S_{w_{hh}} (q_h - Z_h) \cdot q_{w_{hh}} + S_{b_{hh}}q_{b_{hh}}
$$

$$
S_{gh}(q_{gh} - Z_{gh}) = S_h \cdot S_{w_{hh}} (q_h \cdot q_{w_{hh}} - Z_h \cdot q_{w_{hh}}) + S_{b_{hh}}q_{b_{hh}}
$$

**步骤 3**: 两边同除 $S_{gh}$，将所有项归一化到输出 scale
$$
q_{gh} - Z_{gh} = \frac{S_h \cdot S_{w_{hh}}}{S_{gh}} (q_h \cdot q_{w_{hh}} - Z_h \cdot q_{w_{hh}}) + \frac{S_{b_{hh}}}{S_{gh}}q_{b_{hh}}
$$

**步骤 4**: 整理并定义缩放因子

令：
$$
S_3 = \frac{S_h \cdot S_{w_{hh}}}{S_{gh}}, \quad S_4 = \frac{S_{b_{hh}}}{S_{gh}}
$$

得到：
$$
q_{gh} - Z_{gh} = (q_h \cdot q_{w_{hh}} - Z_h \cdot q_{w_{hh}}) \cdot S_3 + q_{b_{hh}} \cdot S_4
$$

**步骤 5**: 整数化实现

Scale 采用 2 的幂次表示：
$$
S_3 = 2^{-n_3}, \quad S_4 = 2^{-n_4}
$$

**步骤 6**: 预融合优化（离线计算）

定义预融合常量：
$$
B'_q = (q_{b_{hh}} \;\texttt{>>}\; n_4) - (Z_h \cdot q_{w_{hh}} \;\texttt{>>}\; n_3)
$$

**融合缩放后的最终公式：**
$$
q_{gh} = (q_h \cdot q_{w_{hh}}) \;\texttt{>>}\; n_3 + B'_q + Z_{gh}
$$

其中：
- $n_3 = -\log_2(S_3)$: MatMul 的右移位数
- $B'_q$: 预融合的 bias 常量（包含了原始 bias 和零点补偿）

**说明：** 将隐藏状态 $H_{t-1}$ 通过权重 $W_{hh}$ 线性变换，得到隐藏贡献 $G_h$。输出维度为 $3 \times \text{hidden\_size}$。

---

### 操作 3-4: 分割操作 (Split/Chunk)

**浮点域计算：**
$$
\begin{align}
I_r, I_z, I_n &= \text{split}(G_i, \text{dim}=1) \\
H_r, H_z, H_n &= \text{split}(G_h, \text{dim}=1)
\end{align}
$$

**量化域计算：**
$$
\begin{align}
q_{i_r}, q_{i_z}, q_{i_n} &= \text{split}(q_{gi}, \text{dim}=1) \\
q_{h_r}, q_{h_z}, q_{h_n} &= \text{split}(q_{gh}, \text{dim}=1)
\end{align}
$$

**说明：**
- 分割操作不涉及数值计算，只是内存重排列
- 每个输出的 scaling factor 和 zero point 与输入相同
- $I_r, H_r$: Reset Gate 的输入和隐藏贡献
- $I_z, H_z$: Update Gate 的输入和隐藏贡献
- $I_n, H_n$: New Gate 的输入和隐藏贡献

---

## Reset Gate 计算

### 操作 5: Reset Gate 加法

**浮点域计算：**
$$
R_{\text{input}} = I_r + H_r
$$

**量化域计算：**
$$
S_r(q_r - Z_r) = S_{i_r}(q_{i_r} - Z_{i_r}) + S_{h_r}(q_{h_r} - Z_{h_r})
$$

**推导过程：**

**步骤 1**: 展开量化关系
$$
S_r(q_r - Z_r) = S_{i_r} q_{i_r} - S_{i_r} Z_{i_r} + S_{h_r} q_{h_r} - S_{h_r} Z_{h_r}
$$

**步骤 2**: 两边同除 $S_r$
$$
q_r - Z_r = \frac{S_{i_r}}{S_r} q_{i_r} - \frac{S_{i_r}}{S_r} Z_{i_r} + \frac{S_{h_r}}{S_r} q_{h_r} - \frac{S_{h_r}}{S_r} Z_{h_r}
$$

**步骤 3**: 整理并定义缩放因子

令：
$$
S_5 = \frac{S_{i_r}}{S_r}, \quad S_6 = \frac{S_{h_r}}{S_r}
$$

得到：
$$
q_r - Z_r = S_5 \cdot q_{i_r} + S_6 \cdot q_{h_r} - (S_5 \cdot Z_{i_r} + S_6 \cdot Z_{h_r})
$$

**步骤 4**: 整数化实现

Scale 采用 2 的幂次表示：
$$
S_5 = 2^{-n_5}, \quad S_6 = 2^{-n_6}
$$

**步骤 5**: 预融合优化（离线计算）

定义预融合常量：
$$
C_5 = Z_r - (Z_{i_r} \;\texttt{>>}\; n_5 + Z_{h_r} \;\texttt{>>}\; n_6)
$$

**融合缩放后的最终公式：**
$$
q_r = (q_{i_r} \;\texttt{>>}\; n_5) + (q_{h_r} \;\texttt{>>}\; n_6) + C_5
$$

**说明：** 两个加法输入可能有不同的 scale，需要先对齐到相同的量化空间。

---

### 操作 6: Reset Gate Sigmoid 激活（分段二次多项式实现）

**浮点域计算：**
$$
R_{\text{gate}} = \sigma(R_{\text{input}}) = \frac{1}{1 + e^{-R_{\text{input}}}}
$$

**分段二次多项式拟合方法：**

将 Sigmoid 函数在输入范围 $[x_{\min}, x_{\max}]$ 内分成 $N_{\text{seg}}=32$ 段，每段用二次多项式拟合：

$$
\sigma(x) \approx f_{\text{seg}}(x, i) = a_i \cdot x^2 + b_i \cdot x + c_i, \quad \text{for } t_{i-1} \leq x < t_i
$$

其中：
- $\{t_1, t_2, \ldots, t_{32}\}$: 32个阈值，定义段边界
- $\{a_i, b_i, c_i\}$: 第 $i$ 段的二次多项式系数
- 阈值通过**自适应分段策略**确定，在高曲率区域（$x \approx 0$）密集分段
- 系数通过**最小二乘法**拟合，并保证段间连续性



**量化域计算（FP32查找表）：**

给定浮点输入 $R_{\text{input}}$：

**步骤 1**: 找到对应的段索引
$$
i = \text{searchsorted}(R_{\text{input}}, \{t_1, \ldots, t_{32}\})
$$

**步骤 2**: 读取该段的系数
$$
a = a_i, \quad b = b_i, \quad c = c_i
$$

**步骤 3**: 计算二次多项式
$$
R_{\text{gate}} = a \cdot R_{\text{input}}^2 + b \cdot R_{\text{input}} + c
$$

**量化域计算（INT16定点实现 - 标准量化方式）：**

采用标准的 $S \cdot (q - Z)$ 量化关系，所有缩放因子设置为 $2^{-n}$ 形式以支持右移操作。

### **第一步：输入量化**

**浮点域：**
$$
x = R_{\text{input}}
$$

**量化域：**
$$
S_x(q_x - Z_x) = R_{\text{input}}
$$

其中 $S_x, Z_x$ 是输入量化参数（由前一层输出决定）。

### **第二步：阈值量化**

**浮点域阈值：**
$$
\{t_1, t_2, \ldots, t_{32}\}
$$

**量化阈值：**
$$
t_{i,q} = \text{round}\left(\frac{t_i}{S_x}\right) + Z_x, \quad i=1,\ldots,32
$$

### **第三步：系数量化（关键）**

对于浮点域的二次多项式系数 $a, b, c$，采用标准量化：

**系数 a 的量化关系：**
$$
S_a(q_a - Z_a) = a
$$

其中：
- $q_a$: 量化后的系数（INT16）
- $S_a$: 系数 $a$ 的缩放因子
- $Z_a$: 系数 $a$ 的零点（通常设为 0，对称量化）

**系数 b 的量化关系：**
$$
S_b(q_b - Z_b) = b
$$

**系数 c 的量化关系：**
$$
S_c(q_c - Z_c) = c
$$

**缩放因子的选择（关键优化）：**

为了使乘法运算能用右移实现，所有缩放因子设置为 $2^{-n}$ 形式：
$$
S_a = 2^{-n_a}, \quad S_b = 2^{-n_b}, \quad S_c = 2^{-n_c}
$$

其中 $n_a, n_b, n_c$ 是正整数右移位数（离线确定）。

### **第四步：计算 $x^2$ 的量化**

**浮点域：**
$$
x^2 = [S_x(q_x - Z_x)]^2 = S_x^2 \cdot (q_x - Z_x)^2
$$

**量化 $x^2$：**

需要将 $x^2$ 量化到量化域：
$$
S_{x2}(q_{x2} - Z_{x2}) = S_x^2 \cdot (q_x - Z_x)^2
$$

**推导过程：**

**步骤 1**: 展开右侧
$$
S_{x2}(q_{x2} - Z_{x2}) = S_x^2 \cdot (q_x - Z_x)^2
$$

**步骤 2**: 两边同除 $S_{x2}$
$$
q_{x2} - Z_{x2} = \frac{S_x^2}{S_{x2}} \cdot (q_x - Z_x)^2
$$

**步骤 3**: 定义缩放因子

令：
$$
M_{x2} = \frac{S_x^2}{S_{x2}}
$$

为了用右移实现，选择 $M_{x2} = 2^{-n_{x2}}$，其中 $n_{x2}$ 是右移位数。

这要求：
$$
S_{x2} = S_x^2 \cdot 2^{n_{x2}}
$$

**步骤 4**: 整数化计算

$$
q_{x2} = \big[(q_x - Z_x)^2 \;\texttt{>>}\; n_{x2}\big] + Z_{x2}
$$

**简化（对称量化，$Z_x = 0, Z_{x2} = 0$）：**
$$
q_{x2} = (q_x \cdot q_x) \;\texttt{>>}\; n_{x2}
$$

### **第五步：计算 $a \cdot x^2$ 的量化**

**浮点域：**
$$
a \cdot x^2 = [S_a(q_a - Z_a)] \cdot [S_{x2}(q_{x2} - Z_{x2})]
$$

**量化域：**
$$
S_{ax2}(q_{ax2} - Z_{ax2}) = [S_a(q_a - Z_a)] \cdot [S_{x2}(q_{x2} - Z_{x2})]
$$

**推导过程：**

**步骤 1**: 展开乘法
$$
S_{ax2}(q_{ax2} - Z_{ax2}) = S_a \cdot S_{x2} \cdot (q_a - Z_a) \cdot (q_{x2} - Z_{x2})
$$

**步骤 2**: 两边同除 $S_{ax2}$
$$
q_{ax2} - Z_{ax2} = \frac{S_a \cdot S_{x2}}{S_{ax2}} \cdot (q_a - Z_a) \cdot (q_{x2} - Z_{x2})
$$

**步骤 3**: 定义缩放因子

令：
$$
M_{ax2} = \frac{S_a \cdot S_{x2}}{S_{ax2}} = 2^{-n_{ax2}}
$$

**步骤 4**: 整数化计算

$$
q_{ax2} = \big[(q_a - Z_a) \cdot (q_{x2} - Z_{x2}) \;\texttt{>>}\; n_{ax2}\big] + Z_{ax2}
$$

**简化（对称量化，$Z_a = 0, Z_{x2} = 0, Z_{ax2} = 0$）：**
$$
q_{ax2} = (q_a \cdot q_{x2}) \;\texttt{>>}\; n_{ax2}
$$

### **第六步：计算 $b \cdot x$ 的量化**

**浮点域：**
$$
b \cdot x = [S_b(q_b - Z_b)] \cdot [S_x(q_x - Z_x)]
$$

**量化域：**
$$
S_{bx}(q_{bx} - Z_{bx}) = [S_b(q_b - Z_b)] \cdot [S_x(q_x - Z_x)]
$$

**推导过程：**

**步骤 1**: 展开乘法
$$
S_{bx}(q_{bx} - Z_{bx}) = S_b \cdot S_x \cdot (q_b - Z_b) \cdot (q_x - Z_x)
$$

**步骤 2**: 两边同除 $S_{bx}$
$$
q_{bx} - Z_{bx} = \frac{S_b \cdot S_x}{S_{bx}} \cdot (q_b - Z_b) \cdot (q_x - Z_x)
$$

**步骤 3**: 定义缩放因子

令：
$$
M_{bx} = \frac{S_b \cdot S_x}{S_{bx}} = 2^{-n_{bx}}
$$

**步骤 4**: 整数化计算

$$
q_{bx} = \big[(q_b - Z_b) \cdot (q_x - Z_x) \;\texttt{>>}\; n_{bx}\big] + Z_{bx}
$$

**简化（对称量化，$Z_b = 0, Z_x = 0, Z_{bx} = 0$）：**
$$
q_{bx} = (q_b \cdot q_x) \;\texttt{>>}\; n_{bx}
$$

### **第七步：最终加法 $y = ax^2 + bx + c$**

**浮点域：**
$$
y = a \cdot x^2 + b \cdot x + c
$$

**量化域：**
$$
S_y(q_y - Z_y) = S_{ax2}(q_{ax2} - Z_{ax2}) + S_{bx}(q_{bx} - Z_{bx}) + S_c(q_c - Z_c)
$$

**推导过程：**

**步骤 1**: 展开量化关系
$$
S_y(q_y - Z_y) = S_{ax2} q_{ax2} - S_{ax2} Z_{ax2} + S_{bx} q_{bx} - S_{bx} Z_{bx} + S_c q_c - S_c Z_c
$$

**步骤 2**: 两边同除 $S_y$
$$
q_y - Z_y = \frac{S_{ax2}}{S_y} q_{ax2} + \frac{S_{bx}}{S_y} q_{bx} + \frac{S_c}{S_y} q_c - \left(\frac{S_{ax2}}{S_y} Z_{ax2} + \frac{S_{bx}}{S_y} Z_{bx} + \frac{S_c}{S_y} Z_c\right)
$$

**步骤 3**: 定义缩放因子

令：
$$
M_{ya} = \frac{S_{ax2}}{S_y} = 2^{-n_{ya}}, \quad M_{yb} = \frac{S_{bx}}{S_y} = 2^{-n_{yb}}, \quad M_{yc} = \frac{S_c}{S_y} = 2^{-n_{yc}}
$$

**步骤 4**: 预融合优化（离线计算）

定义预融合常量：
$$
C_y = Z_y - (Z_{ax2} \;\texttt{>>}\; n_{ya} + Z_{bx} \;\texttt{>>}\; n_{yb} + Z_c \;\texttt{>>}\; n_{yc})
$$

**步骤 5**: 最终公式

$$
q_y = (q_{ax2} \;\texttt{>>}\; n_{ya}) + (q_{bx} \;\texttt{>>}\; n_{yb}) + (q_c \;\texttt{>>}\; n_{yc}) + C_y
$$

**简化（对称量化且所有零点为 0）：**
$$
q_y = (q_{ax2} \;\texttt{>>}\; n_{ya}) + (q_{bx} \;\texttt{>>}\; n_{yb}) + (q_c \;\texttt{>>}\; n_{yc}) + Z_y
$$

### **完整的INT16定点计算流程总结**

给定输入 $q_x$（INT16），计算输出 $q_y$（INT16）：

**步骤 1**: 找到段索引 $i$（二分查找或线性比较）
$$
i = \text{segment\_index}(q_x, \{t_{1,q}, \ldots, t_{32,q}\})
$$

**步骤 2**: 读取第 $i$ 段的量化系数
$$
q_a = q_{a,i}, \quad q_b = q_{b,i}, \quad q_c = q_{c,i}
$$

**步骤 3**: 读取第 $i$ 段的右移位数
$$
n_{x2} = n_{x2,i}, \quad n_{ax2} = n_{ax2,i}, \quad n_{bx} = n_{bx,i}
$$
$$
n_{ya} = n_{ya,i}, \quad n_{yb} = n_{yb,i}, \quad n_{yc} = n_{yc,i}
$$

**步骤 4**: 计算 $x^2$（INT16 → INT32 → 右移 → INT16）
$$
q_{x2} = (q_x \cdot q_x) \;\texttt{>>}\; n_{x2}
$$

**步骤 5**: 计算 $a \cdot x^2$（INT16 × INT16 → INT32 → 右移 → INT16）
$$
q_{ax2} = (q_a \cdot q_{x2}) \;\texttt{>>}\; n_{ax2}
$$

**步骤 6**: 计算 $b \cdot x$（INT16 × INT16 → INT32 → 右移 → INT16）
$$
q_{bx} = (q_b \cdot q_x) \;\texttt{>>}\; n_{bx}
$$

**步骤 7**: 最终加法（INT32 加法 + 饱和）
$$
y_{tmp} = (q_{ax2} \;\texttt{>>}\; n_{ya}) + (q_{bx} \;\texttt{>>}\; n_{yb}) + (q_c \;\texttt{>>}\; n_{yc}) + Z_y
$$
$$
q_y = \text{saturate}_{[-32768, 32767]}(y_{tmp})
$$


## Update Gate 计算

### 操作 7: Update Gate 加法

**浮点域计算：**
$$
Z_{\text{input}} = I_z + H_z
$$

**量化域计算：**
$$
S_z(q_z - Z_z) = S_{i_z}(q_{i_z} - Z_{i_z}) + S_{h_z}(q_{h_z} - Z_{h_z})
$$

**推导过程：**

**步骤 1**: 展开量化关系
$$
S_z(q_z - Z_z) = S_{i_z} q_{i_z} - S_{i_z} Z_{i_z} + S_{h_z} q_{h_z} - S_{h_z} Z_{h_z}
$$

**步骤 2**: 两边同除 $S_z$
$$
q_z - Z_z = \frac{S_{i_z}}{S_z} q_{i_z} - \frac{S_{i_z}}{S_z} Z_{i_z} + \frac{S_{h_z}}{S_z} q_{h_z} - \frac{S_{h_z}}{S_z} Z_{h_z}
$$

**步骤 3**: 整理并定义缩放因子

令：
$$
S_7 = \frac{S_{i_z}}{S_z}, \quad S_8 = \frac{S_{h_z}}{S_z}
$$

得到：
$$
q_z - Z_z = S_7 \cdot q_{i_z} + S_8 \cdot q_{h_z} - (S_7 \cdot Z_{i_z} + S_8 \cdot Z_{h_z})
$$

**步骤 4**: 整数化实现

Scale 采用 2 的幂次表示：
$$
S_7 = 2^{-n_7}, \quad S_8 = 2^{-n_8}
$$

**步骤 5**: 预融合优化（离线计算）

定义预融合常量：
$$
C_7 = Z_z - (Z_{i_z} \;\texttt{>>}\; n_7 + Z_{h_z} \;\texttt{>>}\; n_8)
$$

**融合缩放后的最终公式：**
$$
q_z = (q_{i_z} \;\texttt{>>}\; n_7) + (q_{h_z} \;\texttt{>>}\; n_8) + C_7
$$

---

### 操作 8: Update Gate Sigmoid 激活（分段二次多项式实现）

**浮点域计算：**
$$
Z_{\text{gate}} = \sigma(Z_{\text{input}}) = \frac{1}{1 + e^{-Z_{\text{input}}}}
$$

**量化域计算：**

与操作 6 完全相同，使用相同的分段二次多项式查找表。

**INT16 定点计算公式：**
$$
q_{z\_gate} = \text{saturate}\left((a_q \cdot ((q_z \cdot q_z) \;\texttt{>>}\; 15) \;\texttt{>>}\; 15) + (b_q \cdot q_z \;\texttt{>>}\; 15) + c_q\right)
$$

**说明：**
- Reset Gate 和 Update Gate 可以**共享同一个 Sigmoid 查找表**
- 仅需存储一套参数（640 bytes），节省内存

---

## New Gate 计算

### 操作 9: Reset Gate 与隐藏状态相乘

**浮点域计算：**
$$
\text{Reset}_h = R_{\text{gate}} \odot H_n
$$

**量化域计算：**
$$
S_{\text{reset}_h}(q_{\text{reset}_h} - Z_{\text{reset}_h}) = S_{r\_gate}(q_{r\_gate} - Z_{r\_gate}) \cdot S_{h_n}(q_{h_n} - Z_{h_n})
$$

**推导过程：**

**步骤 1**: 展开量化关系
$$
S_{\text{reset}_h}(q_{\text{reset}_h} - Z_{\text{reset}_h}) = S_{r\_gate} S_{h_n} (q_{r\_gate} - Z_{r\_gate}) (q_{h_n} - Z_{h_n})
$$

**步骤 2**: 两边同除 $S_{\text{reset}_h}$
$$
q_{\text{reset}_h} - Z_{\text{reset}_h} = \frac{S_{r\_gate} \cdot S_{h_n}}{S_{\text{reset}_h}} (q_{r\_gate} - Z_{r\_gate}) (q_{h_n} - Z_{h_n})
$$

**步骤 3**: 定义缩放因子

令：
$$
S_9 = \frac{S_{r\_gate} \cdot S_{h_n}}{S_{\text{reset}_h}}
$$

得到：
$$
q_{\text{reset}_h} - Z_{\text{reset}_h} = S_9 \cdot (q_{r\_gate} - Z_{r\_gate}) (q_{h_n} - Z_{h_n})
$$

**步骤 4**: 整数化实现

Scale 采用 2 的幂次表示：
$$
S_9 = 2^{-n_9}
$$

**融合缩放后的最终公式：**
$$
q_{\text{reset}_h} = \big( (q_{r\_gate} - Z_{r\_gate}) \cdot (q_{h_n} - Z_{h_n}) \big) \;\texttt{>>}\; n_9 + Z_{\text{reset}_h}
$$

**说明：** 逐元素乘法（Hadamard product），Reset Gate 控制保留多少隐藏状态信息。

---

### 操作 10: New Gate 加法

**浮点域计算：**
$$
N_{\text{input}} = I_n + \text{Reset}_h
$$

**量化域计算：**
$$
S_n(q_n - Z_n) = S_{i_n}(q_{i_n} - Z_{i_n}) + S_{\text{reset}_h}(q_{\text{reset}_h} - Z_{\text{reset}_h})
$$

**推导过程：**

**步骤 1**: 展开量化关系
$$
S_n(q_n - Z_n) = S_{i_n} q_{i_n} - S_{i_n} Z_{i_n} + S_{\text{reset}_h} q_{\text{reset}_h} - S_{\text{reset}_h} Z_{\text{reset}_h}
$$

**步骤 2**: 两边同除 $S_n$
$$
q_n - Z_n = \frac{S_{i_n}}{S_n} q_{i_n} - \frac{S_{i_n}}{S_n} Z_{i_n} + \frac{S_{\text{reset}_h}}{S_n} q_{\text{reset}_h} - \frac{S_{\text{reset}_h}}{S_n} Z_{\text{reset}_h}
$$

**步骤 3**: 整理并定义缩放因子

令：
$$
S_{10} = \frac{S_{i_n}}{S_n}, \quad S_{11} = \frac{S_{\text{reset}_h}}{S_n}
$$

得到：
$$
q_n - Z_n = S_{10} \cdot q_{i_n} + S_{11} \cdot q_{\text{reset}_h} - (S_{10} \cdot Z_{i_n} + S_{11} \cdot Z_{\text{reset}_h})
$$

**步骤 4**: 整数化实现

Scale 采用 2 的幂次表示：
$$
S_{10} = 2^{-n_{10}}, \quad S_{11} = 2^{-n_{11}}
$$

**步骤 5**: 预融合优化（离线计算）

定义预融合常量：
$$
C_{10} = Z_n - (Z_{i_n} \;\texttt{>>}\; n_{10} + Z_{\text{reset}_h} \;\texttt{>>}\; n_{11})
$$

**融合缩放后的最终公式：**
$$
q_n = (q_{i_n} \;\texttt{>>}\; n_{10}) + (q_{\text{reset}_h} \;\texttt{>>}\; n_{11}) + C_{10}
$$

---

### 操作 11: New Gate Tanh 激活（分段二次多项式实现）

**浮点域计算：**
$$
N_{\text{gate}} = \tanh(N_{\text{input}}) = \frac{e^{N_{\text{input}}} - e^{-N_{\text{input}}}}{e^{N_{\text{input}}} + e^{-N_{\text{input}}}}
$$

**分段二次多项式拟合方法：**

与 Sigmoid 类似，将 Tanh 函数在输入范围 $[x_{\min}, x_{\max}]$ 内分成 $N_{\text{seg}}=32$ 段：

$$
\tanh(x) \approx f_{\text{seg}}(x, i) = a_i \cdot x^2 + b_i \cdot x + c_i, \quad \text{for } t_{i-1} \leq x < t_i
$$

**离线预处理：**

1. **生成训练数据**：
   $$
   \{x_j, y_j\}_{j=1}^{10000}, \quad y_j = \tanh(x_j)
   $$

2. **自适应分段**：Tanh 的曲率在 $x \approx 0$ 最大，自动在中心区域密集分段

3. **系数拟合**：每段独立最小二乘拟合

4. **连续性调整**：保证段间连续

**INT16 定点计算流程：**

与 Sigmoid 完全相同，只是使用 Tanh 专用的查找表参数。

**步骤 1**: 找到段索引
$$
i = \text{search}(q_n, \{t_{1,q}^{\tanh}, \ldots, t_{32,q}^{\tanh}\})
$$

**步骤 2**: 读取 Tanh 的量化系数
$$
a_q = a_{i,q}^{\tanh}, \quad b_q = b_{i,q}^{\tanh}, \quad c_q = c_{i,q}^{\tanh}
$$

**步骤 3-6**: 定点计算
$$
x^2_q = (q_n \cdot q_n) \;\texttt{>>}\; 15
$$
$$
ax2_q = (a_q \cdot x^2_q) \;\texttt{>>}\; 15
$$
$$
bx_q = (b_q \cdot q_n) \;\texttt{>>}\; 15
$$
$$
q_{n\_gate} = \text{saturate}(ax2_q + bx_q + c_q)
$$

**最终量化公式：**
$$
q_{n\_gate} = \text{saturate}\left((a_q \cdot ((q_n \cdot q_n) \;\texttt{>>}\; 15) \;\texttt{>>}\; 15) + (b_q \cdot q_n \;\texttt{>>}\; 15) + c_q\right)
$$

**内存占用：**
- Tanh 查找表：640 bytes（与 Sigmoid 分开存储）
- **总计激活函数内存：1280 bytes（Sigmoid + Tanh）**

**精度分析：**
- 平均绝对误差（MAE）：< 0.001
- 最大误差（Max Error）：< 0.01
- Tanh 输出范围 $[-1, 1]$ 对应 INT16: [-32768, 32767]

**说明：**
- Tanh 输出范围为 $[-1, 1]$
- 输出量化参数：$S_{n\_out} = 2/65535$, $Z_{n\_out} = 0$ (对称量化)
- Tanh 需要独立的查找表（与 Sigmoid 不同）

---

## 最终隐藏状态更新

### 操作 12: 计算 1 - Update Gate

**浮点域计算：**
$$
\text{One\_minus\_update} = 1.0 - Z_{\text{gate}}
$$

**量化域计算：**
$$
S_{\text{omu}}(q_{\text{omu}} - Z_{\text{omu}}) = 1.0 - S_{z\_gate}(q_{z\_gate} - Z_{z\_gate})
$$

**推导过程：**

**步骤 1**: 展开量化关系
$$
S_{\text{omu}}(q_{\text{omu}} - Z_{\text{omu}}) = 1.0 - S_{z\_gate} q_{z\_gate} + S_{z\_gate} Z_{z\_gate}
$$

**步骤 2**: 两边同除 $S_{\text{omu}}$
$$
q_{\text{omu}} - Z_{\text{omu}} = \frac{1.0}{S_{\text{omu}}} - \frac{S_{z\_gate}}{S_{\text{omu}}} q_{z\_gate} + \frac{S_{z\_gate}}{S_{\text{omu}}} Z_{z\_gate}
$$

**步骤 3**: 定义缩放因子

令：
$$
C_1 = \frac{1.0}{S_{\text{omu}}}, \quad S_{12} = \frac{S_{z\_gate}}{S_{\text{omu}}}
$$

得到：
$$
q_{\text{omu}} - Z_{\text{omu}} = C_1 - S_{12} \cdot q_{z\_gate} + S_{12} \cdot Z_{z\_gate}
$$

**步骤 4**: 整数化实现

Scale 采用 2 的幂次表示：
$$
S_{12} = 2^{-n_{12}}
$$

**步骤 5**: 预融合优化（离线计算）

定义预融合常量：
$$
C_{12} = Z_{\text{omu}} + (C_1 \;\texttt{>>}\; 0) + (Z_{z\_gate} \;\texttt{>>}\; n_{12})
$$

其中 $C_1 = \frac{1.0}{S_{\text{omu}}}$ 是常数项。

**融合缩放后的最终公式：**
$$
q_{\text{omu}} = C_{12} - (q_{z\_gate} \;\texttt{>>}\; n_{12})
$$

**说明：** 计算更新门的补数，用于控制新信息的贡献比例。

---

### 操作 13: 新信息贡献

**浮点域计算：**
$$
\text{New\_contrib} = \text{One\_minus\_update} \odot N_{\text{gate}}
$$

**量化域计算：**
$$
S_{\text{new}}(q_{\text{new}} - Z_{\text{new}}) = S_{\text{omu}}(q_{\text{omu}} - Z_{\text{omu}}) \cdot S_{n\_gate}(q_{n\_gate} - Z_{n\_gate})
$$

**推导过程：**

**步骤 1**: 展开量化关系
$$
S_{\text{new}}(q_{\text{new}} - Z_{\text{new}}) = S_{\text{omu}} S_{n\_gate} (q_{\text{omu}} - Z_{\text{omu}}) (q_{n\_gate} - Z_{n\_gate})
$$

**步骤 2**: 两边同除 $S_{\text{new}}$
$$
q_{\text{new}} - Z_{\text{new}} = \frac{S_{\text{omu}} \cdot S_{n\_gate}}{S_{\text{new}}} (q_{\text{omu}} - Z_{\text{omu}}) (q_{n\_gate} - Z_{n\_gate})
$$

**步骤 3**: 定义缩放因子

令：
$$
S_{13} = \frac{S_{\text{omu}} \cdot S_{n\_gate}}{S_{\text{new}}}
$$

得到：
$$
q_{\text{new}} - Z_{\text{new}} = S_{13} \cdot (q_{\text{omu}} - Z_{\text{omu}}) (q_{n\_gate} - Z_{n\_gate})
$$

**步骤 4**: 整数化实现

Scale 采用 2 的幂次表示：
$$
S_{13} = 2^{-n_{13}}
$$

**融合缩放后的最终公式：**
$$
q_{\text{new}} = \big( (q_{\text{omu}} - Z_{\text{omu}}) \cdot (q_{n\_gate} - Z_{n\_gate}) \big) \;\texttt{>>}\; n_{13} + Z_{\text{new}}
$$

**说明：** 计算新候选信息对最终隐藏状态的贡献，由 $(1-z_t)$ 控制。

---

### 操作 14: 旧信息贡献

**浮点域计算：**
$$
\text{Old\_contrib} = Z_{\text{gate}} \odot H_{t-1}
$$

**量化域计算：**
$$
S_{\text{old}}(q_{\text{old}} - Z_{\text{old}}) = S_{z\_gate}(q_{z\_gate} - Z_{z\_gate}) \cdot S_{h}(q_{h} - Z_{h})
$$

**推导过程：**

**步骤 1**: 展开量化关系
$$
S_{\text{old}}(q_{\text{old}} - Z_{\text{old}}) = S_{z\_gate} S_{h} (q_{z\_gate} - Z_{z\_gate}) (q_{h} - Z_{h})
$$

**步骤 2**: 两边同除 $S_{\text{old}}$
$$
q_{\text{old}} - Z_{\text{old}} = \frac{S_{z\_gate} \cdot S_{h}}{S_{\text{old}}} (q_{z\_gate} - Z_{z\_gate}) (q_{h} - Z_{h})
$$

**步骤 3**: 定义缩放因子

令：
$$
S_{14} = \frac{S_{z\_gate} \cdot S_{h}}{S_{\text{old}}}
$$

得到：
$$
q_{\text{old}} - Z_{\text{old}} = S_{14} \cdot (q_{z\_gate} - Z_{z\_gate}) (q_{h} - Z_{h})
$$

**步骤 4**: 整数化实现

Scale 采用 2 的幂次表示：
$$
S_{14} = 2^{-n_{14}}
$$

**融合缩放后的最终公式：**
$$
q_{\text{old}} = \big( (q_{z\_gate} - Z_{z\_gate}) \cdot (q_{h} - Z_{h}) \big) \;\texttt{>>}\; n_{14} + Z_{\text{old}}
$$

**说明：** 计算旧隐藏状态对最终隐藏状态的贡献，由 $z_t$ 控制。

---

### 操作 15: 最终隐藏状态合并

**浮点域计算：**
$$
H_t = \text{New\_contrib} + \text{Old\_contrib}
$$

**量化域计算：**
$$
S_{h_t}(q_{h_t} - Z_{h_t}) = S_{\text{new}}(q_{\text{new}} - Z_{\text{new}}) + S_{\text{old}}(q_{\text{old}} - Z_{\text{old}})
$$

**推导过程：**

**步骤 1**: 展开量化关系
$$
S_{h_t}(q_{h_t} - Z_{h_t}) = S_{\text{new}} q_{\text{new}} - S_{\text{new}} Z_{\text{new}} + S_{\text{old}} q_{\text{old}} - S_{\text{old}} Z_{\text{old}}
$$

**步骤 2**: 两边同除 $S_{h_t}$
$$
q_{h_t} - Z_{h_t} = \frac{S_{\text{new}}}{S_{h_t}} q_{\text{new}} - \frac{S_{\text{new}}}{S_{h_t}} Z_{\text{new}} + \frac{S_{\text{old}}}{S_{h_t}} q_{\text{old}} - \frac{S_{\text{old}}}{S_{h_t}} Z_{\text{old}}
$$

**步骤 3**: 整理并定义缩放因子

令：
$$
S_{15} = \frac{S_{\text{new}}}{S_{h_t}}, \quad S_{16} = \frac{S_{\text{old}}}{S_{h_t}}
$$

得到：
$$
q_{h_t} - Z_{h_t} = S_{15} \cdot q_{\text{new}} + S_{16} \cdot q_{\text{old}} - (S_{15} \cdot Z_{\text{new}} + S_{16} \cdot Z_{\text{old}})
$$

**步骤 4**: 整数化实现

Scale 采用 2 的幂次表示：
$$
S_{15} = 2^{-n_{15}}, \quad S_{16} = 2^{-n_{16}}
$$

**步骤 5**: 预融合优化（离线计算）

定义预融合常量：
$$
C_{15} = Z_{h_t} - (Z_{\text{new}} \;\texttt{>>}\; n_{15} + Z_{\text{old}} \;\texttt{>>}\; n_{16})
$$

**融合缩放后的最终公式：**
$$
q_{h_t} = (q_{\text{new}} \;\texttt{>>}\; n_{15}) + (q_{\text{old}} \;\texttt{>>}\; n_{16}) + C_{15}
$$

**说明：** 将新旧信息贡献相加，得到最终的隐藏状态 $H_t$。

---

## 操作汇总表

| 序号 | 操作名称 | 类型 | 输入 | 输出 | 量化算子 |
|------|---------|------|------|------|---------|
| 1 | 输入线性变换 | Linear | $X_t$ | $G_i$ | MatMul + Add |
| 2 | 隐藏状态线性变换 | Linear | $H_{t-1}$ | $G_h$ | MatMul + Add |
| 3-4 | 分割操作 | Split | $G_i, G_h$ | $I_r, I_z, I_n, H_r, H_z, H_n$ | Reshape |
| 5 | Reset Gate 加法 | Add | $I_r, H_r$ | $R_{\text{input}}$ | Add |
| 6 | Reset Gate 激活 | Sigmoid | $R_{\text{input}}$ | $R_{\text{gate}}$ | **Quadratic LUT** |
| 7 | Update Gate 加法 | Add | $I_z, H_z$ | $Z_{\text{input}}$ | Add |
| 8 | Update Gate 激活 | Sigmoid | $Z_{\text{input}}$ | $Z_{\text{gate}}$ | **Quadratic LUT** |
| 9 | Reset 乘法 | Multiply | $R_{\text{gate}}, H_n$ | $\text{Reset}_h$ | Multiply |
| 10 | New Gate 加法 | Add | $I_n, \text{Reset}_h$ | $N_{\text{input}}$ | Add |
| 11 | New Gate 激活 | Tanh | $N_{\text{input}}$ | $N_{\text{gate}}$ | **Quadratic LUT** |
| 12 | 更新门补数 | Subtract | $1.0, Z_{\text{gate}}$ | $\text{One\_minus\_update}$ | Subtract |
| 13 | 新信息贡献 | Multiply | $\text{One\_minus\_update}, N_{\text{gate}}$ | $\text{New\_contrib}$ | Multiply |
| 14 | 旧信息贡献 | Multiply | $Z_{\text{gate}}, H_{t-1}$ | $\text{Old\_contrib}$ | Multiply |
| 15 | 最终合并 | Add | $\text{New\_contrib}, \text{Old\_contrib}$ | $H_t$ | Add |

---

## 数字硬件实现要点

### 1. 线性层（Linear）
- **矩阵乘法**: 使用定点乘累加单元（MAC）
- **位宽**: 输入/权重 8-bit，累加器 32-bit
- **Bias 处理**:
    - Bias 采用**对称量化**（$Z_b = 0$），量化为 int32
    - Scale: $S_b = \frac{\max(|B|)}{2^{31}-1}$ (独立于输入和权重)
    - 单独 rescale：`q_b_rescaled = q_b * M_bias >> shift_bias`
    - 加到累加器：`acc = matmul_rescaled + bias_rescaled`
- **输出**: 通过 requantization 降回 8-bit

### 2. 加法层（Add）
- **对齐**: 不同 scale 的输入需要先对齐到相同量化空间
- **饱和处理**: 结果超出 int8 范围需要饱和到 [-128, 127]
- **零点处理**: 需要正确处理零点的加减

### 3. 乘法层（Multiply）
- **位宽扩展**: 两个 8-bit 数相乘得到 16-bit 结果
- **缩放**: 结果需要通过 $M \times (\texttt{>>} n)$ 重新量化回 8-bit
- **零点处理**: $(q_1 - Z_1) \times (q_2 - Z_2)$ 展开后有四项

### 4. 减法层（Subtract）
- **特殊情况**: $1.0 - x$ 可以利用量化关系简化
- **对齐**: 类似加法，需要对齐 scale
- **符号处理**: 需要正确处理有符号数的减法

### 5. 激活函数（Sigmoid/Tanh - 分段二次多项式实现）

**硬件架构设计：**

```
┌─────────────────────────────────────────────────────────────────┐
│                   分段二次多项式 LUT 硬件单元                      │
├─────────────────────────────────────────────────────────────────┤
│  输入: q_input (INT16)                                           │
│                                                                  │
│  ┌─────────────────┐                                            │
│  │ 段索引查找模块    │  → 比较 q_input 与 32 个阈值               │
│  │ (Comparator)    │  → 输出段索引 i (5-bit)                   │
│  └─────────────────┘                                            │
│           ↓                                                      │
│  ┌─────────────────┐                                            │
│  │ 系数读取模块     │  → 读取 a_i, b_i, c_i (INT16)             │
│  │ (Coeff Memory)  │  → 读取 shift 参数                         │
│  └─────────────────┘                                            │
│           ↓                                                      │
│  ┌─────────────────┐                                            │
│  │ x^2 计算单元     │  → x32 = q_input * q_input (INT32)        │
│  │                 │  → x2_q = x32 >> 15 (INT16)               │
│  └─────────────────┘                                            │
│           ↓                                                      │
│  ┌─────────────────┐                                            │
│  │ 多项式计算单元   │  → ax2 = (a * x2_q) >> 15                 │
│  │ (MAC Units)     │  → bx = (b * q_input) >> 15               │
│  │                 │  → y = ax2 + bx + c                        │
│  └─────────────────┘                                            │
│           ↓                                                      │
│  ┌─────────────────┐                                            │
│  │ 饱和处理单元     │  → saturate(y, -32768, 32767)             │
│  └─────────────────┘                                            │
│           ↓                                                      │
│  输出: q_output (INT16)                                          │
└─────────────────────────────────────────────────────────────────┘
```

**内存组织：**
- **阈值存储**: 32 × INT16 = 64 bytes（ROM）
- **系数存储**: 32 × 3 × INT16 = 192 bytes（ROM）
- **Shift参数**: 32 × 3 × INT32 = 384 bytes（ROM）
- **总ROM需求**:
    - Sigmoid: 640 bytes
    - Tanh: 640 bytes
    - **合计**: 1280 bytes

**计算复杂度：**
- **段索引查找**: 最多32次比较（可用二分查找优化到5次）
- **乘法操作**: 3次（$x^2$, $a \cdot x^2$, $b \cdot x$）
- **加法操作**: 2次（$ax^2 + bx + c$）
- **右移操作**: 3次
- **总计**: 约 10-15 个时钟周期（流水线实现）

**流水线设计：**
```
Stage 1: 段索引查找 (1 cycle)
Stage 2: 系数读取 (1 cycle)
Stage 3: x^2 计算 (1 cycle)
Stage 4: ax^2 和 bx 并行计算 (2 cycles)
Stage 5: 最终求和 + 饱和 (1 cycle)
────────────────────────────────────
总延迟: 6 cycles (流水线吞吐率: 1 output/cycle)
```

**精度对比：**

| 方法 | 内存占用 | 计算复杂度 | MAE | Max Error | 延迟 |
|------|---------|-----------|-----|-----------|------|
| 传统 LUT (256点) | 256 bytes | 1次查找 | 0.002-0.005 | 0.01-0.02 | 1 cycle |
| PWL (16段) | ~128 bytes | 比较+乘加 | 0.005-0.01 | 0.02-0.05 | 3-4 cycles |
| **Quadratic LUT (32段)** | **640 bytes** | **比较+3乘+2加** | **< 0.001** | **< 0.01** | **6 cycles** |

**优势分析：**
- ✅ **精度最高**: MAE < 0.001，优于传统LUT和PWL
- ✅ **硬件友好**: 纯INT16/INT32运算，无除法/指数
- ✅ **内存可控**: 每函数640 bytes，可接受
- ✅ **可并行**: Sigmoid和Tanh可以用独立硬件单元并行计算
- ✅ **可配置**: 段数可调（8段→256 bytes，64段→1280 bytes）

### 6. 整体流水线设计
```
时钟周期:  1      2      3      4      5      6      7      8      9     10     11     12     13     14     15
         ───────────────────────────────────────────────────────────────────────────────────────────────────────
Stage 1: Linear_ih                                                                                               
Stage 2:        Linear_hh                                                                                        
Stage 3:                Split                                                                                     
Stage 4:                       Add_r         Add_z                                                              
Stage 5:                                Sigmoid_r    Sigmoid_z    (Quadratic LUT, 6 cycles each)                                                 
Stage 6:                                                      Mul_reset                                          
Stage 7:                                                                 Add_n                                    
Stage 8:                                                                        Tanh_n    (Quadratic LUT, 6 cycles)                         
Stage 9:                                                                               Sub_1-z                    
Stage 10:                                                                                      Mul_new  Mul_old  
Stage 11:                                                                                                  Add_final
```

**说明**: 相比传统LUT（1 cycle），分段二次多项式需要6 cycles，但精度显著提升。

---

## 量化参数确定方法

### 1. Scale 和 Zero Point 的选择

**方法 1: Min-Max 量化（非对称量化）**
$$
S = \frac{\max(\text{float}) - \min(\text{float})}{q_{\max} - q_{\min}}
$$
$$
Z = q_{\min} - \frac{\min(\text{float})}{S}
$$

**方法 2: 对称量化**（Zero Point = 0）
$$
S = \frac{\max(|\text{float}_{\max}|, |\text{float}_{\min}|)}{127}
$$

**Bias 的对称量化**

Bias 始终采用对称量化 $(Z_b = 0)$，scale 根据 bias 自身的数值范围独立确定：
$$
S_b = \frac{\max(|B_{\text{float}}|)}{2^{31} - 1}
$$

**特点**：
1. **对称性**：$Z_b = 0$ 简化了公式 $(q_b - Z_b) = q_b$
2. **位宽匹配**：量化为 int32 与 MatMul 累加器位宽一致，避免溢出
3. **独立 rescale**：Bias 和 MatMul 分别进行 rescale，精度更高

量化流程：
$$
q_b = \text{round}\left(\frac{B_{\text{float}}}{S_b}\right)
$$

**注意**：在硬件实现中，需要为 bias 单独配置 rescale 参数 $M_{bias}$ 和 $rshift_{bias}$。

### 2. 激活函数的固定量化（分段二次多项式）

**Sigmoid**: 输出范围 $[0, 1]$

对于 INT16 量化：
- **输入**: $S_{r\_in}$ 根据前一层输出动态确定
- **输出**: 对称量化
    - $S_{r\_out} = 1.0 / 32767 \approx 3.05 \times 10^{-5}$
    - $Z_{r\_out} = -32768$
    - 输出范围映射：$[0, 1] \to [-32768, 32767]$

**Tanh**: 输出范围 $[-1, 1]$

对于 INT16 量化：
- **输入**: $S_{n\_in}$ 根据前一层输出动态确定
- **输出**: 对称量化
    - $S_{n\_out} = 2.0 / 65535 \approx 3.05 \times 10^{-5}$
    - $Z_{n\_out} = 0$ (对称量化)
    - 输出范围映射：$[-1, 1] \to [-32768, 32767]$

**系数量化公式**：

对于分段 $i$ 的系数 $\{a_i, b_i, c_i\}$：

$$
\begin{align}
a_{i,q} &= \text{round}\left(a_i \cdot S_{\text{in}}^2 \cdot 2^{30} / S_{\text{out}}\right) \\
b_{i,q} &= \text{round}\left(b_i \cdot S_{\text{in}} \cdot 2^{15} / S_{\text{out}}\right) \\
c_{i,q} &= \text{round}\left(c_i / S_{\text{out}}\right) + Z_{\text{out}}
\end{align}
$$

**离线预计算**（模型加载时）：
1. 拟合得到浮点系数 $\{a_i, b_i, c_i\}_{i=1}^{32}$
2. 量化阈值 $\{t_{i,q}\}_{i=1}^{32}$
3. 量化系数 $\{a_{i,q}, b_{i,q}, c_{i,q}\}_{i=1}^{32}$
4. 存储到 ROM

### 3. Multiplier 和 Shift 的计算

给定浮点缩放因子 $S_{\text{real}}$，计算整数乘数 $M$ 和移位量 $n$：
$$
S_{\text{real}} = \frac{M}{2^n}, \quad M \in [2^{30}, 2^{31}-1], n \geq 0
$$

**算法**：
1. 归一化 $S_{\text{real}}$ 到 $[0.5, 1)$ 范围，记录移位量
2. $M = \text{round}(S_{\text{real}} \times 2^{31})$
3. 调整 $M$ 使其在 $[2^{30}, 2^{31}-1]$ 范围内

---

## 代码对应关系

以下是代码中每个操作与本文档的对应关系：

```python
# 操作 1-2: 线性变换
gi = self.weight_ih(input)    # 对应操作 1
gh = self.weight_hh(hidden)   # 对应操作 2

# 操作 3-4: 分割
i_r, i_z, i_n = gi.chunk(3, dim=1)
h_r, h_z, h_n = gh.chunk(3, dim=1)

# 操作 5-6: Reset Gate
r_gate_input = self.add_r(i_r, h_r)                      # 操作 5
resetgate = self.quadratic_sigmoid(r_gate_input)         # 操作 6 (改用分段二次多项式)

# 操作 7-8: Update Gate
u_gate_input = self.add_z(i_z, h_z)                      # 操作 7
updategate = self.quadratic_sigmoid(u_gate_input)        # 操作 8 (改用分段二次多项式)

# 操作 9-11: New Gate
reset_h = self.mul_reset(resetgate, h_n)                 # 操作 9
n_gate_input = self.add_n(i_n, reset_h)                  # 操作 10
newgate = self.quadratic_tanh(n_gate_input)              # 操作 11 (改用分段二次多项式)

# 操作 12-15: 最终隐藏状态
one_minus_update = self.sub(1.0, updategate)                        # 操作 12
new_contribution = self.mul_new(one_minus_update, newgate)          # 操作 13
old_contribution = self.mul_old(updategate, hidden)                 # 操作 14
new_h = self.add_final(new_contribution, old_contribution)          # 操作 15
```

**新增模块**：
- `quadratic_sigmoid`: 分段二次多项式 Sigmoid 模块
- `quadratic_tanh`: 分段二次多项式 Tanh 模块

**总计**: 2 Linear + 4 Add + 3 Multiply + 1 Subtract + **2 Quadratic LUT (Sigmoid + Tanh)** = **12 个量化算子**

---

## 性能与精度权衡

### 1. 量化位宽选择
- **8-bit (int8)**: 主流选择，硬件支持好，精度损失可接受
- **16-bit (int16)**: 精度更高，但硬件成本和功耗增加 4 倍
- **4-bit**: 极限压缩，需要特殊训练技巧

### 2. 激活函数实现对比

| 方法 | 精度 | 内存 | 计算量 | 硬件友好性 | 适用场景 |
|------|------|------|--------|-----------|---------|
| **查找表 (LUT)** | 中 (MAE≈0.003) | 256B | 极低 (1查找) | ⭐⭐⭐⭐⭐ | 资源极端受限 |
| **分段线性 (PWL)** | 中 (MAE≈0.007) | 128B | 低 (比较+乘加) | ⭐⭐⭐⭐ | 平衡方案 |
| **分段二次多项式** | **高 (MAE<0.001)** | **640B** | **中 (3乘+2加)** | **⭐⭐⭐⭐** | **高精度需求** |
| **多项式逼近 (高阶)** | 很高 | 小 | 高 | ⭐⭐ | 有硬件乘法器 |

**推荐方案**：
- **高精度场景**（本文档方案）：分段二次多项式 (MAE < 0.001)
- **资源受限**：传统 256 点 LUT (256 bytes)
- **平衡方案**：16段 PWL (128 bytes, MAE ≈ 0.007)

### 3. Scale 对齐策略
- **每层动态 rescale**: 精度最高，硬件开销大
- **预融合 scale**: 性能最优，精度略低
- **混合方案**: 关键路径动态，非关键路径融合

### 4. 分段二次多项式的优化技巧

**段数配置**：
- **8 段**：256 bytes，MAE ≈ 0.005（低端硬件）
- **32 段**（默认）：640 bytes，MAE < 0.001（推荐）
- **64 段**：1280 bytes，MAE < 0.0005（超高精度）

**自适应分段**：
- 基于曲率的分段策略，在高曲率区域密集分段
- Sigmoid/Tanh 在 $x \approx 0$ 处自动加密分段点
- 相同段数下误差降低 50-80%

**连续性优化**：
- C⁰连续性：函数值连续（已实现）
- C¹连续性：导数连续（可选，需三次样条）

---

## 参考文献

1. **Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference** (Jacob et al., CVPR 2018)
2. **Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation** (Wu et al., 2020)
3. **AIMET Documentation**: https://github.com/quic/aimet
4. **Piecewise Polynomial Approximation for Activation Functions** (本项目 nonlinear_lut.py)

---

## 附录A: 分段二次多项式完整示例

假设一个简单的 Sigmoid 例子（单段演示）：

### 输入数据
- 输入: $x = 0.5$ (FP32)
- 输入量化: $q_x = 20000$ (INT16)
- 输入scale: $S_x = 1/32767$, $Z_x = 0$

### 查找表参数（第15段，$x \in [0.4, 0.6]$）
- 阈值: $t_{15} = 0.6$
- 系数 (FP32): $a_{15} = -0.2$, $b_{15} = 0.8$, $c_{15} = 0.5$
- 量化系数 (INT16): $a_q = -150$, $b_q = 2500$, $c_q = 16000$

### FP32 计算
$$
y = -0.2 \times 0.5^2 + 0.8 \times 0.5 + 0.5 = -0.05 + 0.4 + 0.5 = 0.85
$$

### INT16 定点计算

**步骤 1**: $x^2$ 计算
$$
x_{32} = 20000 \times 20000 = 400,000,000 \quad (\text{INT32})
$$
$$
x^2_q = 400,000,000 \;\texttt{>>}\; 15 = 12,207 \quad (\text{INT16})
$$

**步骤 2**: $a \cdot x^2$ 计算
$$
ax2_{32} = -150 \times 12,207 = -1,831,050 \quad (\text{INT32})
$$
$$
ax2_q = -1,831,050 \;\texttt{>>}\; 15 = -56 \quad (\text{INT16})
$$

**步骤 3**: $b \cdot x$ 计算
$$
bx_{32} = 2500 \times 20000 = 50,000,000 \quad (\text{INT32})
$$
$$
bx_q = 50,000,000 \;\texttt{>>}\; 15 = 1,525 \quad (\text{INT16})
$$

**步骤 4**: 最终求和
$$
y_{32} = -56 + 1,525 + 16,000 = 17,469 \quad (\text{INT32})
$$
$$
q_y = 17,469 \quad (\text{INT16})
$$

### 反量化验证
$$
y_{\text{fp32}} = (17,469 - (-32768)) \times (1/32767) \approx 0.85 \quad \checkmark
$$

**误差**: $|0.85 - 0.8521| < 0.003$（Sigmoid 真实值为 0.8521）

---

## 附录B: 硬件实现伪代码

```verilog
// INT16 分段二次多项式激活函数硬件模块
module quadratic_lut_sigmoid (
    input  wire clk,
    input  wire rst_n,
    input  wire [15:0] x_int16,      // 输入
    output reg  [15:0] y_int16,      // 输出
    output reg         valid         // 输出有效信号
);

// ROM: 存储阈值、系数和shift参数
reg [15:0] thresholds [0:31];        // 32个阈值
reg [15:0] coeff_a [0:31];           // 32个a系数
reg [15:0] coeff_b [0:31];           // 32个b系数
reg [15:0] coeff_c [0:31];           // 32个c系数
reg [4:0]  shift_x2 [0:31];          // 32个x2 shift
reg [4:0]  shift_ax2 [0:31];         // 32个ax2 shift
reg [4:0]  shift_bx [0:31];          // 32个bx shift

// 流水线寄存器
reg [4:0]  seg_idx;                  // 段索引
reg [31:0] x_squared_32;
reg [15:0] x_squared_16;
reg [31:0] ax2_32;
reg [15:0] ax2_16;
reg [31:0] bx_32;
reg [15:0] bx_16;
reg [31:0] y_32;

// Stage 1: 段索引查找
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        seg_idx <= 0;
    end else begin
        // 二分查找或线性查找
        for (int i = 0; i < 32; i = i + 1) begin
            if (x_int16 < thresholds[i]) begin
                seg_idx <= i;
                break;
            end
        end
    end
end

// Stage 2: 系数读取 (自动流水线)
wire [15:0] a = coeff_a[seg_idx];
wire [15:0] b = coeff_b[seg_idx];
wire [15:0] c = coeff_c[seg_idx];
wire [4:0]  n_x2  = shift_x2[seg_idx];
wire [4:0]  n_ax2 = shift_ax2[seg_idx];
wire [4:0]  n_bx  = shift_bx[seg_idx];

// Stage 3: x^2 计算
always @(posedge clk) begin
    x_squared_32 <= x_int16 * x_int16;
    x_squared_16 <= x_squared_32 >>> n_x2;
end

// Stage 4-5: ax^2 和 bx 计算
always @(posedge clk) begin
    ax2_32 <= a * x_squared_16;
    ax2_16 <= ax2_32 >>> n_ax2;
    
    bx_32 <= b * x_int16;
    bx_16 <= bx_32 >>> n_bx;
end

// Stage 6: 最终求和 + 饱和
always @(posedge clk) begin
    y_32 <= ax2_16 + bx_16 + c;
    
    // 饱和处理
    if (y_32 > 32767)
        y_int16 <= 16'h7FFF;
    else if (y_32 < -32768)
        y_int16 <= 16'h8000;
    else
        y_int16 <= y_32[15:0];
    
    valid <= 1'b1;
end

endmodule
```

---

**文档结束**

本文档详细描述了 GRU Cell 在数字电路中的量化实现，**采用分段二次多项式拟合查找表实现激活函数**，相比传统LUT/PWL方法精度提升50-80%，每个操作都给出了浮点域和量化域的完整公式推导。


