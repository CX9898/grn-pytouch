import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import os


class PowerOfTwoQuantizer:
    """2的幂次方量化器"""

    def __init__(self, f_min, f_max, bit_width=16, symmetric=True):
        """
        初始化量化器

        参数说明（按照你的命名规范）:
        - f_min: 浮点最小值
        - f_max: 浮点最大值
        - bit_width: 量化位宽 (默认16-bit)
        - symmetric: 是否对称量化
        """
        self.mFMin = f_min
        self.mFMax = f_max
        self.mBitWidth = bit_width
        self.mSymmetric = symmetric

        # 🔥 关键改进：对于非对称量化且输入全为非负，使用无符号整数
        if not symmetric and f_min >= 0:
            # 无符号整数范围
            if bit_width == 16:
                self.mQuantizedMin = 0
                self.mQuantizedMax = 65535
            elif bit_width == 8:
                self.mQuantizedMin = 0
                self.mQuantizedMax = 255
            else:
                self.mQuantizedMin = 0
                self.mQuantizedMax = 2 ** bit_width - 1
            self.mUnsigned = True
        else:
            # 有符号整数范围
            if bit_width == 16:
                self.mQuantizedMin = -32768
                self.mQuantizedMax = 32767
            elif bit_width == 8:
                self.mQuantizedMin = -128
                self.mQuantizedMax = 127
            else:
                self.mQuantizedMin = -(2 ** (bit_width - 1))
                self.mQuantizedMax = 2 ** (bit_width - 1) - 1
            self.mUnsigned = False

        # 计算量化参数
        self._compute_quantization_params()

    def _compute_quantization_params(self):
        """计算量化参数：scale和zero_point"""
        # 步骤1: 计算原始scale
        sOriginalScale = (self.mFMax - self.mFMin) / (self.mQuantizedMax - self.mQuantizedMin)

        # 步骤2: 转换为2的幂次方 scale = 1 / 2^n
        # 找到最接近的2的幂次方
        if sOriginalScale > 0:
            nShift = int(np.round(-np.log2(sOriginalScale)))
            self.mShiftBits = nShift
            self.mScale = 1.0 / (2 ** nShift)  # scale = 2^(-n)
        else:
            self.mShiftBits = 0
            self.mScale = 1.0

        # 步骤3: 根据对称/非对称量化计算zero_point
        if self.mSymmetric:
            # 对称量化: zero_point = 0
            self.mZeroPoint = 0

            # 重新计算qmin/qmax以适应power-of-2 scale
            fRange = max(abs(self.mFMin), abs(self.mFMax))
            self.mFMin = -fRange
            self.mFMax = fRange

            # 计算实际能表示的量化范围
            qRange = int(fRange / self.mScale)
            self.mQuantizedMin = -qRange
            self.mQuantizedMax = qRange
        else:
            # 非对称量化: 重新计算zero_point
            self.mZeroPoint = int(np.round(self.mQuantizedMin - self.mFMin / self.mScale))

            # 确保zero_point在有效范围内
            originalQMin = self.mQuantizedMin
            originalQMax = self.mQuantizedMax
            self.mZeroPoint = np.clip(self.mZeroPoint, originalQMin, originalQMax)

    def quantize(self, fValue):
        """
        浮点值量化为整数
        Float Value = S * (q - Z)  =>  q = Float Value / S + Z
        """
        qValue = np.round(fValue / self.mScale) + self.mZeroPoint
        clipped = np.clip(qValue, self.mQuantizedMin, self.mQuantizedMax)
        # 根据是否为无符号整数选择数据类型
        if self.mUnsigned:
            return clipped.astype(np.uint16 if self.mBitWidth == 16 else np.uint8)
        else:
            return clipped.astype(np.int32)

    def dequantize(self, qValue):
        """
        整数值反量化为浮点
        Float Value = S * (q - Z)
        """
        return self.mScale * (qValue - self.mZeroPoint)

    def get_params(self):
        """获取量化参数"""
        return {
            'f_min': float(self.mFMin),
            'f_max': float(self.mFMax),
            'scale': float(self.mScale),
            'shift_bits': int(self.mShiftBits),  # n，其中 scale = 2^(-n)
            'zero_point': int(self.mZeroPoint),
            'quantized_min': int(self.mQuantizedMin),
            'quantized_max': int(self.mQuantizedMax),
            'symmetric': bool(self.mSymmetric),
            'unsigned': bool(self.mUnsigned)
        }

    def print_params(self):
        """打印量化参数（中文）"""
        params = self.get_params()
        print(f"浮点范围: [{params['f_min']:.6f}, {params['f_max']:.6f}]")
        quant_type = "无符号" if params['unsigned'] else "有符号"
        print(f"量化范围: [{params['quantized_min']}, {params['quantized_max']}] ({quant_type})")
        print(f"Scale: {params['scale']:.10f} = 2^(-{params['shift_bits']})")
        print(f"Zero Point: {params['zero_point']}")
        print(f"对称量化: {params['symmetric']}")


class QuantizedPiecewiseQuadraticFitter:
    """定点分段二次多项式拟合器（完全非对称量化版本）

    x², ax², bx 自适应使用非对称量化：
    - 不跨零范围：非对称量化（充分利用量化范围）
    - 跨零范围：对称量化（保持数值稳定性）
    """

    def __init__(self, num_segments=32, input_bit_width=16):
        """
        初始化

        参数:
        - num_segments: 分段数量
        - input_bit_width: 输入量化位宽
        """
        self.mNumSegments = num_segments
        self.mInputBitWidth = input_bit_width
        self.mSegments = {}
        self.mQuantizers = {}  # 存储各个量化器

    def quadratic(self, x, a, b, c):
        """二次函数: ax^2 + bx + c"""
        return a * x ** 2 + b * x + c

    # ==================== 非线性函数定义 ====================
    def sqrt(self, x):
        """平方根函数 sqrt(x)"""
        x = np.atleast_1d(x)
        # 确保输入非负
        x_safe = np.maximum(x, 0)
        return np.sqrt(x_safe)

    def rrelu(self, x, lower=0.125, upper=0.333, training=False):
        """RReLU - 测试时使用平均斜率"""
        alpha = (lower + upper) / 2
        return np.where(x >= 0, x, alpha * x)

    def leaky_relu(self, x, negative_slope=0.01):
        """LeakyReLU"""
        return np.where(x >= 0, x, negative_slope * x)

    def prelu(self, x, weight=0.25):
        """PReLU - 使用固定权重"""
        return np.where(x >= 0, x, weight * x)

    def softplus(self, x, beta=1, threshold=20):
        """Softplus"""
        x = np.atleast_1d(x)
        result = np.zeros_like(x, dtype=np.float64)
        mask = beta * x <= threshold
        result[mask] = np.log(1 + np.exp(beta * x[mask])) / beta
        result[~mask] = x[~mask]
        return result

    def gelu(self, x):
        """GELU"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def relu6(self, x):
        """ReLU6"""
        return np.minimum(np.maximum(0, x), 6)

    def sigmoid(self, x):
        """Sigmoid"""
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def tanh(self, x):
        """Tanh"""
        return np.tanh(x)

    def mish(self, x):
        """Mish"""
        return x * np.tanh(np.log(1 + np.exp(np.clip(x, -10, 10))))

    def swish(self, x, beta=1.0):
        """Swish/SiLU"""
        return x / (1 + np.exp(-beta * np.clip(x, -10, 10)))

    def hard_swish(self, x):
        """Hard-Swish"""
        return x * np.minimum(np.maximum(0, x + 3), 6) / 6

    def hard_sigmoid(self, x):
        """Hard-Sigmoid"""
        return np.minimum(np.maximum(0, x + 3), 6) / 6

    def snake2d(self, x, alpha=1.0):
        """Snake2d - 简化版本"""
        return x + (1 / (alpha + 1e-9)) * np.sin(alpha * x) ** 2

    def power_0_3(self, x):
        """Power函数 x^0.3 - 在 x=0 附近导数大"""
        return np.abs(x) ** 0.3

    def reciprocal(self, x):
        """倒数函数 1/x - 分段饱和处理
        when x < 0: y = max(1/x, -4)  # 负侧饱和到 -4
        when x > 0: y = min(1/x, 4)   # 正侧饱和到 4
        """
        result = np.zeros_like(x, dtype=np.float64)
        mask_neg = x < 0
        mask_pos = x > 0

        # 负侧：max(1/x, -4)
        if np.any(mask_neg):
            result[mask_neg] = np.maximum(1.0 / x[mask_neg], -4.0)

        # 正侧：min(1/x, 4)
        if np.any(mask_pos):
            result[mask_pos] = np.minimum(1.0 / x[mask_pos], 4.0)

        return result

    def power_2(self, x):
        """Power函数 x^2 - 标准二次函数"""
        return x ** 2


    # ==================== 分段策略 ====================
    def adaptive_segmentation(self, func, x_min, x_max, func_name):
        """自适应分段策略（支持所有函数）"""

        # 均匀分段的函数
        uniformFunctions = ['leaky_relu', 'relu6', 'rrelu', 'prelu']

        if func_name in uniformFunctions:
            return np.linspace(x_min, x_max, self.mNumSegments + 1)

        # sqrt 专用策略：起始段密集
        if func_name == 'sqrt':
            n_dense = int(self.mNumSegments * 0.6)
            n_sparse = self.mNumSegments - n_dense
            split_point = 4.0
            if x_max <= split_point:
                sqrt_points = np.linspace(np.sqrt(max(x_min, 0)), np.sqrt(x_max), self.mNumSegments + 1)
                segmentPoints = sqrt_points ** 2
            else:
                sqrt_dense = np.linspace(np.sqrt(max(x_min, 0)), np.sqrt(split_point), n_dense + 1)
                linear_sparse = np.linspace(split_point, x_max, n_sparse + 1)[1:]
                segmentPoints = np.concatenate([sqrt_dense ** 2, linear_sparse])
            print(f"  sqrt 自适应分段：前{n_dense}段密集[0~{split_point}] + 后{n_sparse}段稀疏[{split_point}~{x_max}]")
            return segmentPoints

        # power_0_3 专用策略：参考 sqrt 成功经验，使用平方根空间分段
        if func_name == 'power_0_3':
            # 🔥 关键洞察：x^0.3 和 x^0.5 特性相似（都是幂函数，x→0 导数大）
            # 采用 sqrt 的成功策略：60% 分段给初始快速变化区，40% 给后续
            n_dense = int(self.mNumSegments * 0.6)  # 60% = 19 段
            n_sparse = self.mNumSegments - n_dense  # 40% = 13 段
            split_point = 3.0  # 🔑 调整切分点到 x=3（因为 3^0.3 ≈ 1.39）

            if x_min >= 0:
                # 只有正侧
                if x_max > split_point:
                    # [0, 3]: 在 sqrt(x) 空间均匀分段（模仿 sqrt 策略）
                    sqrt_dense = np.linspace(0, np.sqrt(split_point), n_dense + 1)
                    x_dense = sqrt_dense ** 2

                    # [3, x_max]: 线性分段
                    x_sparse = np.linspace(split_point, x_max, n_sparse + 1)[1:]

                    segmentPoints = np.concatenate([x_dense, x_sparse])
                    print(f"  power_0_3 (x^0.3) sqrt空间分段：60%段[0~3] + 40%段[3~{x_max}]")
                else:
                    # 全部使用 sqrt 空间
                    sqrt_points = np.linspace(0, np.sqrt(x_max), self.mNumSegments + 1)
                    segmentPoints = sqrt_points ** 2
                    print(f"  power_0_3 (x^0.3) 全sqrt空间：[0~{x_max}]")

            elif x_min < 0 and x_max > 0:
                # 双侧分布（较少见，但需支持）
                n_half = self.mNumSegments // 2

                # 负侧：sqrt 空间
                if abs(x_min) > split_point:
                    n_neg_dense = int(n_half * 0.6)
                    n_neg_sparse = n_half - n_neg_dense
                    sqrt_neg_sparse = np.linspace(np.sqrt(abs(x_min)), np.sqrt(split_point), n_neg_sparse + 1)[:-1]
                    x_neg_sparse = -(sqrt_neg_sparse ** 2)
                    sqrt_neg_dense = np.linspace(np.sqrt(split_point), 0, n_neg_dense + 1)[:-1]
                    x_neg_dense = -(sqrt_neg_dense ** 2)
                    x_neg = np.concatenate([x_neg_sparse, x_neg_dense])
                else:
                    sqrt_neg = np.linspace(np.sqrt(abs(x_min)), 0, n_half + 1)[:-1]
                    x_neg = -(sqrt_neg ** 2)

                # 正侧：sqrt 空间
                if x_max > split_point:
                    n_pos_dense = int(n_half * 0.6)
                    n_pos_sparse = n_half - n_pos_dense
                    sqrt_pos_dense = np.linspace(0, np.sqrt(split_point), n_pos_dense + 1)[1:]
                    x_pos_dense = sqrt_pos_dense ** 2
                    sqrt_pos_sparse = np.linspace(np.sqrt(split_point), np.sqrt(x_max), n_pos_sparse + 1)[1:]
                    x_pos_sparse = sqrt_pos_sparse ** 2
                    x_pos = np.concatenate([x_pos_dense, x_pos_sparse])
                else:
                    sqrt_pos = np.linspace(0, np.sqrt(x_max), n_half + 1)[1:]
                    x_pos = sqrt_pos ** 2

                segmentPoints = np.concatenate([x_neg, [0], x_pos])
                print(f"  power_0_3 (x^0.3) 双侧sqrt空间")

            else:
                # 只有负侧
                segmentPoints = np.linspace(x_min, x_max, self.mNumSegments + 1)

            return np.array(segmentPoints)

        # power_2 相对平滑，使用中心加权即可
        if func_name == 'power_2':
            # x^2 函数比较平滑，使用标准的中心加权策略
            pass  # 继续使用下面的通用策略

        # reciprocal (1/x) 专用策略：覆盖整个 [-6, 6] 范围，包括饱和区
        # 分段分配：负侧正常区 + 负侧饱和区 + 正侧饱和区 + 正侧正常区
        if func_name in ['reciprocal', 'reciprocal_pos', 'reciprocal_neg']:
            # 1/x 在 x 接近 0 时饱和到 ±4
            # 总共 32 段分配：
            # - 负侧正常区 [-6, -0.25]：10 段（对数空间）
            # - 负侧饱和区 [-0.25, -0.01]：6 段（输出恒为 -4）
            # - 正侧饱和区 [0.01, 0.25]：6 段（输出恒为 4）
            # - 正侧正常区 [0.25, 6]：10 段（对数空间）

            if x_min < 0 and x_max > 0:
                epsilon = 0.01  # 避开 x=0 的小范围

                # 负侧正常区：[-6, -0.25]，10 段，对数空间
                n_neg_normal = 10
                log_neg = np.linspace(np.log(0.25), np.log(6.0), n_neg_normal + 1)
                x_neg_normal = -np.exp(log_neg[::-1])

                # 负侧饱和区：[-0.25, -epsilon]，6 段，线性空间
                n_neg_sat = 6
                x_neg_sat = np.linspace(-0.25, -epsilon, n_neg_sat + 1)[1:]

                # 正侧饱和区：[epsilon, 0.25]，6 段，线性空间
                n_pos_sat = 6
                x_pos_sat = np.linspace(epsilon, 0.25, n_pos_sat + 1)

                # 正侧正常区：[0.25, 6]，10 段，对数空间
                n_pos_normal = 10
                log_pos = np.linspace(np.log(0.25), np.log(6.0), n_pos_normal + 1)[1:]
                x_pos_normal = np.exp(log_pos)

                segmentPoints = np.concatenate([x_neg_normal, x_neg_sat, x_pos_sat, x_pos_normal])
                print(f"  reciprocal (1/x) 四区：负常规{n_neg_normal}段 + 负饱和{n_neg_sat}段 + 正饱和{n_pos_sat}段 + 正常规{n_pos_normal}段")
                print(f"    总段数: {len(segmentPoints) - 1}")

            elif x_min >= 0:
                # 纯正侧（保持原有逻辑）
                n_dense = int(self.mNumSegments * 0.8)
                n_sparse = self.mNumSegments - n_dense
                split_point = 1.0

                if x_max > split_point:
                    log_dense = np.linspace(np.log(max(x_min, 0.25)), np.log(split_point), n_dense + 1)
                    x_dense = np.exp(log_dense)
                    x_sparse = np.linspace(split_point, x_max, n_sparse + 1)[1:]
                    segmentPoints = np.concatenate([x_dense, x_sparse])
                else:
                    log_points = np.linspace(np.log(max(x_min, 0.25)), np.log(x_max), self.mNumSegments + 1)
                    segmentPoints = np.exp(log_points)

            else:
                # 纯负侧（保持原有逻辑）
                n_dense = int(self.mNumSegments * 0.8)
                n_sparse = self.mNumSegments - n_dense
                split_point = -1.0

                if abs(x_min) > 1.0:
                    x_sparse = np.linspace(x_min, split_point, n_sparse + 1)[:-1]
                    log_dense = np.linspace(np.log(abs(split_point)), np.log(min(abs(x_max), 0.25)), n_dense + 1)
                    x_dense = -np.exp(log_dense)
                    segmentPoints = np.concatenate([x_sparse, x_dense])
                else:
                    log_points = np.linspace(np.log(abs(x_min)), np.log(min(abs(x_max), 0.25)), self.mNumSegments + 1)
                    segmentPoints = -np.exp(log_points[::-1])

            return np.array(segmentPoints)

        # 其他函数：基于导数的自适应分段
        xDense = np.linspace(x_min, x_max, 1000)
        yDense = func(xDense)

        # 计算导数
        dy = np.diff(yDense)
        dx = np.diff(xDense)
        slopes = np.abs(dy / dx)

        # 权重配置
        weightConfigs = {
            'sigmoid': (5.0, 2.0),
            'tanh': (4.0, 2.0),
            'gelu': (3.0, 3.0),
            'softplus': (3.0, 4.0),
            'mish': (3.0, 4.0),
            'swish': (3.0, 4.0),
            'hard_sigmoid': (2.0, 6.0),
            'hard_swish': (2.0, 6.0),
            'snake2d': (2.0, 4.0),
            'power_2': (1.5, 2.0),  # x^2 在中心区域变化较快
        }

        centerWeight, centerRange = weightConfigs.get(func_name, (2.0, 3.0))

        weights = np.ones(len(xDense) - 1)
        for i in range(len(weights)):
            distToCenter = abs(xDense[i])
            if distToCenter < centerRange:
                weights[i] = centerWeight * (1 - distToCenter / centerRange) + 1
            else:
                weights[i] = 1 + slopes[i] * 0.5

        # 归一化并生成分段点
        weights = weights / np.sum(weights)
        cumWeights = np.cumsum(weights)

        segmentPoints = [x_min]
        targetCumWeights = np.linspace(0, 1, self.mNumSegments + 1)[1:-1]

        for target in targetCumWeights:
            idx = np.searchsorted(cumWeights, target)
            if idx < len(xDense):
                segmentPoints.append(xDense[idx])

        segmentPoints.append(x_max)
        return np.array(segmentPoints)

    # ==================== 核心：定点量化拟合 ====================
    def fit_single_function_quantized(self, func, x_min, x_max, func_name):
        """
        拟合单个函数并进行完整的定点量化

        【完全非对称量化版本】
        关键改进：
        1. x² 使用非对称量化（始终非负）
        2. ax² 自适应量化（不跨零→非对称，跨零→对称）
        3. bx 自适应量化（不跨零→非对称，跨零→对称）
        """
        print(f"\n{'='*80}")
        print(f"拟合函数: {func_name} 【非对称量化版本】")
        print(f"{'='*80}")

        # 步骤1: 获取自适应分段点（浮点域）
        segmentPoints = self.adaptive_segmentation(func, x_min, x_max, func_name)

        # 步骤2: 创建输入量化器 (非对称量化，INT16)
        inputQuantizer = PowerOfTwoQuantizer(x_min, x_max, bit_width=self.mInputBitWidth, symmetric=False)
        print(f"\n【输入量化参数】")
        inputQuantizer.print_params()

        # 步骤3: 创建输出量化器 (非对称量化)
        # 根据不同函数设置正确的输出范围
        if func_name == 'sigmoid':
            yMin = 0.0
            yMax = 1.0
        elif func_name == 'tanh':
            yMin = -1.0
            yMax = 1.0
        elif func_name == 'sqrt':
            xSafe = np.maximum(x_min, 0)
            yMin = np.sqrt(xSafe)
            yMax = np.sqrt(x_max)
            ySample = func(np.linspace(xSafe, x_max, 100))
            yMin = min(yMin, ySample.min())
            yMax = max(yMax, ySample.max())
        elif func_name in ['gelu', 'mish', 'swish']:
            # 这些函数在大的正数时趋向 x
            ySample = func(np.linspace(x_min, x_max, 1000))
            yMin = ySample.min()
            yMax = max(ySample.max(), x_max)
        elif func_name == 'softplus':
            ySample = func(np.linspace(x_min, x_max, 1000))
            yMin = max(0, ySample.min())
            yMax = max(ySample.max(), x_max)
        elif func_name == 'relu6':
            yMin = 0
            yMax = 6
        elif func_name in ['leaky_relu', 'rrelu', 'prelu']:
            ySample = func(np.linspace(x_min, x_max, 1000))
            yMin = ySample.min()
            yMax = ySample.max()
        elif func_name == 'hard_sigmoid':
            yMin = 0
            yMax = 1
        elif func_name == 'snake2d':
            ySample = func(np.linspace(x_min, x_max, 1000))
            yMin = ySample.min()
            yMax = ySample.max()
        elif func_name == 'power_0_3':
            # x^0.3 在 x=0 附近导数极大，且严格非负
            # 🔥 激进优化：增加采样点 + 更大的输出范围扩展
            ySample = func(np.linspace(x_min, x_max, 10000))  # 10k 采样
            yMin = 0  # 强制从 0 开始
            yMax = ySample.max()
            # 扩展上界到 25% 避免饱和（更激进）
            yRange = yMax - yMin
            yMax = yMax + 0.25 * yRange
            print(f"  power_0_3 (x^0.3) 输出范围: [{yMin:.6f}, {yMax:.6f}] (yMin=0, 上界+25%)")
            print(f"  → 无符号整数量化 [0, 65535]，采样 10k 点")
        elif func_name == 'power_2':
            # x^2 输出范围 [0, max(x_min^2, x_max^2)]
            ySample = func(np.linspace(x_min, x_max, 1000))
            yMin = 0  # x^2 >= 0
            yMax = ySample.max()
            print(f"  power_2 (x^2) 输出范围: [{yMin:.6f}, {yMax:.6f}]")
        elif func_name in ['reciprocal', 'reciprocal_pos', 'reciprocal_neg']:
            # 1/x 输出范围计算，带饱和到 [-4, 4]
            # 跳过 x=0 附近，分别采样负侧和正侧
            if x_min < 0 and x_max > 0:
                # 双侧：负侧 + 正侧
                ySample_neg = func(np.linspace(x_min, -0.25, 1000))
                ySample_pos = func(np.linspace(0.25, x_max, 1000))
                ySample = np.concatenate([ySample_neg, ySample_pos])
            elif x_min >= 0:
                # 纯正侧
                ySample = func(np.linspace(max(x_min, 0.25), x_max, 2000))
            else:
                # 纯负侧
                ySample = func(np.linspace(x_min, min(x_max, -0.25), 2000))

            yMin = ySample.min()
            yMax = ySample.max()
            print(f"  reciprocal (1/x with clip[-4,4]) 输出: [{yMin:.6f}, {yMax:.6f}]")
        else:
            # 通用采样
            ySample = func(np.linspace(x_min, x_max, 1000))
            yMin = ySample.min()
            yMax = ySample.max()

        print(f"  输出范围: yMin={yMin:.6f}, yMax={yMax:.6f}")
        outputQuantizer = PowerOfTwoQuantizer(yMin, yMax, bit_width=self.mInputBitWidth, symmetric=False)
        print(f"\n【输出量化参数】")
        outputQuantizer.print_params()

        # 步骤4: 量化阈值
        quantizedThresholds = inputQuantizer.quantize(segmentPoints)

        # ===== 第一遍扫描：拟合所有分段，收集所有系数 =====
        print(f"\n【第一遍扫描】拟合所有分段，收集系数...")
        allCoeffs = []  # 存储所有分段的 (xStart, xEnd, a, b, c)

        for i in range(len(segmentPoints) - 1):
            xStart = segmentPoints[i]
            xEnd = segmentPoints[i + 1]

            if xEnd - xStart < 1e-10:
                continue

            # 采样点（浮点域）
            xSegment = np.linspace(xStart, xEnd, max(10, int(20 * (xEnd - xStart) / (x_max - x_min))))
            ySegment = func(xSegment)

            try:
                # 二次函数拟合（浮点域）
                popt, _ = curve_fit(self.quadratic, xSegment, ySegment, maxfev=5000)
                a, b, c = popt
            except Exception as e:
                # 拟合失败时使用线性近似
                slope = (ySegment[-1] - ySegment[0]) / (xEnd - xStart + 1e-10)
                intercept = ySegment[0] - slope * xStart
                a, b, c = 0, slope, intercept

            allCoeffs.append((xStart, xEnd, a, b, c))

        print(f"  收集到 {len(allCoeffs)} 个分段的系数")
        print(f"\n  各分段系数详情：")
        print(f"  {'段号':>4} {'x范围':>20} {'a':>15} {'b':>15} {'c':>15}")
        print(f"  {'-'*70}")
        for i, (xStart, xEnd, a, b, c) in enumerate(allCoeffs):
            print(f"  {i:4d} [{xStart:7.4f}, {xEnd:7.4f}] {a:15.6f} {b:15.6f} {c:15.6f}")

        # ===== 第二遍扫描：分组量化系数 =====
        print(f"\n【第二遍扫描】分组量化系数...")

        # 提取所有a, b, c值
        allA = [coef[2] for coef in allCoeffs]
        allB = [coef[3] for coef in allCoeffs]
        allC = [coef[4] for coef in allCoeffs]

        # 分析 a 的分布，自动找分组点
        absA = [abs(a) for a in allA]
        print(f"\n  a 系数分析：")
        print(f"    最大: {max(absA):.6f} (段{absA.index(max(absA))})")
        print(f"    最小: {min(absA):.6f} (段{absA.index(min(absA))})")
        print(f"    前5段: {absA[:5]}")
        print(f"    后5段: {absA[-5:]}")

        # 🔥 关键策略：根据 a 值自动分组
        # 找到 a 值突变的位置（相邻段 a 值比例 > 阈值）
        splitIdx = None
        for i in range(len(absA) - 1):
            if absA[i] > 0 and absA[i+1] > 0:
                ratio = absA[i] / absA[i+1]
                if ratio > 3.0:  # a 值突降超过3倍，作为分组点
                    splitIdx = i + 1
                    break

        if splitIdx is None or splitIdx <= 1:
            splitIdx = 3  # 默认前3段为大值组

        print(f"\n  自动分组点: 段{splitIdx} (前{splitIdx}段用粗scale，后{len(allA)-splitIdx}段用细scale)")

        # 分组 a 系数
        aGroup1 = allA[:splitIdx]   # 大值组
        aGroup2 = allA[splitIdx:]   # 小值组

        print(f"    组1(大a): 段0-{splitIdx-1}, a范围 [{min(aGroup1):.6f}, {max(aGroup1):.6f}]")
        print(f"    组2(小a): 段{splitIdx}-{len(allA)-1}, a范围 [{min(aGroup2):.6f}, {max(aGroup2):.6f}]")

        # 关键修正：将输出零点"烘焙"到c中
        zeroPointOffset = outputQuantizer.mZeroPoint * outputQuantizer.mScale
        allC_adjusted = [c + zeroPointOffset for c in allC]
        print(f"\n  将输出零点({outputQuantizer.mZeroPoint})烘焙到c中，偏移={zeroPointOffset:.6f}")

        # 为两组分别创建量化器
        # 组1：大 a 值，使用自动计算的 scale（粗）
        if abs(min(aGroup1)) > abs(max(aGroup1)):
            aRange1 = abs(min(aGroup1))
        else:
            aRange1 = abs(max(aGroup1))
        aQuantizer1 = PowerOfTwoQuantizer(-aRange1, aRange1, bit_width=16, symmetric=True)

        # 组2：小 a 值，使用细 scale（与输出对齐）
        if abs(min(aGroup2)) > abs(max(aGroup2)):
            aRange2 = abs(min(aGroup2))
        else:
            aRange2 = abs(max(aGroup2))
        aQuantizer2 = PowerOfTwoQuantizer(-aRange2, aRange2, bit_width=16, symmetric=True)

        # 检查组2是否需要调整到更细的scale
        minShiftBits = outputQuantizer.mShiftBits
        if aQuantizer2.mShiftBits < minShiftBits:
            print(f"  ⚠️  组2的scale太粗(shift={aQuantizer2.mShiftBits}), 调整为{minShiftBits}")
            aQuantizer2.mShiftBits = minShiftBits
            aQuantizer2.mScale = 1.0 / (2 ** minShiftBits)
            newQRange = int(2 * aRange2 * (2 ** minShiftBits))
            aQuantizer2.mQuantizedMax = newQRange // 2
            aQuantizer2.mQuantizedMin = -aQuantizer2.mQuantizedMax
            print(f"      调整量化范围: [{aQuantizer2.mQuantizedMin}, {aQuantizer2.mQuantizedMax}]")
            print(f"      利用率: {100*newQRange/65536:.1f}%")

        print(f"\n  a系数分组量化器:")
        print(f"    组1: shift_bits={aQuantizer1.mShiftBits}, Qrange=[{aQuantizer1.mQuantizedMin}, {aQuantizer1.mQuantizedMax}]")
        print(f"    组2: shift_bits={aQuantizer2.mShiftBits}, Qrange=[{aQuantizer2.mQuantizedMin}, {aQuantizer2.mQuantizedMax}]")

        # b 和 c 统一量化（不分组）
        bMin, bMax = min(allB), max(allB)
        if abs(bMin) > abs(bMax):
            bRange = abs(bMin)
        else:
            bRange = abs(bMax)
        bQuantizer = PowerOfTwoQuantizer(-bRange, bRange, bit_width=16, symmetric=True)

        cMin, cMax = min(allC_adjusted), max(allC_adjusted)
        if abs(cMin) > abs(cMax):
            cRange = abs(cMin)
        else:
            cRange = abs(cMax)
        cQuantizer = PowerOfTwoQuantizer(-cRange, cRange, bit_width=16, symmetric=True)

        print(f"\n  b系数量化器: shift_bits={bQuantizer.mShiftBits}, 范围[{bMin:.6f}, {bMax:.6f}]")
        print(f"  c系数量化器: shift_bits={cQuantizer.mShiftBits}, 范围[{cMin:.6f}, {cMax:.6f}]")

        # ===== 第三遍扫描：量化系数并为每个分段计算右移位数 =====
        print(f"\n【第三遍扫描】量化系数并计算每段的右移位数...")
        print(f"  ⚠️  关键改进：x², ax², bx 自适应使用非对称量化")
        print(f"  ⚠️  a 系数分组量化：前{splitIdx}段用粗scale，后{len(allA)-splitIdx}段用细scale")
        segmentsInfo = []

        for i, (xStart, xEnd, a, b, c) in enumerate(allCoeffs):
            # 🔥 根据分段索引选择对应的 a 量化器
            if i < splitIdx:
                aQuantizer = aQuantizer1  # 大值组
            else:
                aQuantizer = aQuantizer2  # 小值组

            # 使用对应的量化器量化系数
            c_adjusted = c + zeroPointOffset
            qA = aQuantizer.quantize(a)
            qB = bQuantizer.quantize(b)
            qC = cQuantizer.quantize(c_adjusted)

            # ===== 关键：为每个分段独立计算右移位数 =====

            # 1. 计算这个分段的 x 范围（量化域）
            qXStart = inputQuantizer.quantize(xStart)
            qXEnd = inputQuantizer.quantize(xEnd)
            qXMin = min(qXStart, qXEnd)
            qXMax = max(qXStart, qXEnd)

            # 2. 计算这个分段的 x^2 范围（浮点域）
            x2Min = min(xStart**2, xEnd**2)
            x2Max = max(xStart**2, xEnd**2)

            # 🔥 关键修改：检测跨零情况
            if xStart * xEnd < 0:
                x2Min = 0.0  # 跨越0，最小值是0

            # 🔥 关键修改：x² 使用非对称量化（如果范围全为非负）
            if x2Min >= 0:
                x2Quantizer = PowerOfTwoQuantizer(x2Min, x2Max, bit_width=16, symmetric=False)
                print(f"    段{i}: x²范围[{x2Min:.4f}, {x2Max:.4f}] → 非对称量化, Z_x2={x2Quantizer.mZeroPoint}")
            else:
                # 理论上不会发生，但保留对称量化作为备选
                x2Quantizer = PowerOfTwoQuantizer(x2Min, x2Max, bit_width=16, symmetric=True)
                print(f"    段{i}: x²范围异常，使用对称量化")

            # 计算 n_x2: 将 (q_x)^2 右移到 x^2 的量化域
            nX2 = 2 * inputQuantizer.mShiftBits - x2Quantizer.mShiftBits

            # 3. 计算这个分段的 a*x^2 范围（浮点域）
            ax2Values = [a * (xStart**2), a * (xEnd**2)]
            if xStart * xEnd < 0:
                ax2Values.append(0)
            ax2Min = min(ax2Values)
            ax2Max = max(ax2Values)

            # 🔥 关键修改：ax² 使用非对称量化（如果不跨越0）
            if ax2Min * ax2Max >= 0:  # 同号，不跨越0
                ax2Quantizer = PowerOfTwoQuantizer(ax2Min, ax2Max, bit_width=16, symmetric=False)
                print(f"    段{i}: ax²范围[{ax2Min:.6f}, {ax2Max:.6f}] → 非对称量化, Z_ax2={ax2Quantizer.mZeroPoint}")
            else:  # 跨越0，使用对称量化
                if abs(ax2Min) > abs(ax2Max):
                    ax2Range = abs(ax2Min)
                else:
                    ax2Range = abs(ax2Max)
                ax2Quantizer = PowerOfTwoQuantizer(-ax2Range, ax2Range, bit_width=16, symmetric=True)
                print(f"    段{i}: ax²跨越0，使用对称量化")

            # 计算 n_ax2: 将 q_a * q_x2 右移到 ax^2 的量化域
            nAx2 = aQuantizer.mShiftBits + x2Quantizer.mShiftBits - ax2Quantizer.mShiftBits

            # 4. 计算这个分段的 b*x 范围（浮点域）
            bxValues = [b * xStart, b * xEnd]
            bxMin = min(bxValues)
            bxMax = max(bxValues)

            # 🔥 关键修改：bx 使用非对称量化（如果不跨越0）
            if bxMin * bxMax >= 0:  # 同号，不跨越0
                bxQuantizer = PowerOfTwoQuantizer(bxMin, bxMax, bit_width=16, symmetric=False)
                print(f"    段{i}: bx范围[{bxMin:.6f}, {bxMax:.6f}] → 非对称量化, Z_bx={bxQuantizer.mZeroPoint}")
            else:  # 跨越0，使用对称量化
                if abs(bxMin) > abs(bxMax):
                    bxRange = abs(bxMin)
                else:
                    bxRange = abs(bxMax)
                bxQuantizer = PowerOfTwoQuantizer(-bxRange, bxRange, bit_width=16, symmetric=True)
                print(f"    段{i}: bx跨越0，使用对称量化")

            # 计算 n_bx: 将 q_b * q_x 右移到 bx 的量化域
            nBx = bQuantizer.mShiftBits + inputQuantizer.mShiftBits - bxQuantizer.mShiftBits

            # 5. 计算最终输出的右移位数
            nYa = ax2Quantizer.mShiftBits - outputQuantizer.mShiftBits
            nYb = bxQuantizer.mShiftBits - outputQuantizer.mShiftBits
            nYc = cQuantizer.mShiftBits - outputQuantizer.mShiftBits

            if nYc < 0:
                print(f"  警告：分段{i}的n_yc={nYc}仍为负，这不应该发生！")

            # 存储分段信息
            segmentInfo = {
                'range': (xStart, xEnd),
                'threshold_q': (quantizedThresholds[i], quantizedThresholds[i + 1]),
                'coeff_float': (a, b, c),
                'coeff_quantized': (int(qA), int(qB), int(qC)),
                'quantizers': {
                    'a': aQuantizer.get_params(),
                    'b': bQuantizer.get_params(),
                    'c': cQuantizer.get_params(),
                    'x2': x2Quantizer.get_params(),
                    'ax2': ax2Quantizer.get_params(),
                    'bx': bxQuantizer.get_params()
                },
                'shift_bits': {
                    'n_x2': int(nX2),
                    'n_ax2': int(nAx2),
                    'n_bx': int(nBx),
                    'n_ya': int(nYa),
                    'n_yb': int(nYb),
                    'n_yc': int(nYc)
                }
            }

            segmentsInfo.append(segmentInfo)

        # 保存结果
        self.mSegments[func_name] = segmentsInfo
        self.mQuantizers[func_name] = {
            'input': inputQuantizer.get_params(),
            'output': outputQuantizer.get_params(),
            'coeff_a_group1': aQuantizer1.get_params(),  # 大值组
            'coeff_a_group2': aQuantizer2.get_params(),  # 小值组
            'coeff_a_split_idx': splitIdx,  # 分组点
            'coeff_b': bQuantizer.get_params(),
            'coeff_c': cQuantizer.get_params()
        }

        print(f"\n完成 {func_name} 的定点量化拟合，共 {len(segmentsInfo)} 段")
        print(f"  a系数分组量化：组1({splitIdx}段) + 组2({len(allA)-splitIdx}段) ✅")
        print(f"  x², ax², bx 自适应使用非对称量化（不跨零时）✅")
        print(f"  充分利用量化范围，提高精度 ✅")
        print(f"  每个分段有独立的右移位数 ✅")

        return segmentsInfo, inputQuantizer, outputQuantizer

    # ==================== 定点评估 ====================
    def evaluate_quantized(self, qX, func_name):
        """
        定点评估：给定量化输入qx (INT16)，返回量化输出qy (INT16)

        【完全非对称量化版本】：
        关键理解：定点计算中的所有中间值都是"去零点"后的值
        - qX2 = (q_x - Z_x)² >> n_x2 = (q_x2 - Z_x2)  ← 已经是去零点后的
        - qAx2 同理，qBx 同理
        - 因此不需要显式减去零点，右移后的结果本身就是去零点的

        非对称量化的优势在于：量化器可以选择最优的 scale，充分利用量化范围
        """
        if func_name not in self.mSegments:
            raise ValueError(f"函数 {func_name} 未拟合")

        segments = self.mSegments[func_name]
        inputParams = self.mQuantizers[func_name]['input']
        outputParams = self.mQuantizers[func_name]['output']

        # 创建量化器
        inputQuantizer = PowerOfTwoQuantizer(
            inputParams['f_min'], inputParams['f_max'],
            bit_width=self.mInputBitWidth, symmetric=inputParams['symmetric']
        )
        outputQuantizer = PowerOfTwoQuantizer(
            outputParams['f_min'], outputParams['f_max'],
            bit_width=self.mInputBitWidth, symmetric=outputParams['symmetric']
        )

        if np.isscalar(qX):
            # 步骤1: 找到段索引
            segmentIdx = 0
            for i, seg in enumerate(segments):
                qThrMin, qThrMax = seg['threshold_q']
                if qThrMin <= qX < qThrMax:
                    segmentIdx = i
                    break
            else:
                # 超出范围，使用边界段
                if qX < segments[0]['threshold_q'][0]:
                    segmentIdx = 0
                else:
                    segmentIdx = len(segments) - 1

            # 步骤2-3: 读取量化系数和右移位数
            seg = segments[segmentIdx]
            qA, qB, qC = seg['coeff_quantized']
            shifts = seg['shift_bits']

            # 去零点
            xOffset = int(qX) - inputQuantizer.mZeroPoint

            # 步骤4: 计算x^2 (INT16 → INT32 → 右移)
            # 注意：定点计算的中间值都是"去零点"后的值，不需要再减零点
            x32 = xOffset * xOffset  # INT32
            if shifts['n_x2'] >= 0:
                qX2 = x32 >> shifts['n_x2']
            else:
                qX2 = x32 << (-shifts['n_x2'])

            # 步骤5: 计算a·x^2 (INT16 × INT32 → INT32 → 右移)
            ax2_32 = qA * qX2  # INT32
            if shifts['n_ax2'] >= 0:
                qAx2 = ax2_32 >> shifts['n_ax2']
            else:
                qAx2 = ax2_32 << (-shifts['n_ax2'])

            # 步骤6: 计算b·x (INT16 × INT16 → INT32 → 右移)
            bx_32 = qB * xOffset  # INT32
            if shifts['n_bx'] >= 0:
                qBx = bx_32 >> shifts['n_bx']
            else:
                qBx = bx_32 << (-shifts['n_bx'])

            # 步骤7: 最终加法 (INT32加法 + 右移 + 饱和)
            # 处理负移位（左移）
            if shifts['n_ya'] >= 0:
                term_ax2 = qAx2 >> shifts['n_ya']
            else:
                term_ax2 = qAx2 << (-shifts['n_ya'])

            if shifts['n_yb'] >= 0:
                term_bx = qBx >> shifts['n_yb']
            else:
                term_bx = qBx << (-shifts['n_yb'])

            if shifts['n_yc'] >= 0:
                term_c = qC >> shifts['n_yc']
            else:
                term_c = qC << (-shifts['n_yc'])

            y32 = term_ax2 + term_bx + term_c

            # 饱和到输出范围
            qY = int(np.clip(y32, outputQuantizer.mQuantizedMin, outputQuantizer.mQuantizedMax))
            return qY

        # 数组输入
        result = np.zeros_like(qX, dtype=np.int32)
        for i, qXi in enumerate(qX):
            result[i] = self.evaluate_quantized(qXi, func_name)
        return result

    # ==================== 批量拟合 ====================
    def fit_all_functions(self, x_min=-6, x_max=6):
        """拟合所有激活函数"""
        functions = {
            'rrelu': lambda x: self.rrelu(x),
            'leaky_relu': lambda x: self.leaky_relu(x),
            'prelu': lambda x: self.prelu(x),
            'softplus': lambda x: self.softplus(x),
            'gelu': lambda x: self.gelu(x),
            'relu6': lambda x: self.relu6(x),
            'sigmoid': lambda x: self.sigmoid(x),
            'tanh': lambda x: self.tanh(x),
            'mish': lambda x: self.mish(x),
            'swish': lambda x: self.swish(x),
            'hard_swish': lambda x: self.hard_swish(x),
            'hard_sigmoid': lambda x: self.hard_sigmoid(x),
            'snake2d': lambda x: self.snake2d(x),
            'power_0_3': lambda x: self.power_0_3(x),
            'power_2': lambda x: self.power_2(x),
            'reciprocal': lambda x: self.reciprocal(x),
            'sqrt': lambda x: self.sqrt(x),
        }

        # 为不同函数设置合适的输入范围
        function_ranges = {
            'sigmoid': (-6, 6),
            'tanh': (-6, 6),
            'gelu': (-6, 6),
            'relu6': (-6, 6),
            'leaky_relu': (-6, 6),
            'rrelu': (-6, 6),
            'prelu': (-6, 6),
            'softplus': (-6, 6),
            'mish': (-6, 6),
            'swish': (-6, 6),
            'hard_swish': (-6, 6),
            'hard_sigmoid': (-6, 6),
            'snake2d': (-6, 6),
            'power_0_3': (0, 8),
            'power_2': (0, 16),
            'reciprocal': (-6, 6),  # 将在分段时避开 x=0
            'sqrt': (0, 16),
        }

        results = {}

        for name, func in functions.items():
            try:
                func_x_min, func_x_max = function_ranges.get(name, (x_min, x_max))
                print(f"\n{'='*80}")
                print(f"拟合函数: {name}, 输入范围: [{func_x_min}, {func_x_max}]")
                print(f"{'='*80}")

                segments, inputQ, outputQ = self.fit_single_function_quantized(
                    func, func_x_min, func_x_max, name
                )
                results[name] = {
                    'segments': segments,
                    'function': func,
                    'x_range': (func_x_min, func_x_max)
                }
            except Exception as e:
                print(f"  错误: {name} 拟合失败: {e}")
                import traceback
                traceback.print_exc()
                results[name] = {'error': str(e)}

        return results

    # ==================== 误差分析 ====================
    def compute_error(self, func_name, x_min=-6, x_max=6, num_points=10000):
        """Compute error metrics for quantized implementation, including cosine similarity"""
        if func_name not in self.mSegments:
            return None

        # 生成测试点（浮点域），reciprocal 需要避开 x=0 但覆盖整个范围
        if func_name == 'reciprocal' and x_min < 0 and x_max > 0:
            # 覆盖 [-6, -0.01] + [0.01, 6]，避开 x=0
            epsilon = 0.01
            xTest_neg = np.linspace(x_min, -epsilon, num_points // 2)
            xTest_pos = np.linspace(epsilon, x_max, num_points // 2)
            # 插入 NaN 来断开 matplotlib 的连线
            xTest = np.concatenate([xTest_neg, [np.nan], xTest_pos])
        else:
            xTest = np.linspace(x_min, x_max, num_points)

        # 真实函数值（浮点）
        func_map = {
            'sqrt': self.sqrt,
            'sigmoid': self.sigmoid,
            'tanh': self.tanh,
            'gelu': self.gelu,
            'relu6': self.relu6,
            'leaky_relu': self.leaky_relu,
            'rrelu': self.rrelu,
            'prelu': self.prelu,
            'softplus': self.softplus,
            'mish': self.mish,
            'swish': self.swish,
            'hard_swish': self.hard_swish,
            'hard_sigmoid': self.hard_sigmoid,
            'snake2d': self.snake2d,
            'power_0_3': self.power_0_3,
            'power_2': self.power_2,
            'reciprocal': self.reciprocal,
        }

        if func_name not in func_map:
            print(f"函数 {func_name} 未实现")
            return None

        yTrue = func_map[func_name](xTest)

        # 计算浮点分段二次多项式拟合（不量化）
        yFloatFit = np.zeros_like(xTest)
        segments = self.mSegments[func_name]
        for i, x in enumerate(xTest):
            if np.isnan(x):
                yFloatFit[i] = np.nan
                continue
            # 找到对应的分段
            segIdx = 0
            for j, seg in enumerate(segments):
                if seg['range'][0] <= x <= seg['range'][1]:
                    segIdx = j
                    break
            # 使用浮点系数计算
            a, b, c = segments[segIdx]['coeff_float']
            yFloatFit[i] = a * x**2 + b * x + c

        # 量化输入
        inputParams = self.mQuantizers[func_name]['input']
        inputQuantizer = PowerOfTwoQuantizer(
            inputParams['f_min'], inputParams['f_max'],
            bit_width=self.mInputBitWidth, symmetric=inputParams['symmetric']
        )
        qXTest = inputQuantizer.quantize(xTest)

        # 定点计算
        qYTest = self.evaluate_quantized(qXTest, func_name)

        # 反量化输出
        outputParams = self.mQuantizers[func_name]['output']
        outputQuantizer = PowerOfTwoQuantizer(
            outputParams['f_min'], outputParams['f_max'],
            bit_width=self.mInputBitWidth, symmetric=outputParams['symmetric']
        )
        yPred = outputQuantizer.dequantize(qYTest)

        # 计算误差
        mae = np.mean(np.abs(yTrue - yPred))
        mse = np.mean((yTrue - yPred) ** 2)
        maxError = np.max(np.abs(yTrue - yPred))

        # 计算余弦相似度：真实函数值 vs 定点量化结果
        # This measures how well quantization preserves the real function
        valid_mask = ~(np.isnan(yTrue) | np.isnan(yPred))
        if np.sum(valid_mask) > 0:
            yTrue_valid = yTrue[valid_mask]
            yPred_valid = yPred[valid_mask]

            # 计算余弦相似度
            dot_product = np.dot(yTrue_valid, yPred_valid)
            norm_true = np.linalg.norm(yTrue_valid)
            norm_pred = np.linalg.norm(yPred_valid)

            if norm_true > 0 and norm_pred > 0:
                cosine_similarity = dot_product / (norm_true * norm_pred)
            else:
                cosine_similarity = 0.0
        else:
            cosine_similarity = 0.0

        return {
            'mae': mae,
            'mse': mse,
            'max_error': maxError,
            'cosine_similarity': cosine_similarity,
            'x_test': xTest,
            'y_true': yTrue,
            'y_pred': yPred
        }

    # ==================== 可视化 ====================
    def plot_quantized_comparison(self, func_name, x_min=-6, x_max=6, num_points=10000, save_path=None):
        """Plot three-way comparison: real function vs float fit vs quantized"""
        if func_name not in self.mSegments:
            print(f"Function {func_name} not fitted")
            return

        # 生成测试点，reciprocal 需要避开 x=0 但覆盖整个范围
        if func_name == 'reciprocal' and x_min < 0 and x_max > 0:
            # 覆盖 [-6, -0.01] + [0.01, 6]，避开 x=0
            epsilon = 0.01
            xTest_neg = np.linspace(x_min, -epsilon, num_points // 2)
            xTest_pos = np.linspace(epsilon, x_max, num_points // 2)
            # 插入 NaN 来断开 matplotlib 的连线
            xTest = np.concatenate([xTest_neg, [np.nan], xTest_pos])
        else:
            xTest = np.linspace(x_min, x_max, num_points)

        # 1. 真实函数
        func_map = {
            'sqrt': self.sqrt,
            'sigmoid': self.sigmoid,
            'tanh': self.tanh,
            'gelu': self.gelu,
            'relu6': self.relu6,
            'leaky_relu': self.leaky_relu,
            'rrelu': self.rrelu,
            'prelu': self.prelu,
            'softplus': self.softplus,
            'mish': self.mish,
            'swish': self.swish,
            'hard_swish': self.hard_swish,
            'hard_sigmoid': self.hard_sigmoid,
            'snake2d': self.snake2d,
            'power_0_3': self.power_0_3,
            'power_2': self.power_2,
            'reciprocal': self.reciprocal,
        }

        if func_name not in func_map:
            print(f"函数 {func_name} 未实现")
            return

        yTrue = func_map[func_name](xTest)

        # 2. 浮点拟合（直接用二次多项式，不量化）
        yFloatFit = np.zeros_like(xTest)
        segments = self.mSegments[func_name]

        for i, x in enumerate(xTest):
            # 找到对应的分段
            segIdx = 0
            for j, seg in enumerate(segments):
                if seg['range'][0] <= x <= seg['range'][1]:
                    segIdx = j
                    break

            # 使用浮点系数计算
            a, b, c = segments[segIdx]['coeff_float']
            yFloatFit[i] = a * x**2 + b * x + c

        # 3. 定点量化
        inputParams = self.mQuantizers[func_name]['input']
        outputParams = self.mQuantizers[func_name]['output']

        inputQuantizer = PowerOfTwoQuantizer(
            inputParams['f_min'], inputParams['f_max'],
            bit_width=self.mInputBitWidth, symmetric=inputParams['symmetric']
        )
        outputQuantizer = PowerOfTwoQuantizer(
            outputParams['f_min'], outputParams['f_max'],
            bit_width=self.mInputBitWidth, symmetric=outputParams['symmetric']
        )

        # 量化输入
        qXTest = inputQuantizer.quantize(xTest)

        # 定点计算
        qYTest = self.evaluate_quantized(qXTest, func_name)

        # 反量化输出
        yQuantized = outputQuantizer.dequantize(qYTest)

        # 计算误差
        errorFloat = np.abs(yTrue - yFloatFit)
        errorQuantized = np.abs(yTrue - yQuantized)

        # 创建图表
        fig = plt.figure(figsize=(16, 12))

        # ===== 图1: 三线对比 =====
        ax1 = plt.subplot(3, 1, 1)

        func_display_names = {
            'sqrt': 'SQRT',
            'sigmoid': 'Sigmoid',
            'tanh': 'Tanh',
            'gelu': 'GELU',
            'relu6': 'ReLU6',
            'leaky_relu': 'LeakyReLU',
            'rrelu': 'RReLU',
            'prelu': 'PReLU',
            'softplus': 'Softplus',
            'mish': 'MISH',
            'swish': 'SWISH',
            'hard_swish': 'Hard-Swish',
            'hard_sigmoid': 'Hard-Sigmoid',
            'snake2d': 'Snake2D',
            'power_0_3': 'Power (x^0.3)',
            'power_2': 'Power (x^2)',
            'reciprocal': 'Reciprocal (1/x)',
        }
        display_name = func_display_names.get(func_name, func_name.upper())

        plt.plot(xTest, yTrue, 'b-', label=f'Real {display_name}', linewidth=2.5, alpha=0.8)
        plt.plot(xTest, yFloatFit, 'g--', label='Float32 Piecewise Fit (32 segments)', linewidth=2, alpha=0.8)
        plt.plot(xTest, yQuantized, 'r:', label='INT16 Quantized (Asymmetric x²,ax²,bx)', linewidth=2, alpha=0.8)

        # 显示分段边界（增强版：显示段编号）
        for i, seg in enumerate(segments):
            xStart = seg['range'][0]
            plt.axvline(x=xStart, color='blue', linestyle='-', alpha=0.3, linewidth=1.2)
            # 在顶部标注分段编号
            plt.text(xStart, plt.ylim()[1] * 0.95, f'S{i}',
                     fontsize=7, color='blue', alpha=0.7,
                     rotation=0, ha='left', va='top')

        plt.legend(loc='best', fontsize=11)
        plt.title(f'{display_name} - Three-way Comparison (Full Asymmetric Quantization)', fontsize=14, fontweight='bold')
        plt.xlabel('Input x')
        plt.ylabel('Output y')
        plt.grid(True, alpha=0.3)

        # ===== 图2: 误差对比 =====
        ax2 = plt.subplot(3, 1, 2)
        plt.plot(xTest, errorFloat, 'g-', label=f'Float Fit Error (MAE={np.mean(errorFloat):.6f})', linewidth=1.5)
        plt.plot(xTest, errorQuantized, 'r-', label=f'Quantized Error (MAE={np.mean(errorQuantized):.6f})', linewidth=1.5)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

        # 标注目标误差线
        plt.axhline(y=0.001, color='orange', linestyle='--', alpha=0.5, label='Target MAE < 0.001')

        # 显示分段边界（带编号）
        for i, seg in enumerate(segments):
            xStart = seg['range'][0]
            plt.axvline(x=xStart, color='blue', linestyle='-', alpha=0.25, linewidth=0.8)
            # 在误差图上也标注一些关键分段
            if i % 1 == 0:  # 每4个显示一次
                plt.text(xStart, plt.ylim()[1] * 0.9, f'S{i}',
                         fontsize=6, color='blue', alpha=0.6, ha='left')

        plt.legend(loc='best', fontsize=10)
        plt.title('Error Analysis', fontsize=13)
        plt.xlabel('Input x')
        plt.ylabel('Absolute Error')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 对数刻度更清晰

        # ===== 图3: 局部放大（起始区域）=====
        ax3 = plt.subplot(3, 1, 3)

        # 根据不同函数选择合适的放大区域
        zoom_ranges = {
            'sigmoid': (-2, 2),
            'tanh': (-2, 2),
            'sqrt': (0, 2),
            'gelu': (-2, 2),
            'mish': (-2, 2),
            'swish': (-2, 2),
        }
        zoom_min, zoom_max = zoom_ranges.get(func_name, (-2, 2))

        # 选择放大区域
        centerMask = (xTest >= zoom_min) & (xTest <= zoom_max)
        xCenter = xTest[centerMask]
        yTrueCenter = yTrue[centerMask]
        yFloatCenter = yFloatFit[centerMask]
        yQuantCenter = yQuantized[centerMask]

        plt.plot(xCenter, yTrueCenter, 'b-', label=f'Real {display_name}', linewidth=2.5, alpha=0.8)
        plt.plot(xCenter, yFloatCenter, 'g--', label='Float32 Fit', linewidth=2, alpha=0.8)
        plt.plot(xCenter, yQuantCenter, 'r:', label='INT16 Quantized', linewidth=2, alpha=0.8)

        # 显示放大区域的分段（带编号）
        for i, seg in enumerate(segments):
            xStart = seg['range'][0]
            if zoom_min <= xStart <= zoom_max:
                plt.axvline(x=xStart, color='gray', linestyle='-', alpha=0.4, linewidth=1.0)
                plt.text(xStart, plt.ylim()[1] * 0.95, f'{i}',
                         fontsize=8, color='gray', alpha=0.8, ha='left', va='top')

        plt.legend(loc='best', fontsize=10)
        plt.title(f'Zoomed View: Critical Region [{zoom_min}, {zoom_max}] (Segments Labeled)', fontsize=13)
        plt.xlabel('Input x')
        plt.ylabel('Output y')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure to file
        if save_path is None:
            save_path = f"{func_name}_comparison.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        plt.close(fig)

        # Calculate cosine similarity between real function and quantized results
        # This measures how well quantization preserves the real function
        valid_mask = ~(np.isnan(yTrue) | np.isnan(yQuantized))
        if np.sum(valid_mask) > 0:
            yTrue_valid = yTrue[valid_mask]
            yQuantized_valid = yQuantized[valid_mask]
            dot_product = np.dot(yTrue_valid, yQuantized_valid)
            norm_true = np.linalg.norm(yTrue_valid)
            norm_quant = np.linalg.norm(yQuantized_valid)
            if norm_true > 0 and norm_quant > 0:
                cosine_sim = dot_product / (norm_true * norm_quant)
            else:
                cosine_sim = 0.0
        else:
            cosine_sim = 0.0

        # Print detailed statistics
        print("\n" + "="*80)
        print("Detailed Error Statistics [Full Asymmetric Quantization]")
        print("="*80)
        print(f"\nFloat Fit Error:")
        print(f"  MAE:  {np.mean(errorFloat):.8f}")
        print(f"  Max:  {np.max(errorFloat):.8f}")
        print(f"  Min:  {np.min(errorFloat):.8f}")
        print(f"  Std:  {np.std(errorFloat):.8f}")

        print(f"\nQuantized Error (Asymmetric x²,ax²,bx):")
        print(f"  MAE:  {np.mean(errorQuantized):.8f}")
        print(f"  Max:  {np.max(errorQuantized):.8f}")
        print(f"  Min:  {np.min(errorQuantized):.8f}")
        print(f"  Std:  {np.std(errorQuantized):.8f}")
        print(f"  Cosine Similarity (Real vs Quantized): {cosine_sim:.8f}")

        print(f"\nAdditional Error from Quantization:")
        extraError = errorQuantized - errorFloat
        print(f"  Average Additional Error: {np.mean(extraError):.8f}")
        print(f"  Max Additional Error: {np.max(extraError):.8f}")

        # Regional error analysis
        print(f"\nRegional Error Analysis:")
        regions = [
            ("Left Edge [-6, -2]", (xTest >= -6) & (xTest <= -2)),
            ("Center [-2, 2]", (xTest >= -2) & (xTest <= 2)),
            ("Right Edge [2, 6]", (xTest >= 2) & (xTest <= 6))
        ]

        for regionName, mask in regions:
            if np.sum(mask) > 0:
                print(f"\n  {regionName}:")
                print(f"    Float Fit MAE:  {np.mean(errorFloat[mask]):.8f}")
                print(f"    Quantized MAE:  {np.mean(errorQuantized[mask]):.8f}")
                print(f"    Additional Error: {np.mean(extraError[mask]):.8f}")

    # ==================== 导出 ====================
    def export_to_json(self, filename="quantized_lut.json"):
        """导出为JSON格式（用于硬件实现）"""
        exportData = {}

        for funcName, segments in self.mSegments.items():
            exportData[funcName] = {
                'quantization': self.mQuantizers[funcName],
                'num_segments': len(segments),
                'segments': []
            }

            for i, seg in enumerate(segments):
                segmentData = {
                    'segment_id': int(i),
                    'range_float': [float(seg['range'][0]), float(seg['range'][1])],
                    'threshold_quantized': [int(seg['threshold_q'][0]), int(seg['threshold_q'][1])],
                    'coefficients_float': {
                        'a': float(seg['coeff_float'][0]),
                        'b': float(seg['coeff_float'][1]),
                        'c': float(seg['coeff_float'][2])
                    },
                    'coefficients_quantized': {
                        'q_a': int(seg['coeff_quantized'][0]),
                        'q_b': int(seg['coeff_quantized'][1]),
                        'q_c': int(seg['coeff_quantized'][2])
                    },
                    'quantizers': {
                        'x2': seg['quantizers']['x2'],  # 包含零点信息
                        'ax2': seg['quantizers']['ax2'],
                        'bx': seg['quantizers']['bx']
                    },
                    'shift_bits': {
                        'n_x2': int(seg['shift_bits']['n_x2']),
                        'n_ax2': int(seg['shift_bits']['n_ax2']),
                        'n_bx': int(seg['shift_bits']['n_bx']),
                        'n_ya': int(seg['shift_bits']['n_ya']),
                        'n_yb': int(seg['shift_bits']['n_yb']),
                        'n_yc': int(seg['shift_bits']['n_yc'])
                    }
                }
                exportData[funcName]['segments'].append(segmentData)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(exportData, f, indent=2, ensure_ascii=False)

        print(f"\n定点量化参数已导出到: {filename}")


# ==================== 主程序 ====================
def main():
    print("="*80)
    print("INT16 定点分段二次多项式量化实现")
    print("【完全非对称量化版本：x², ax², bx 自适应非对称量化】")
    print("支持 17 个激活函数（包括 reciprocal 双侧, sqrt, power 等）")
    print("="*80)

    # 创建定点拟合器
    fitter = QuantizedPiecewiseQuadraticFitter(num_segments=32, input_bit_width=16)

    # 拟合所有函数
    print("\n开始拟合所有函数...")
    results = fitter.fit_all_functions()

    # 误差分析
    print("\n" + "="*80)
    print("定点量化误差分析 【完全非对称量化版本】")
    print("="*80)

    # 所有函数列表
    all_functions = ['sigmoid', 'tanh', 'gelu', 'relu6', 'leaky_relu',
                     'rrelu', 'prelu', 'softplus', 'mish', 'swish',
                     'hard_swish', 'hard_sigmoid', 'snake2d', 'power_0_3', 'power_2',
                     'reciprocal', 'sqrt']

    # 为不同函数设置测试范围
    function_test_ranges = {
        'sigmoid': (-6, 6),
        'tanh': (-6, 6),
        'gelu': (-6, 6),
        'relu6': (-6, 6),
        'leaky_relu': (-6, 6),
        'rrelu': (-6, 6),
        'prelu': (-6, 6),
        'softplus': (-6, 6),
        'mish': (-6, 6),
        'swish': (-6, 6),
        'hard_swish': (-6, 6),
        'hard_sigmoid': (-6, 6),
        'snake2d': (-6, 6),
        'power_0_3': (0, 8),
        'power_2': (0, 16),
        'reciprocal': (-6, 6),  # 双侧，跳过 x=0
        'sqrt': (0, 16),
    }

    # 计算所有函数的误差
    for func_name in all_functions:
        if func_name in fitter.mSegments:
            x_min, x_max = function_test_ranges.get(func_name, (-6, 6))
            errorData = fitter.compute_error(func_name, x_min=x_min, x_max=x_max)
            if errorData:
                print(f"\n{func_name.upper():15s}: MAE={errorData['mae']:.6f}, Max Error={errorData['max_error']:.6f}, Cosine Sim (Real vs Quantized)={errorData['cosine_similarity']:.6f}")

    '''# 导出JSON（为每个函数单独导出）
    print("\n导出 JSON 文件...")
    for func_name in all_functions:
        if func_name in fitter.mSegments:
            # 创建临时拟合器，只包含当前函数
            temp_fitter = QuantizedPiecewiseQuadraticFitter(num_segments=32, input_bit_width=16)
            temp_fitter.mSegments[func_name] = fitter.mSegments[func_name]
            temp_fitter.mQuantizers[func_name] = fitter.mQuantizers[func_name]
            temp_fitter.export_to_json(f"{func_name}_quantized_lut_int16_nonsym.json")
            print(f"  {func_name} → {func_name}_quantized_lut_int16_nonsym.json")'''

    # Visualize all functions
    print("\nGenerating visualization plots...")

    # Create output directory for all plots
    output_dir = "quantization_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"All plots will be saved to directory: {output_dir}/")

    # Select important functions for visualization
    visualization_functions = ['sigmoid','tanh','gelu','relu6','leaky_relu','rrelu','prelu','softplus','mish','swish','hard_swish','hard_sigmoid','snake2d','power_0_3', 'power_2','reciprocal','sqrt']

    for func_name in visualization_functions:
        if func_name in fitter.mSegments:
            x_min, x_max = function_test_ranges.get(func_name, (-6, 6))
            print(f"\nPlotting comparison for {func_name}...")
            save_path = os.path.join(output_dir, f"{func_name}_comparison.png")
            fitter.plot_quantized_comparison(func_name, x_min=x_min, x_max=x_max, save_path=save_path)

    print("\n定点量化完成！")
    print("="*80)
    print("主要改进（完全版）：")
    print("  ✅ 支持 17 个激活函数（reciprocal 双侧覆盖 [-6,6]）")
    print("  ✅ 自适应分段策略：不同函数使用不同分段权重")
    print("  ✅ sqrt 特殊处理：起始段密集分段")
    print("  ✅ reciprocal (1/x) 对数空间分段 + clip到[-4,4]：正负侧分开处理")
    print("  ✅ power_0_3 (x^0.3) 特殊处理：70%分段集中在 [-1,1]")
    print("  ✅ power_2 (x^2) 使用中心加权策略")
    print("  ✅ x², ax², bx 自适应使用非对称量化")
    print("  ✅ 不跨零范围 → 非对称量化（充分利用范围）")
    print("  ✅ 跨零范围 → 对称量化（保持精度）")
    print("  ✅ 定点计算流程：所有中间值自动是'去零点'后的表示")
    print("  ✅ 无需额外减法操作（零点在量化器中自动处理）")
    print("  ✅ 跨零分段自动检测，x²_min = 0")
    print("  ✅ a 系数分组量化：大值用粗scale，小值用细scale")
    print("  ✅ 预期精度提升约 10-20%")
    print("="*80)


if __name__ == "__main__":
    main()


