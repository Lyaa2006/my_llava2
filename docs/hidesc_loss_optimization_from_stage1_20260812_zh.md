# HiDESC Loss 优化方案（基于 Stage 1 边界实验）

日期：2026 年 8 月 12 日

## 1. 问题来源

前置实验得到的 `b1/b2` 不是任意切分点，而是不同模型内部阶段转移的经验窗口：

- `b1` 对应 reasoning-dominant 到 objective-dominant 的过渡带
- `b2` 对应后层 style working core

因此，这组边界不应被理解为“固定常数”，而应被理解为“loss 作用区间”的模型相关超参数。

## 2. 设计原则

本轮方法描述聚焦训练侧 loss 设计，不改以下部分：

- 分层架构
- expert 结构
- anchor / prototype 机制
- 路由主干逻辑

也就是说，`b1/b2` 不进入路由分段，而进入监督分配。

eval 侧是否联动调整，不作为本方法的固定前提。

## 3. 从前置实验到实现的映射

前置实验的意义是提供“阶段性先验”：

- `b1` 告诉我们：哪一段更适合施加跨模态对齐约束
- `b2` 告诉我们：哪一段更适合施加描述结构约束

因此对应到 HiDESC，可以采用：

- `B1` band 主要承载 `align loss`
- `B2` band 主要承载 `struct loss`
- `CE` 仍保持全局主监督，不做分层改造

这不是把实验结果生搬硬套到模型里，而是把“边界分析”转译成“监督位置选择”。

更具体地说，这里的逻辑关系如下。

### 3.1 为什么 `b1` 对应 align

`b1` 来自 reasoning-dominant 到 objective-dominant 的过渡分析。这个过渡带的核心特征，不是语言风格已经稳定，也不是最终任务输出已经充分定型，而是模型正在把前层较强的 reasoning / exploration 状态，逐步压缩到更可判别的 objective-oriented 表征上。

在这个阶段，最需要稳定的是“图像信息与文本信息是否已经在同一语义子空间内形成可用对齐”。因此，对 `B1` 施加跨模态 align，有两个作用：

- 约束中层过渡不要过早偏向纯文本内部组织，而忽略视觉 grounding
- 帮助 objective-dominant 中层在进入后续 task/style specialization 之前，先具备更稳定的 image-text shared representation

因此，`b1` 不是因为“它在中间”才对应 align，而是因为前置实验表明这里正处于表征目标从 reasoning 向 objective 收敛的窗口，适合用跨模态一致性约束来稳定过渡。

### 3.2 为什么 `b2` 对应 struct

`b2` 来自后层 style working core 分析。这个区间反映的不是简单的分类判别边界，而是后层在生成式表达、描述组织和输出风格稳定化方面开始占主导。

HiDESC 现有 `struct loss` 的两部分：

- `focus loss`：约束描述 token 的变化集中在关键位置
- `energy loss`：约束描述隐藏状态变化强度不过度扩散

这两类约束都更接近“后层输出结构塑形”，而不是“中层跨模态融合”。因此，把 `struct loss` 放到 `B2` 上是因为前置实验已经告诉我们：`B2` 更像 late-stage description/style working zone，在这里施加结构约束具有更好的语义匹配性。

因此，`b2` 也不是被机械地拿来替换原始 boundary，而是被解释为“description structure regularization 更应集中的 late working band”。

## 4. 具体实现思路

### 4.1 B1 band：align loss 的区间化

对 `B1 = [b1_low, b1_high]` 内每一层都计算 align loss，再按自适应权重聚合：

```text
L_align^B1 = sum_{l in B1} alpha_l(t) * L_align^(l)
```

其中 `alpha_l(t)` 满足：

- 非负
- 区间内归一化
- 位置上整体从 `b1_low -> b1_high` 单调增强
- 允许根据当前 batch 的对齐难度做轻微自适应调整

这里建议把 `L_align^(l)` 直接定义为第 `l` 层的 image-text pooled cosine loss：

```text
L_align^(l) = 1 - cos( pool_img(h_l), pool_txt(h_l) )
```

其中 `pool_img` 和 `pool_txt` 复用当前 HiDESC 的 image/text token mask 平均池化方式。

### 4.1.1 B1 难度自适应的具体实现

令 `B1` 内第 `l` 层的即时难度为：

```text
d_l^B1(t) = detach( L_align^(l) )
```

再做一个轻量 EMA：

```text
m_l^B1(t) = gamma * m_l^B1(t-1) + (1-gamma) * d_l^B1(t)
```

其中 `gamma` 可取 `0.8 ~ 0.95`。

为了避免不同层 loss 尺度不可比，再做 band 内归一化：

```text
z_l^B1(t) = ( m_l^B1(t) - mean_B1(m^B1) ) / ( std_B1(m^B1) + eps )
```

位置先验采用单调递增 ramp：

```text
p_l^B1 = eps_p + (1-eps_p) * (l - b1_low) / max(1, b1_high - b1_low)
```

最终 band 权重定义为：

```text
alpha_l(t) = softmax_B1( log p_l^B1 + eta_b1 * z_l^B1(t) )
```

其中：

- `eta_b1` 控制自适应强度，建议初值 `0.3 ~ 0.8`
- `detach` 表示难度只参与权重分配，不反传二阶梯度
- `softmax_B1` 表示只在 `B1` 内归一化

这样定义后：

- 越靠近 `b1_high` 的层，默认会获得更高先验权重
- 如果 band 内某层当前对齐更差，它会临时获得更大权重
- 但这种放大不会完全打破 `B1` 的整体过渡趋势

### 4.2 B2 band：struct loss 的区间化

为了尽量不改 cache，`B2` 仍使用单层 description cache，但不再只监督一个层点，而是让 `B2` 内多层共享同一个 late anchor：

```text
L_struct^B2 = sum_{l in B2} beta_l(t) * [ w_f * L_focus^(l) + w_e * L_energy^(l) ]
```

其中：

- cache 仍然只保存 `b2_high` 对应的 reference
- `beta_l(t)` 在 `B2` 内自适应分配
- 越靠近 `b2_high`，先验权重越大
- band 内更“难”的层可获得更高临时权重

这里的 `L_focus^(l)` 和 `L_energy^(l)` 不是“每层各有自己的 cache”，而是：

- 当前层使用 `h_l`
- reference 统一使用 `b2_high` 对应的 cached hidden `r_ref`

即：

```text
L_focus^(l)  = FocusCross( h_l, r_ref )
L_energy^(l) = EnergyCross( h_l, r_ref )
```

这等价于让 `B2` 内多层共享同一个 late anchor，同时保持 cache 仍然是单层。

### 4.2.1 B2 难度自适应的具体实现

定义 `B2` 内第 `l` 层的即时结构难度为：

```text
d_l^B2(t) = detach( L_focus^(l) + rho * L_energy^(l) )
```

其中 `rho` 用于把 `energy` 缩放到与 `focus` 同一量级，建议初值可从 `1.0` 开始，再结合日志调节。

同样做 EMA：

```text
m_l^B2(t) = gamma * m_l^B2(t-1) + (1-gamma) * d_l^B2(t)
```

再在 `B2` 内标准化：

```text
z_l^B2(t) = ( m_l^B2(t) - mean_B2(m^B2) ) / ( std_B2(m^B2) + eps )
```

位置先验也采用单调递增形式：

```text
p_l^B2 = eps_p + (1-eps_p) * (l - b2_low) / max(1, b2_high - b2_low)
```

最终：

```text
beta_l(t) = softmax_B2( log p_l^B2 + eta_b2 * z_l^B2(t) )
```

其中 `eta_b2` 建议小于 `eta_b1`，例如 `0.2 ~ 0.6`，避免后层结构约束抖动过大。

这样定义后：

- `b2_high` 附近的层天然更接近 late working core
- 如果 `B2` 内某层当前结构偏移更大，它会获得更高临时 struct 权重
- 但 band 外层不参与自适应，仍保持原设定不变

## 5. 总损失形式

最终总损失可写成：

```text
L_total = w_ce * L_ce + w_align * L_align^B1 + w_struct * L_struct^B2
```

其中 `w_struct` 内部再拆成 `w_f` 和 `w_e`。

如果写得更展开，可写为：

```text
L_total
= w_ce * L_ce
+ w_align * sum_{l in B1} alpha_l(t) * L_align^(l)
+ sum_{l in B2} beta_l(t) * [ w_f * L_focus^(l) + w_e * L_energy^(l) ]
```

## 6. 为什么这样不会被质疑为硬套

原因有三点：

1. `b1/b2` 来自前置实验的多 seed 边界统计，不是人工拍脑袋设定。
2. 这里没有把边界当成结构常数，而是当成监督窗口，所以模型结构本身没有被强行改写。
3. 权重不是死板一刀切，而是“位置先验 + 难度自适应”，保留了阶段过渡的连续性。
4. 自适应项使用 `detach + EMA`，其角色只是调整 band 内监督分配，而不是重新定义主训练目标，因此工程上是可解释且稳定的。

## 7. 计算与代价

这套方案不会显著增加主干 FLOPs，因为：

- 不增加 expert 前向
- 不增加 anchor 机制
- 只是在 band 内多算若干层的 loss 聚合

真正需要注意的是：

- `B1/B2` 不宜太宽
- `B2` 继续复用单层 cache 时，若 anchor 层变化，需要重新生成 cache

## 8. 推荐写法

论文或方案描述中，建议用下面的表述：

> 基于 Stage 1 边界实验得到的 `b1/b2` 区间，我们将其作为 loss 的阶段先验，而非固定架构分界；其中 `B1` 负责对齐约束，`B2` 负责结构约束，二者均采用区间内自适应加权，从而在保持前中后过渡趋势的同时，避免对模型层划分作硬编码。
