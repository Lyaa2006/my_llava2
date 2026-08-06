# 隐式自适应 HiDESC 方案

## 1. 动机

当前 HiDESC 仍然过于接近显式分层：

- `L_align` 绑定在一个固定边界层上。
- `L_struct` / `L_focus` 绑定在一个固定的 description 层上。
- eval 还使用分阶段的固定融合规则。

这会让层级看起来像是手工设计出来的，并且容易受数据集影响。

## 2. 目标重构

把显式的 `deep / middle / shallow` 划分，改成一种**隐式的功能层级**：

- `semantic`
- `task`
- `format`

模型不再被强行规定“哪一层属于哪一种角色”，而是自己学习每一层对不同角色的贡献权重。

## 3. 训练阶段方案

### 3.1 `L_align`

保留当前的对齐目标，但改变层的分配方式：

- 不再绑定到单一固定层；
- 使用一个很小的候选窗口；
- 在候选层上学习软权重。

推荐形式：

```text
L_align^impl = Σ_l β_l(x) · L_align^l
```

其中：

- `β_l(x)` 是学习得到的路由权重；
- `L_align^l` 仍然是原来的 image-text cosine alignment loss。

这样计算量仍然可控，因为只涉及少数几层。

### 3.2 `L_struct` / `L_focus`

这里采用更保守、也更稳的改法：

- `L_focus` 本身保持不变；
- 仍然只在固定的 `description layer` 上计算；
- 不再把多个层的表示先融合再算 `L_focus`；
- 真正改动的是 `description layer` 之前的层，让它们通过可学习权重影响最终到达 `description layer` 的表示。

也就是说：

```text
H_desc = F_≤d(x; α)
L_struct = w_focus · L_focus(H_desc, H_ref) + w_energy · L_energy(H_desc, H_ref)
```

其中：

- `H_desc` 是固定 description layer 的输出；
- `H_ref` 是缓存的 reference description state；
- `F_≤d` 表示 description layer 之前的可学习前置路由/加权路径；
- `α` 是这些前置层的软权重。

关键点：

- reference cache 仍然保持单层、紧凑；
- `L_focus` 不改定义，避免引入额外不稳定因素；
- 自适应性来自 description layer 之前的层，而不是来自 focus 监督本身。

### 3.3 总训练目标

保持全局 loss 形式稳定：

```text
L_total = w_ce · L_ce + L_struct^impl + w_align · L_align^impl
```

变化的不是 loss 家族，而是“哪些层归属于哪些 loss”的方式。

### 3.4 折中可实现版本

为了同时满足“权重随层变化”和“权重跟数据相关”，但又不把计算开销拉太高，推荐把 `lambda_align` 和 `lambda_focus` 写成**连续深度门控**：

```text
λ_k(l, x) = Λ_k · g_k(x) · ρ_k(l) / Σ_j ρ_k(j)
```

其中：

- `k ∈ {align, focus}`
- `g_k(x)` 是样本级门控，由当前样本的一个轻量 summary 向量经过小 MLP 得到
- `ρ_k(l)` 是随层深度单调衰减的先验，例如 `ρ_k(l)=exp(-τ_k · d_l)`
- `d_l` 是归一化层深度，越靠前越小，越靠后越大
- `Λ_k` 控制该类 loss 的整体强度

这样可以保证：

- 同一种 loss 在不同层上的权重不同
- 同一样本和不同样本的权重也不同
- 整体趋势仍然是前面强、后面弱

实现上不需要为每层单独建一个 router，也不需要额外的全层缓存；第一版只要复用当前训练流程里已经能拿到的样本特征，再加一个很小的 gating MLP 即可。若后续要进一步省算力，可以只在少量均匀采样的 probe layers 上计算这条连续权重曲线。

### 3.5 Train 阶段模型草图

下面给出一个按“第一版可实现”思路整理的 train 架构草图。

#### 结构草图

```text
input sample (image, text)
        │
        ├── vision tower / tokenizer
        │
        ├── sample summary extractor
        │       └── pooled image/text summary
        │
        ├── lightweight gating MLP
        │       └── g_align(x), g_focus(x)
        │
        └── backbone transformer
                │
                ├── probe layer l1  ──> hidden state h_l1 ──> L_align^l1
                ├── probe layer l2  ──> hidden state h_l2 ──> L_align^l2
                ├── probe layer l3  ──> hidden state h_l3 ──> L_align^l3
                │          ...
                │
                └── fixed description layer d
                         └── H_desc
                               ├── with reference cache H_ref ──> L_focus / L_energy
                               └── with LM head             ──> L_ce
```

#### 一次前向流程

```text
Step 1. 正常跑一次多模态 backbone 前向，拿到：
        - LM supervision 所需输出
        - 少量 probe layers hidden states
        - fixed description layer hidden state

Step 2. 从当前 sample 提取轻量 summary：
        x_summary = pool(image guide, text guide, 或指定 token hidden)

Step 3. 用小 MLP 计算样本级门控：
        g_align(x), g_focus(x)

Step 4. 结合层深先验 ρ(l)，生成各 probe layers 的 loss 权重：
        λ_align(l, x) = Λ_align · g_align(x) · ρ_align(l)
        λ_focus(l, x) = Λ_focus · g_focus(x) · ρ_focus(l)

Step 5. 在少量 probe layers 上计算：
        L_align^impl = Σ_l λ_align(l, x) · L_align^l

Step 6. 在固定 description layer 上计算：
        L_struct = w_focus · L_focus(H_desc, H_ref) + w_energy · L_energy(H_desc, H_ref)

Step 7. 合成总损失：
        L_total = w_ce · L_ce + L_struct + w_align · L_align^impl
```

#### Train 阶段的实现约束

- `MLP router` 只输出样本级 gate，不直接替代主干前向。
- `L_align` 只在少量 `probe layers` 上计算，不做全层监督。
- `L_focus` / `L_energy` 仍然只在固定 description layer 上计算。
- 因此新增成本主要来自少量 hidden state 读取和小 MLP，而不是额外再跑一条完整 backbone。

## 4. 评测阶段方案

eval 应该和 train 的路由逻辑保持一致。

### 4.1 层路由

对每个样本：

1. 根据当前 hidden states 计算 router score；
2. 给 `description layer` 之前的层分配软权重；
3. 对前置层输出做加权融合，再送入固定的 description layer。

不再保留硬性的 `deep/middle/shallow` 切换。

### 4.2 融合

推荐的 eval 结构：

```text
h_out(x) = Σ_l Σ_r a_{l,r}(x) · [Σ_e b_{l,r,e}(x) · F_{l,r,e}(h_l)]
```

其中：

- `l` 表示层；
- `r` 表示 role；
- `e` 表示 expert。

这样同时支持：

- `role-wise fuse`
- `expert-wise fuse`

并且整个架构仍然是自适应的。

### 4.3 尖锐程度

不同 role 可以使用不同温度：

- `semantic`：更平滑的融合；
- `task`：更选择性的融合；
- `format`：最尖锐的融合。

因此，整体架构是统一的，但不同 role 的路由行为不同。

### 4.4 自适应递进后验更新

如果 eval 直接写成“每一层都对所有 role / expert 重新做一次全局 softmax，再线性加权融合”，那么它虽然形式统一，但本质上仍然过于接近全局线性 attention：

- 每层都在重复同一种扁平混合；
- 路由缺少“上一层决策如何影响下一层”的条件依赖；
- 计算量也会随着层数线性放大。

因此，更推荐把自适应机制写成一种**轻量的递进后验更新**，而不是写成全量 attention。

核心思想：

- 入口先计算一次 coarse routing，得到初始 `role posterior` 和 `intra-role expert posterior`；
- 后续层不再从头对全体 experts 做全局竞争；
- 而是只对少量候选 role / experts 的后验分布做逐层修正。

推荐形式：

```text
log p_l(r | x) = log p_{l-1}(r | x) + Δ_l^role(r; x, h_l)
log p_l(e | r, x) = log p_{l-1}(e | r, x) + Δ_l^intra(e; r, x, h_l)
```

其中：

- `p_l(r | x)` 表示第 `l` 层时样本属于各个 role 的后验；
- `p_l(e | r, x)` 表示在 role `r` 内部的 expert 后验；
- `h_l` 是当前层的轻量 summary state；
- `Δ_l^role` 和 `Δ_l^intra` 是小型 scorer 或 gating MLP 给出的增量修正项。

这样做的关键好处是：

- 路由是**递进式**的：上一层的判断会影响下一层；
- 路由是**自适应**的：当前层 hidden state 可以对已有判断做纠偏；
- 路由不是标准 token-to-token attention，而是 sample-level posterior refinement。

最终每层实际使用的权重可以写成：

```text
w_l(r, e | x) = p_l(r | x) · p_l(e | r, x)
```

这时“早层更宽、中层更细、后层更尖”的行为，不需要通过硬编码 `deep/middle/shallow` 来规定，而是由后验分布自身逐层收缩得到。

为了避免算力显著上升，第一版实现建议增加三个约束：

- 只在输入阶段保留 `top-M roles`；
- 每个 role 内只保留 `top-K experts`；
- 不必每层都更新 posterior，只在少量 `probe layers` 上更新，其余层沿用最近一次结果。

于是额外开销主要变成：

- 少量层上的 summary pooling；
- 少量候选上的小 MLP / scorer 打分；
- 少量 posterior 的归一化更新。

而不是：

- 对所有层、所有 role、所有 experts 做全量重打分和全量融合。

因此，这套机制既保留了递进层级协同的表达力，也能把实现复杂度和推理成本控制在一个较稳的范围内。

### 4.5 Eval 阶段模型草图

下面给出一个按“可实现的轻量递进后验更新”整理的 eval 架构草图。

#### 结构草图

```text
input sample (image, text)
        │
        ├── vision tower / tokenizer
        │
        ├── sample summary extractor
        │       └── pooled image/text summary
        │
        ├── coarse routing initializer
        │       ├── sample-to-role scoring
        │       ├── sample-to-anchor scoring
        │       └── initial posterior:
        │              p_0(r | x), p_0(e | r, x)
        │
        └── backbone transformer + LoRA experts
                │
                ├── probe layer l1
                │      ├── summary h_l1
                │      ├── posterior update MLP / scorer
                │      └── updated weights w_l1(r, e | x)
                │
                ├── normal layers
                │      └── reuse nearest posterior
                │
                ├── probe layer l2
                │      ├── summary h_l2
                │      ├── posterior update MLP / scorer
                │      └── updated weights w_l2(r, e | x)
                │
                └── final LM head
                         └── output logits / generation
```

#### 一次前向流程

```text
Step 1. 从输入样本提取 image/text summary。

Step 2. 用已有历史 memory 做 coarse initialization：
        - role prototypes -> 初始 role posterior
        - task anchors    -> 初始 intra-role expert posterior

Step 3. 只保留少量候选：
        - top-M roles
        - top-K experts per role

Step 4. 进入 backbone 推理。
        在每个 probe layer：
        - 从当前 hidden state 提取轻量 summary h_l
        - 用小 scorer / MLP 计算 posterior 修正项
        - 更新：
              log p_l(r | x)     = log p_{l-1}(r | x)     + Δ_l^role
              log p_l(e | r, x)  = log p_{l-1}(e | r, x)  + Δ_l^intra
        - 得到当前层 expert 混合权重：
              w_l(r, e | x) = p_l(r | x) · p_l(e | r, x)

Step 5. 在非 probe layers：
        - 不重新全量打分
        - 直接沿用最近一次 posterior，或做轻量插值

Step 6. 用当前层权重控制对应 LoRA experts 的融合，继续前向。

Step 7. 在后期层中，posterior 自然变尖锐：
        - role 分布逐渐收缩
        - role 内 expert 分布逐渐收缩
        - 最终收敛到少量高置信 experts
```

#### Eval 阶段的实现约束

- 不做“每层对所有 experts 全量重算”的全局 attention 式路由。
- 后验更新只发生在少量 `probe layers`。
- 只在 `top-M roles` 与 `top-K experts` 上维护 posterior。
- `MLP router` / scorer 只负责增量修正，不替代初始 similarity-based prior。

## 5. 消融故事

后续可以通过消融来论证最终 cache layer 的选择：

- 固定层 vs 小窗口；
- 1 层 vs 3 层；
- 性能 vs 计算成本。

如果小窗口和固定层效果接近，那么选择单一 canonical layer 就是一个效率最优的近似。
