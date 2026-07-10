# HiDeCL 训练阶段设计

## 1. 文档目的

本文定义 HiDeCL 在训练阶段的完整设计。该设计只讨论当前 task expert 的训练，不讨论推理阶段的 role-aware routing。

本文的核心结论是：

- `description` 不参与路由决策；
- `description` 的职责是约束当前 task LoRA 学到的增量功能；
- 训练阶段的约束对象不是 LoRA 裸参数，也不是当前 hidden state 的绝对值，而是相对上一阶段冻结模型的 `hidden delta / function delta`；
- 当前实现中的 `description utility loss` 不再保留原形式，而改为更直接的跨分支 anti-drift 约束。


## 2. 设计前提

### 2.1 路由与训练解耦

我们当前已经有较明确的推理路由机制，因此训练阶段不再让 `description` 参与路由。训练阶段只负责把“当前 task expert 应该学到什么”训练清楚；推理阶段再由现有路由机制决定“什么时候调用它”。

### 2.2 当前 task LoRA 在训练时 100% 生效

在训练阶段，当前 task expert 是明确挂载并参与前向的。因此训练时可以直接把：

- 上一阶段冻结模型 `M_{t-1}`
- 当前训练模型 `M_t`

之间的差异解释为“当前 task LoRA 带来的增量功能”。

这和推理阶段的 sample-dependent activation 不是同一个问题，不需要混在一起考虑。

### 2.3 设计目标

训练阶段希望同时满足四个目标：

1. 当前 task 的标准答题能力必须优先学好；
2. `description` 应该塑造当前 expert 的视觉语义更新方向；
3. `description` 分支学到的语义更新必须真正进入 answer branch，而不是自成一套表示；
4. `description` 的更新应聚焦在语义关键 token 上，而不是被 prompt 模板和无关 token 主导。


## 3. 为什么不继续使用当前的 `align + utility`

### 3.1 当前 `align_loss` 的问题

当前实现中的 `align_loss` 本质上是：

`L_align = ||h_desc^cur - h_desc^ref||^2`

它约束的是当前 description hidden state 的绝对值，而不是当前 task LoRA 额外学到了什么。

这会带来三个问题：

1. 它约束的是最终状态，不是当前 LoRA 的增量功能；
2. 它容易把当前 task 应该学到的 task-specific 偏移也当成误差；
3. 它对 prompt 模板、padding、token 位置较敏感，容易产生高波动。

### 3.2 当前 `utility_loss` 的问题

当前实现中的 `utility_loss` 试图让 `description` 对答案生成有用，但非 MPT 路径实际上是把 description states 当成 prefix，再做一遍 text-only answer CE。

这会导致：

1. `utility_loss` 很容易演化为第二个答题目标；
2. 它和主 `CE loss` 的优化方向重叠，但噪声更大；
3. 它并没有直接刻画 text-image anti-drift，而是在间接逼 description branch 参与生成。

因此，我们保留 “anti-drift” 的目标，但不保留当前 `utility_loss` 的具体实现形式。


## 4. 新设计的核心思想

### 4.1 `description` 的角色

`description` 在新设计中的角色不是：

- 预测路由；
- 替 answer branch 再做一遍答题；
- 直接约束 LoRA 参数数值。

`description` 的角色是：

- 提供当前样本的视觉语义更新参照；
- 约束当前 task LoRA 相对 `M_{t-1}` 产生的增量方向；
- 保证该增量方向会被 answer branch 真正使用。

### 4.2 训练时真正要约束的对象

我们约束的是：

- `description branch` 上的语义增量；
- `answer branch` 上的功能增量；
- 两者之间的一致性。

这比直接约束 `h_cur` 或直接约束 LoRA 参数都更合理。


## 5. 记号定义与三段式对应

设：

- `M_{t-1}`：上一阶段冻结模型；
- `M_t`：当前训练模型，其中仅当前 task expert 可训练；
- `x`：当前样本；
- `d(x)`：该样本的 description prompt 输入；
- `qa(x)`：该样本的原始问答输入；
- `l_e`：`early` 段的代表层；
- `l_m`：`middle` 段的代表层；
- `late`：最后若干层，不挂 description 辅助损失；
- `K`：description 中的语义关键 token 集合；
- `A`：answer branch 中参与监督的 answer token 集合；
- `Pool(., mask)`：masked mean pooling。

为控制复杂度，训练阶段不做逐层 loss，而是只选两个代表层：

- `early` 代表层 `l_e`
- `middle` 代表层 `l_m`

`late` 段不选代表层，因为它只保留 `CE`。

对任一层 `l in {l_e, l_m}`，定义四个 pooled representation：

- `z_desc_prev = Pool(h_l(M_{t-1}, d(x)), K)`
- `z_desc_cur = Pool(h_l(M_t, d(x)), K)`
- `z_ans_prev = Pool(h_l(M_{t-1}, qa(x)), A)`
- `z_ans_cur = Pool(h_l(M_t, qa(x)), A)`

进一步定义训练时的两类增量：

- `Delta_desc = z_desc_cur - stopgrad(z_desc_prev)`
- `Delta_ans = z_ans_cur - stopgrad(z_ans_prev)`

其中：

- `Delta_desc` 表示当前 task LoRA 在 description branch 上额外引入的语义更新；
- `Delta_ans` 表示当前 task LoRA 在 answer branch 上额外引入的答题功能更新。

训练时将分别得到：

- `Delta_desc^e`, `Delta_ans^e`
- `Delta_desc^m`, `Delta_ans^m`

分别对应 `early` 与 `middle` 的增量约束。


## 6. 新训练目标

### 6.1 标准监督损失

主损失仍然是标准自回归交叉熵：

`L_ce = CE(M_t(qa(x)), y)`

这是训练阶段的主目标，优先级最高。

### 6.2 `early` 段跨分支 anti-drift 损失

为了防止 text 和 image 之间出现漂移，我们不再使用当前的 prefix-style utility CE，而是直接约束：

`description` 分支学到的增量方向，必须与 `answer` 分支真正使用的增量方向一致。

在 `early` 代表层上定义：

`L_bridge^e = 1 - cos(Delta_desc^e, Delta_ans^e)`

该项的含义是：

- 如果当前 task LoRA 在 `early` 段学到了一种视觉语义更新；
- 那么 answer branch 在 `early` 段也必须沿着近似相同的方向发生变化；
- 否则说明 description 学到的东西没有真正进入答题功能，出现了跨模态漂移或分支脱耦。

### 6.3 `early` 段语义聚焦损失

为了让 description 更新集中在真正有意义的 token 上，而不是被模板词、系统提示词、无关位置主导，我们引入语义聚焦损失。

对 `early` 代表层的 description branch token-level delta 定义：

- `delta_i = h_l^cur(i) - stopgrad(h_l^prev(i))`

分别在 key tokens 和 non-key tokens 上做范数统计：

- `E_key = mean_{i in K} ||delta_i||_2`
- `E_nonkey = mean_{i not in K} ||delta_i||_2`

定义：

`L_focus = max(0, E_nonkey - alpha * E_key)`

其中 `alpha` 为小于 1 的常数，例如 `0.3 ~ 0.5`。

该项的作用是：

- 允许关键语义 token 上有明确更新；
- 压制非关键 token 上无意义的大幅扰动；
- 降低 description prompt 模板对训练的污染。

### 6.4 `middle` 段弱一致性损失

`middle` 段的职责不是继续强推视觉语义，而是保证 `early` 学到的更新没有在中层推理过程中完全消失。

因此这里只使用更弱的 bridge loss：

`L_bridge^m = 1 - cos(Delta_desc^m, Delta_ans^m)`

并建议：

- 不在 `middle` 段使用 `L_focus`
- `middle` 段权重显著小于 `early`

### 6.5 增量幅度稳定项

训练日志表明 description 相关损失波动较大，因此需要对增量能量增加一个弱约束，避免 description 分支的更新幅度失控。

在 `early` 代表层上定义：

`L_energy = max(0, ||Delta_desc^e||_2 - m)^2`

其中 `m` 为允许的增量半径阈值。

该项不是要求 `Delta_desc` 越小越好，而是避免其出现异常爆发式抖动。

### 6.6 总损失

最终训练目标定义为：

`L_total = L_ce + lambda_e * L_bridge^e + lambda_f * L_focus + lambda_m * L_bridge^m + lambda_g * L_energy`

推荐原则：

- `L_ce` 始终为主项；
- `lambda_e` 取小到中等权重；
- `lambda_f` 取小权重；
- `lambda_m` 小于 `lambda_e`；
- `lambda_g` 取很小权重，仅作为稳定器；
- `late` 段不挂任何 auxiliary loss，因此天然是 `CE only`。


## 7. 训练流程与前向次数

### 7.1 前向流程

对每个训练样本，训练流程如下：

1. 使用当前模型 `M_t` 计算标准 answer branch 前向，得到 logits、`L_ce`，并抓取 `l_e/l_m` 的 hidden；
2. 使用当前模型 `M_t` 计算 description branch 前向，并抓取 `l_e/l_m` 的 hidden；
3. 使用冻结模型 `M_{t-1}` 在 `torch.no_grad()` 下计算同一样本的 answer branch 前向，并抓取 `l_e/l_m` 的 hidden；
4. 使用冻结模型 `M_{t-1}` 在 `torch.no_grad()` 下计算同一样本的 description branch 前向，并抓取 `l_e/l_m` 的 hidden；
5. 基于上述 hidden 计算 `Delta_desc^e/Delta_ans^e/Delta_desc^m/Delta_ans^m`；
6. 计算 `L_bridge^e`、`L_focus`、`L_bridge^m`、`L_energy`；
7. 汇总为 `L_total`，仅对当前 task expert 回传梯度。

### 7.2 前向次数

若完全在线计算，则每 step 的前向次数为：

- 当前可训练模型：2 次
- 冻结参考模型：2 次 `no_grad`
- 总计：4 次

这是推荐的基线实现。

若需要进一步降成本，可以采用半缓存方案：

- 在线计算 `M_t` 的 answer + description：2 次
- 离线缓存 `M_{t-1}` 的 description 表示
- 在线计算 `M_{t-1}` 的 answer：1 次 `no_grad`

则总计可降为 3 次前向。

### 7.3 为什么不需要单独“关闭 late 梯度”

只要 `L_bridge^e` 与 `L_bridge^m` 分别挂在 `l_e` 与 `l_m` 的 hidden 上，那么它们的梯度天然只会回传到这些层及其之前的层。

因此：

- `late` 段不会收到 auxiliary loss 的梯度；
- `late` 段只会收到最终 `CE` 的梯度；
- 不需要额外写“关闭 late 梯度”的手工逻辑。

### 7.4 关键点

该流程中：

- `M_{t-1}` 完全冻结，只作为 baseline；
- 不修改当前推理路由设计；
- 不要求任何历史 expert 在训练时参与协同激活；
- 只训练当前 task expert，使其相对 `M_{t-1}` 学到语义可控的功能增量；
- 不再保留当前的 `description utility forward`。


## 8. 面向 Coding Agent 的实现规格

### 8.1 推荐新增的训练参数

建议在 [train_MOE.py](/mnt/lyaa/MCITlib/LLaVA/HiDeCL/llava/train/train_MOE.py:149) 的 `TrainingArguments` 中新增：

- `description_early_layer: int`
- `description_middle_layer: int`
- `description_late_ce_only_start: int`
- `description_bridge_early_weight: float`
- `description_bridge_middle_weight: float`
- `description_focus_weight: float`
- `description_energy_weight: float`
- `description_energy_margin: float`
- `description_loss_warmup_ratio: float`
- `reference_answer_cache_dir: Optional[str] = None`

其中：

- `description_hidden_layer` 将不再作为唯一 loss 层；
- 可以保留它作为兼容旧配置的 fallback；
- 新实现优先使用 `early/middle` 两个指定层。

### 8.2 对 `llava_trainer.py` 的改造原则

当前 [llava_trainer.py](/mnt/lyaa/MCITlib/LLaVA/HiDeCL/llava/train/llava_trainer.py:485) 的 `compute_loss` 结构是：

1. answer forward
2. description forward
3. utility forward
4. absolute align loss

建议改成：

1. `M_t` answer forward
2. `M_t` description forward
3. `M_{t-1}` answer forward `no_grad`
4. `M_{t-1}` description forward `no_grad`
5. 计算 stage-aware auxiliary losses

即：

- 删除 `_compute_description_utility_loss`
- 删除当前 token-level absolute `description_align_loss`
- 保留 `_masked_mean_pool`
- 扩展 `_extract_description_states` 为“可抽取多个层”

### 8.3 抓层方式

不建议继续依赖 `output_hidden_states=True` 返回全层 hidden，因为这会额外保存所有层的激活，显存和通信都更重。

推荐实现方式：

1. 在 transformer block `l_e` 和 `l_m` 上注册 forward hook；
2. 在 answer forward 时抓取这两层输出；
3. 在 description forward 时抓取这两层输出；
4. 前向结束后立刻移除或清空本次 step 的缓存。

推荐 helper 形式：

- `_capture_stage_hidden_states(model, target_layers)`
- `_run_with_stage_hooks(model, forward_kwargs, target_layers, no_grad=False)`

返回结构建议为：

- `{"early": tensor, "middle": tensor}`

### 8.4 token mask 的复用

当前实现中已有：

- `description_key_mask`
- `labels.ne(IGNORE_INDEX)` 形成的 answer mask

这两者可以直接复用：

- `K` 继续使用 `description_key_mask`
- `A` 继续使用 answer token mask

因此 Coding Agent 不需要重写数据集侧的 key token 逻辑，只需把 mask 接到新的 stage loss 计算中。

### 8.5 参考模型的组织方式

推荐在 `Trainer` 初始化时额外持有一份冻结参考模型：

- 从 `previous_task_model_path` 加载；
- `eval()`；
- `requires_grad_(False)`；
- 放在与训练模型相同 device；
- 所有参考前向均包裹在 `torch.no_grad()` 中。

推荐 helper：

- `_build_reference_model_once()`
- `_run_reference_forward(...)`

不要在每个 step 内反复重新加载参考模型。

### 8.6 缓存策略

缓存分三档：

1. 最简实现：完全不缓存 answer 侧参考表示，每步在线跑 2 次 `no_grad`
2. 推荐实现：缓存 `M_{t-1}` 的 description pooled representation，answer 侧在线算
3. 激进实现：同时缓存 `M_{t-1}` 的 description / answer pooled representation

推荐先实现第 2 档，因为：

- description prompt 固定，缓存收益高；
- answer 侧受样本截断和 token mask 影响更复杂，先在线算更稳。

### 8.7 损失计算 helper 建议

建议新增以下 helper：

- `_pool_description_stage(hidden, key_mask)`
- `_pool_answer_stage(hidden, answer_mask)`
- `_compute_stage_delta(cur, ref)`
- `_compute_bridge_loss(delta_desc, delta_ans)`
- `_compute_focus_loss(cur_desc_hidden, ref_desc_hidden, key_mask)`
- `_compute_energy_loss(delta_desc, margin)`
- `_get_aux_loss_scale()` 用于 warmup 缩放

其中 `_get_aux_loss_scale()` 建议按 global step 做线性 warmup：

- warmup 前仅 `CE`
- warmup 后逐步放大 auxiliary loss

### 8.8 `compute_loss` 伪代码

```python
def compute_loss(...):
    answer_cur = run_answer_forward_trainable(...)
    desc_cur = run_description_forward_trainable(...)

    with torch.no_grad():
        answer_ref = run_answer_forward_reference(...)
        desc_ref = run_description_forward_reference(...)

    delta_desc_e = pool(desc_cur["early"]) - pool(desc_ref["early"])
    delta_ans_e = pool(answer_cur["early"]) - pool(answer_ref["early"])
    delta_desc_m = pool(desc_cur["middle"]) - pool(desc_ref["middle"])
    delta_ans_m = pool(answer_cur["middle"]) - pool(answer_ref["middle"])

    bridge_e = bridge(delta_desc_e, delta_ans_e)
    focus = focus_loss(desc_cur["early"], desc_ref["early"], description_key_mask)
    bridge_m = bridge(delta_desc_m, delta_ans_m)
    energy = energy_loss(delta_desc_e)

    aux_scale = get_aux_loss_scale()
    total = (
        ce_loss
        + aux_scale * w_e * bridge_e
        + aux_scale * w_f * focus
        + aux_scale * w_m * bridge_m
        + aux_scale * w_g * energy
    )
    return total
```

### 8.9 与当前代码的直接映射

Coding Agent 在当前代码中应优先改这些位置：

1. [train_MOE.py](/mnt/lyaa/MCITlib/LLaVA/HiDeCL/llava/train/train_MOE.py:149)
   添加新的 `TrainingArguments`
2. [llava_trainer.py](/mnt/lyaa/MCITlib/LLaVA/HiDeCL/llava/train/llava_trainer.py:316)
   将单层 `_extract_description_states` 扩展成多层 stage 抽取
3. [llava_trainer.py](/mnt/lyaa/MCITlib/LLaVA/HiDeCL/llava/train/llava_trainer.py:396)
   删除或废弃 `_compute_description_utility_loss`
4. [llava_trainer.py](/mnt/lyaa/MCITlib/LLaVA/HiDeCL/llava/train/llava_trainer.py:485)
   重写 `compute_loss`
5. [Taskn.sh](/mnt/lyaa/MCITlib/LLaVA/HiDeCL/scripts/MCITlib/Train/Taskn.sh:247)
   增加新的 CLI 参数透传

### 8.10 不建议的实现

不建议：

1. 为 `early/middle/late` 各跑一遍独立前向
2. 用每一层都计算 auxiliary loss 的方式实现三段式
3. 保留当前 `utility_loss` 再叠加新 loss
4. 让 `late` 层继续接收 description 相关 loss
5. 依赖全量 `output_hidden_states=True` 长期开启全层缓存


## 9. 与当前实现相比需要替换的部分

### 8.1 删除或替换的部分

当前实现中以下两部分应被替换：

1. 基于 token-level cached description states 的绝对 MSE 对齐；
2. 基于 description prefix 的 text-only utility CE。

### 8.2 保留的部分

当前实现中以下部分仍可保留：

1. description prompt 的构造方式；
2. description key token mask 的构造逻辑；
3. 选定 hidden layer 的机制；
4. 标准 answer CE 的训练主流程。

### 8.3 缓存策略调整

原方案缓存的是完整 description token hidden states。新方案更建议缓存以下信息：

1. `M_{t-1}` 的 description pooled representation；
2. 如果显存或计算允许，可在线计算 `M_{t-1}` 的 answer pooled representation；
3. 若需要进一步减负，可考虑离线缓存 `z_desc_prev`，仅在线计算 `z_ans_prev`。

这样可以减少对完整 token-level absolute state matching 的依赖。


## 10. 为什么这版更适合当前问题

结合当前训练现象：

- `standard_ce` 很快下降；
- 其余 description loss 波动较大；

说明旧设计里 `description` 更像与主任务并列竞争的第二目标，而不是稳定的辅助约束。

新设计更合理的原因在于：

1. `L_ce` 仍然明确承担“学会当前任务”的职责；
2. `L_bridge` 直接针对 anti-drift，而不是间接让 description 再做一次生成；
3. `L_focus` 抑制模板噪声；
4. `L_energy` 用于稳定增量尺度；
5. 全部附加损失都围绕“当前 task LoRA 的增量功能”展开，而不是围绕 hidden state 绝对值展开。


## 11. 推荐训练策略

### 10.1 权重启用策略

为避免训练前期附加损失过早干扰主任务，建议采用 warmup 式启用策略：

1. 前 `5% ~ 10%` steps 仅训练 `L_ce`；
2. 随后线性增大 `lambda_bridge`；
3. `lambda_focus` 和 `lambda_energy` 从更小值开始。

### 10.2 日志监控项

建议训练时额外记录：

1. `||Delta_desc||_2`
2. `||Delta_ans||_2`
3. `cos(Delta_desc, Delta_ans)`
4. `E_key`
5. `E_nonkey`

这些指标比原先单纯记录 `align_loss` 和 `utility_loss` 更能解释训练是否稳定。


## 12. 最终结论

HiDeCL 的训练阶段应采用如下原则：

1. `description` 不负责路由；
2. `description` 负责塑造当前 task LoRA 相对上一阶段模型的增量方向；
3. anti-drift 的核心不是让 description 参与生成，而是让 `description delta` 与 `answer delta` 保持一致；
4. 训练目标应从 “absolute hidden alignment + utility CE” 转向 “delta alignment + cross-branch consistency + semantic focus”。

这套设计更符合当前系统中“训练阶段当前 task LoRA 100% 生效、推理阶段再做样本路由”的真实设定。
