# HiDESC 方法与前置实验总说明

日期：2026-08-13

更新：2026-08-16

本次更新重点同步两类最新改动：

1. `text` 侧 prototype 不再直接聚合 raw pooled text embedding，而是改成 `text activation index -> EMA anchor -> offline contrastive bank`。
2. `eval` 侧不再用旧版 “late top-k task sparse routing” 作为主描述，而是改成 `role prior + task score` 的三阶段 route basis，再通过 `stage1 band schedule` 做逐层插值。

## 1. 结论先行

HiDESC 不是单纯的“多专家 LoRA”，而是一个把持续学习拆成三件事的统一框架：

1. 训练时，用固定 `description cache` 作为跨任务语义锚点，联合优化 `CE + L_struct + L_align`。
2. 推理时，用 `early / middle / late` 三阶段 progressive routing 做 role-aware 专家协同，其中 text 路由信号来自频域化后的 `text activation prototype`。
3. 证据上，用两个前置实验分别支撑 `L_focus` 的必要性和 `b1/b2` 阶段边界的合理性。

对应实现分别在：

- [训练入口](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/train_MOE.py>)
- [训练损失](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/llava_trainer.py>)
- [分层路由](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/llava_arch.py>)
- [默认路由配置](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/language_model/llava_llama.py>)

## 2. HiDESC 的训练主线

### 2.1 数据流

训练侧先把每条样本构造成两条视图：

1. 主任务视图：`input_ids / labels / images`
2. description 视图：`description_input_ids`

如果启用 description continual learning，还会额外读取：

1. `reference_description_states`
2. `reference_description_mask`
3. `reference_description_available`

这些字段由 [train_MOE.py](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/train_MOE.py#903>) 的 `LazySupervisedDataset.__getitem__` 和 [DataCollatorForSupervisedDataset](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/train_MOE.py#976>) 拼接出来。

### 2.2 训练损失

HiDESC 的训练损失是三部分：

1. `standard CE`
2. `L_align`：作用在 `B1` band
3. `L_struct = L_focus + L_energy`：作用在 `B2` band

代码上，`compute_loss()` 会先跑主任务 forward，再抽取 `B1` 的 hidden states 做对齐，最后抽取 `B2` 的 description states 做结构约束，见 [llava_trainer.py](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/llava_trainer.py#756>)。

其中：

- `L_align` 的实现是 pooled image/text cosine loss，见 [#379](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/llava_trainer.py#379>)。
- `L_focus` 的核心是把 token change mass 压到 key tokens 上，见 [#517](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/llava_trainer.py#517>)。
- `L_energy` 用 mean change 和 margin 做平方 hinge，见 [#558](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/llava_trainer.py#558>)。

进一步写成公式，当前代码对应的是：

`L_total = w_ce L_ce + L_struct^B2 + w_align L_align^B1`

其中：

`L_align^B1 = \sum_{l \in B1} \alpha_l^{(B1)} L_align}^{(l)}`

`L_struct^B2 = \sum_{l \in B2} \beta_l^{(B2)} \left(w_f L_focus^{(l)} + w_e L_energy^{(l)}\right)`

这里 `B1 = [b1_low_layer, b1_high_layer]`，`B2 = [b2_low_layer, b2_high_layer]`，默认值在 [train_MOE.py](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/train_MOE.py#231>) 中分别是 `15-18` 与 `29-31`。

### 2.2.1 band 内自适应权重

HiDESC 在 `B1/B2` 内不是简单平均，而是显式做了“位置先验 + 难度自适应”的 band weighting，核心函数是 `_compute_band_weights()`，见 [llava_trainer.py#416](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/llava_trainer.py#416>)。

对 band 内每层先定义一个难度值 `d_l`：

- `B1` 中，`d_l = detach(L_align^{(l)})`
- `B2` 中，`d_l = detach(L_focus^{(l)} + \rho L_energy^{(l)})`

其中 `\rho = struct_band_energy_rho`，见 [#644](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/llava_trainer.py#644>)。

然后对每层做 EMA：

`m_l^{(t)} = \gamma m_l^{(t-1)} + (1-\gamma)d_l^{(t)}`

这里 `\gamma = loss_band_ema_gamma`，默认 `0.9`，见 [train_MOE.py#242](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/train_MOE.py#242>)。

再做 band 内标准化：

`z_l = \frac{m_l - \mu_B}{\sigma_B + \epsilon}`

同时构造一个从 band 低层到高层单调增加的位置先验：

`p_l = \epsilon_p + (1-\epsilon_p)\frac{l-l_{min}}{l_{max}-l_{min}}`

其中 `\epsilon_p = loss_band_position_eps`，默认 `0.05`，对应 [llava_trainer.py#421](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/llava_trainer.py#421>) 到 [#437](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/llava_trainer.py#437>)。

最终 band 权重是：

`w_l = softmax_B(\log p_l + \eta z_l)`

其中：

- `\eta = align_band_eta` 用于 `B1`
- `\eta = struct_band_eta` 用于 `B2`

所以它不是纯手工权重，而是：

1. 后层默认更重要
2. 当前更难的层临时拿到更大权重
3. 但这种放大被 band 内 softmax 和 EMA 平滑约束住

### 2.2.2 B1 与 B2 的具体对应

`B1` 的聚合在 `_compute_align_band_loss()` 中实现，见 [#440](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/llava_trainer.py#440>)。其单层 loss 为：

`L_align^{(l)} = 1 - cos(\bar{h}_{img}^{(l)}, \bar{h}_{txt}^{(l)})`

其中 `\bar{h}_{img}^{(l)}` 和 `\bar{h}_{txt}^{(l)}` 是 image token 与 text token 的 masked mean pooling，见 [#401](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/llava_trainer.py#401>) 到 [#406](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/llava_trainer.py#406>)。

`B2` 的聚合在 `_compute_struct_band_loss()` 中实现，见 [#572](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/llava_trainer.py#572>)。这里所有层共享同一个 reference description 序列，但每一层都单独计算：

- `L_focus^{(l)}`
- `L_energy^{(l)}`

再用 `\beta_l^{(B2)}` 聚合。

### 2.3 description cache

HiDESC 不是每次训练都在线重算 reference，而是先离线抽 cache。`extract_description_cache()` 会在 `b2_high_layer` 处截取 description hidden states，再用 `select_expanded_description_tokens()` 过滤出真正的 description token 片段，见 [train_MOE.py](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/train_MOE.py#1207>) 与 [description_utils.py](<__REPO_ROOT__/LLaVA/HiDESC/llava/train/description_utils.py#1>)。

这一步的意义是：训练时所有后续任务都对同一个 reference 空间做约束，而不是让 reference 跟着任务漂移。

## 3. HiDESC 的推理主线

### 3.1 样本级路由

HiDESC eval 不是依赖任务 id，而是先从图像与文本摘要里算 `task scores`，再映射到 role scores 和 expert weights。

核心路径在 [llava_arch.py](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/llava_arch.py>)：

- `_compute_shared_task_scores()`：基于 `spectral_image_anchors + text_anchors + history prior` 计算 task logits
- `_score_roles()`：基于 `role_text_prototypes + role_spectral_prototypes + role prior` 计算 role logits
- `_build_progressive_route_plan()`：把 early/middle/late 三套 basis 组装成逐层 route plan

更具体地，task-level routing 的形式可以写成：

`s_t = \lambda_{img} s_t^{img} + \lambda_{txt} s_t^{txt} + \lambda_{hist} s_t^{hist}`

其中：

- `s_t^{img}` 来自当前样本的频域图像描述子与 `spectral_image_anchors[t]` 的相似度
- `s_t^{txt}` 来自当前文本的 `activation index` 与 `text_anchors[t]` 的相似度
- `s_t^{hist}` 来自 `expert_usage_prior[t]`

这里现在必须明确区分两件事：

1. `text_guide_features` 仍然来自 text tower。
2. 真正写入 `text_anchors`、也真正参与 eval 相似度计算的，不再是 raw pooled text feature，而是 `_extract_text_activation_index()` 产出的频域化 text descriptor。

对应实现见 [llava_arch.py](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/llava_arch.py>) 与新增的 [relation_text_utils.py](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/relation_text_utils.py>)。

### 3.1.1 text prototype 的最新聚合方式

当前 `text` 侧 anchor 的真实实现是：

1. 先取每个样本的 text tower 输出 `h \in \mathbb{R}^{D}`。
2. 去掉 DC 分量：

`\tilde{h} = h - mean(h)`

3. 对 channel 维做一维 FFT：

`F = FFT(\tilde{h})`

4. 用频率绝对值构造高通权重：

`w_k = \left(\frac{|freq_k|}{\max_j |freq_j| + \epsilon}\right)^p`

其中 `p = text_activation_highpass_exponent`，当前默认 `0.75`。

5. 对加权频谱 `F^{hp}_k = w_k F_k`，分别取：

- `log(1 + |F^{hp}|)`
- `Re(F^{hp})`
- `Im(F^{hp})`

6. 三个分支先各自归一化，再做加权求和并归一化：

`z_{txt} = norm(\alpha_{mag} norm(log(1 + |F^{hp}|)) + \alpha_{real} norm(Re(F^{hp})) + \alpha_{imag} norm(Im(F^{hp})))`

当前主配置是：

- `text_activation_magnitude_weight = 0.0`
- `text_activation_real_weight = 0.5`
- `text_activation_imag_weight = 0.5`
- `text_activation_use_fftshift = true`

所以这轮调整的关键点是：`text` 路由更强调“去模板化之后的高频激活形状”，而不是直接保留原始 pooled embedding 的低频均值信息。

训练或 cache 抽取时，`text_anchors[task_id]` 也不再做简单 running mean，而是做 batch summary 后的 EMA：

`a_t^{new} = norm(\delta a_t^{old} + (1-\delta)\bar{z}_{txt})`

其中：

- `\bar{z}_{txt}` 是当前 batch 的 mean activation index
- `\delta = text_activation_ema_decay`，当前默认 `0.8`

对应实现见 [relation_text_utils.py](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/relation_text_utils.py#14>) 与 [llava_arch.py](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/llava_arch.py#504>)。

### 3.1.2 offline text/image anchor bank 的补充聚合

如果走离线 prototype cache 链路，当前不是把每个 task 的均值直接塞回 checkpoint，而是会额外做一次 task-bank 级别的 refine，入口在 [analyze_hidesc_spectral_routing.py](<__REPO_ROOT__/scripts/analyze_hidesc_spectral_routing.py>) 与 [inject_hidesc_prototype_cache.py](<__REPO_ROOT__/LLaVA/HiDESC/scripts/MCITlib/inject_hidesc_prototype_cache.py>)。

离线 bank 的逻辑是：

1. 先对每个 task 求 raw mean，得到 `u_t`。
2. 做 row normalize。
3. 如果启用 `offline_anchor_remove_global_mean=true`，则先去全局 task mean：

`\tilde{u}_t = norm(u_t - mean_j(u_j))`

4. 如果启用 contrastive refine，则减掉最相近 hard negatives 的加权混合：

`\hat{u}_t = norm(\tilde{u}_t - \lambda \sum_{j \in \mathcal{N}_t} \pi_{tj} \tilde{u}_j)`

5. 最后再用 `offline_anchor_preserve_mean_weight` 混回一部分原始方向。

当前 `UCIT` 主配置里：

- `offline_anchor_remove_global_mean = true`
- `offline_anchor_negative_top_k = 2`
- `offline_anchor_preserve_mean_weight = 0.15`
- `offline_image_anchor_contrast_weight = 0.40`
- `offline_text_anchor_contrast_weight = 0.20`

这意味着最新版本的 text prototype 聚合不是单点改动，而是分成两层：

1. 样本内：`raw text feature -> activation index`
2. task 间：`EMA/task mean -> global-mean removal -> hard-negative contrast refine`

### 3.1.3 FFT image prototype 的计算

HiDESC 不是用全局 image embedding 做视觉记忆，而是先把 patch grid 变成频域描述子。实现是 `_extract_image_spectral_descriptor()`，见 [llava_arch.py#342](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/llava_arch.py#342>)。

对 projector 后的 patch feature `X \in \mathbb{R}^{H \times W \times D}`，代码等价于：

1. 对空间维做二维 FFT：

`F = FFT2(X)`

2. 把频谱拆成：

- magnitude: `log(1 + |F|)`
- real part
- imaginary part

3. 用半径分桶，把频率划成 low/high 两个区域，再各自细分为若干 bins。默认：

- `spectral_cutoff = 0.33`
- `spectral_low_bins = 4`
- `spectral_high_bins = 4`

见 [llava_arch.py#286](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/llava_arch.py#286>)。

4. 对每个 bin 做平均池化，得到：

`v^{mag}, v^{real}, v^{imag}`

5. 最终描述子是三者的加权和再归一化：

`z_{fft} = norm(0.55 norm(v^{mag}) + 0.25 norm(v^{real}) + 0.20 norm(v^{imag}))`

对应代码见 [#405](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/llava_arch.py#405>) 到 [#413](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/llava_arch.py#413>)。

### 3.2 三阶段路由

构造函数里的默认值写在 [llava_llama.py](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/language_model/llava_llama.py#110>)，当前 `UCIT` eval 主路径使用 prototype-only role gate 和 role-constrained late expert selection，核心配置是 [ucit_role_new_partition_eval_late_role_prototype_only.json](<__REPO_ROOT__/configs/routing_configs/HiDESC/ucit_role_new_partition_eval_late_role_prototype_only.json>)。

当前 eval 侧最关键的几个开关是：

- `routing_early_layers = 16`
- `routing_middle_layers = 13`
- `routing_late_layers = 3`
- `routing_score_normalization = zscore`
- `use_stage1_band_schedule_eval = true`
- `eval_use_role_spectral_prototype = true`
- `eval_disable_role_image_prototype = true`
- `routing_late_top_k = 1`

但更重要的是，这轮之后文档不该再把 early/middle/late 写成三套彼此独立的 heuristic，而应该写成统一的 `role-prior task routing`：

1. 先算 role scores。
2. 再把 role scores 变成 task-level role support。
3. 最后把 `task scores` 与 `log(role support)` 融合成每个 stage 的 route basis。

也就是对某个 stage，可写成：

`w_t^{stage} = softmax(\frac{s_t}{\tau_{task}} + \gamma_{role}\log(q_t + \epsilon))`

其中：

- `s_t` 是 task score
- `q_t` 是由 role weights 扩散到 member tasks 后得到的 role support
- `\tau_{task}` 是 stage-specific task temperature
- `\gamma_{role}` 是 stage-specific role strength

所以当前主路径已经不再是“late 阶段直接对 task logits 做 top-k sparse selection”的旧描述。现在 early/middle/late 三个 basis 都统一由 `_build_role_conditioned_task_weights()` 生成，区别主要来自：

- `routing_*_role_temperature`
- `routing_*_task_temperature`
- `routing_*_role_strength`
- `routing_role_task_floor`

只有 `candidate_experts` 这一步会把 `late_basis` 的 `argmax` 记下来作为最终候选 expert 列表，因此“late 更尖锐、更接近单专家”仍然成立，但它不再等价于旧版的硬 top-k 稀疏权重。

代码里真正的层级切分由 `_get_layer_stage()` 完成，band 过渡则由 `stage1_band_schedule` 接管；更准确地说，当前实现是在 `route weights` 上做 band 内插值，而不是混输出 logits，见 [llava_arch.py](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/llava_arch.py#1429>) 和 [llava_stage1_band_eval_schedule.json](<__REPO_ROOT__/configs/routing_configs/HiDESC/llava_stage1_band_eval_schedule.json#1>)。

### 3.2.1 role 是如何划分的

role 不是外部标签，而是训练过程中根据 task anchors 逐步归纳出来的 latent grouping。这里的 `text anchor` 现在应理解为 `text activation anchor`，而不是旧口径下的 raw text pooled prototype。

模型里保存这套记忆的核心张量是：

- `spectral_image_anchors`
- `text_anchors`
- `role_spectral_prototypes`
- `role_text_prototypes`
- `task_role_membership`
- `role_task_count`
- `active_role_count`

定义见 [llava_llama.py#71](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/language_model/llava_llama.py#71>) 到 [#109](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/language_model/llava_llama.py#109>)。

当一个 task 训练完成后，会调用 `finalize_current_task_role_memory()`，进而进入 `_finalize_current_task_role_memory_impl()`，见 [llava_llama.py#219](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/language_model/llava_llama.py#219>) 和 [llava_arch.py#1666](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/llava_arch.py#1666>)。

其逻辑可以概括为：

1. 取当前 task 的 `(image_anchor, text_anchor)`
2. 用历史 task anchors 先算 `expert_logits`
3. 再对每个已有 role 算 compatibility score
4. 若没有 role 足够相似，则新建 role
5. 否则把当前 task 软分配到 top-k role

role compatibility 的核心实现是 `_compute_role_pair_compatibility()`，见 [llava_arch.py#855](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/llava_arch.py#855>)。默认策略是 `task_affinity_complete_link`，也就是：

`score(r, t) = \min_{j \in role(r)} sim(anchor_t, anchor_j)`

其中 `sim` 同时融合 image/text 两个 anchor，相当于 complete-link clustering。

这里还要区分两套阈值口径：

1. 模型构造函数中的默认在线阈值更严格，用于通用运行时初始化。
2. 当前 `UCIT` 的离线注入 + eval 主配置更宽松，因为它假设 task anchors 已经过离线 bank refine。

构造函数默认值是：

- `role_assignment_min_similarity = 0.85`
- `role_birth_threshold = 0.80`
- `role_assignment_margin = 0.02`
- `role_assignment_pair_weight = 0.50`

而当前 `UCIT` eval config 中覆盖为：

- `role_assignment_min_similarity = 0.00`
- `role_birth_threshold = 0.29`
- `role_assignment_margin = 0.01`
- `role_assignment_pair_weight = 0.30`

所以论文口径里更稳妥的说法应是：role assignment 仍是 complete-link 风格的 image/text 双锚兼容性聚类，但实际阈值由具体 eval routing config 决定，而不是固定死在 checkpoint 里。

另外，若是离线注入 prototype cache，注入脚本会先清空旧 role memory，再由 `ensure_role_bank_initialized()` 按新的 task anchors 在运行时重建 role prototypes、membership 与 `active_role_count`，见 [inject_hidesc_prototype_cache.py](<__REPO_ROOT__/LLaVA/HiDESC/scripts/MCITlib/inject_hidesc_prototype_cache.py>) 与 [llava_arch.py#1606](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/llava_arch.py#1606>)。

### 3.2.2 role prototype 如何更新

当 task 被分配到某个 role 后，真正的 prototype 更新在 `_commit_task_to_roles()` 中完成，见 [llava_arch.py#1551](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/llava_arch.py#1551>)。

若是新 role：

- 直接令 `role_spectral_prototype = task_image_anchor`
- `role_text_prototype = task_text_anchor`

若并入已有 role，则做带计数的加权均值：

`p_r^{new} = \frac{n_r p_r^{old} + w_{tr} a_t}{n_r + w_{tr}}`

其中：

- `a_t` 是当前 task anchor
- `w_{tr}` 是 task 对 role 的 membership weight
- `n_r` 是 `role_task_count[r]`

这正对应 [llava_arch.py#1564](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/llava_arch.py#1564>) 到 [#1600](<__REPO_ROOT__/LLaVA/HiDESC/llava/model/llava_arch.py#1600>)。唯一需要更新的理解是：这里的 `task_text_anchor` 与 `role_text_prototype` 都已经处在 `activation index` 空间中。

### 3.3 stage1 band schedule

当前默认 schedule 是：

- `early_core = [1, 14]`
- `b1_band = [15, 18]`
- `middle_core = [19, 28]`
- `b2_band = [29, 31]`
- `late_core = [32, 32]`

并且在 band 内预先给出离散 `alpha_by_layer`，例如 `15 -> 0.25`、`16/17 -> 0.5`、`18 -> 0.75`，见 [配置文件](<__REPO_ROOT__/configs/routing_configs/HiDESC/llava_stage1_band_eval_schedule.json#1>)。

这与代码中的 `_get_stage1_band_region()`、`_get_stage1_band_alpha()`、`_build_progressive_route_plan()` 是一致的。当前正确的实现表述应是：

1. 先计算 `early_basis / middle_basis / late_basis`
2. 再按照 `b1/b2` 的 `alpha_by_layer`，在相邻阶段 basis 之间做 route-weight 插值
3. band 外不再额外拟合逐层 noisy schedule

## 4. 前置实验一：Stage 1 边界实验

这个实验解决的问题不是“哪一层最好”，而是“是否存在稳定的阶段过渡带”。

### 4.1 实现口径

分析脚本在 [stage1_analyze_boundaries.py](<__REPO_ROOT__/scripts/stage1_analyze_boundaries.py>)，它会对每层生成三条曲线：

1. `reasoning_signal`
2. `objective_signal`
3. `style_signal`

默认的 `b1_only_crossover` 策略在 `b1_search_low=10` 到 `b1_search_high=20` 之间搜索，并要求：

- `transition_min_run = 3`
- `min_transition_support = 0.01`
- `b1_band_score_tolerance = 0.03`
- `b1_band_min_width = 3`
- `b1_band_max_width = 5`

对应代码见 [#486](<__REPO_ROOT__/scripts/stage1_analyze_boundaries.py#486>)。

### 4.1.1 三个 signal 的正式公式

这三个 signal 的真实实现来自 `compute_split_curves()`，见 [stage1_analyze_boundaries.py#902](<__REPO_ROOT__/scripts/stage1_analyze_boundaries.py#902>)。

`reasoning_signal` 与 `objective_signal` 都调用 `fisher_separation()`，见 [#267](<__REPO_ROOT__/scripts/stage1_analyze_boundaries.py#267>)。对某一层 `l`，其定义是：

`Signal(l) = \frac{\sum_c n_c ||\mu_c^{(l)} - \mu^{(l)}||_2^2}{\sum_c \sum_{i \in c} ||h_i^{(l)} - \mu_c^{(l)}||_2^2 + \epsilon}`

其中：

- `h_i^{(l)}` 是样本 `i` 在层 `l` 的 hidden vector
- `\mu_c^{(l)}` 是类别 `c` 的层内均值
- `\mu^{(l)}` 是全局均值

所以：

- `reasoning_signal(l)` 是在 reasoning-family 样本上，用 `objective_label` 分组后的 Fisher separation
- `objective_signal(l)` 是在 objective-family 样本上，用 `objective_label` 分组后的 Fisher separation

`style_signal` 则来自 style-family 样本在同一 paired group 内的 JS divergence，见 [#945](<__REPO_ROOT__/scripts/stage1_analyze_boundaries.py#945>)。对同组不同模板的 style logits `u_i^{(l)}` 与 `u_j^{(l)}`：

`p_i^{(l)} = softmax(u_i^{(l)})`

`JS(p_i^{(l)}, p_j^{(l)}) = \frac{1}{2}KL(p_i^{(l)} || m^{(l)}) + \frac{1}{2}KL(p_j^{(l)} || m^{(l)})`

`m^{(l)} = \frac{1}{2}(p_i^{(l)} + p_j^{(l)})`

最终 `style_signal(l)` 是组内所有模板对的平均 JS。

### 4.1.2 从三条 signal 到 `b1`

在 `b1_only_crossover` 策略下，先对三条曲线做：

1. `min-max normalize`
2. `window=3` 的 moving average smoothing

见 [#637](<__REPO_ROOT__/scripts/stage1_analyze_boundaries.py#637>) 到 [#639](<__REPO_ROOT__/scripts/stage1_analyze_boundaries.py#639>)。

然后对每个候选边界 `b1`，定义：

`pre\_margin(b1) = mean(R_l - O_l), l \in [b1-k, b1)`

`post\_margin(b1) = mean(O_l - R_l), l \in [b1, b1+k)`

`boundary\_contrast(b1) = pre\_margin(b1) + post\_margin(b1)`

`local\_score(b1) = min(pre\_margin, post\_margin) + boundary\_contrast + \lambda_{late}(b1-low)`

其中 `k = transition_min_run`。对应代码见 [#651](<__REPO_ROOT__/scripts/stage1_analyze_boundaries.py#651>) 到 [#664](<__REPO_ROOT__/scripts/stage1_analyze_boundaries.py#664>)。

### 4.2 结论

这轮多 seed 统计的主共识是：

- `b1` 更适合写成区间，而不是单层
- 主共识区间是 `15-18`
- 峰值支持在 `16`

完整结果见 [stage1_b1_interval_report_20260811_zh.md](<__REPO_ROOT__/docs/stage1_b1_interval_report_20260811_zh.md>)。

### 4.3 对 HiDESC 的映射

这个实验直接支撑了 HiDESC 的推理分段：

1. `b1` 不是硬编码单层，而是 `early -> middle` 的过渡带。
2. `b2` 同理对应 `middle -> late` 的过渡带。
3. 所以 eval 中的 route plan 应当使用 band schedule，而且插值对象应是 `route weights`，而不是只用单点边界或直接混输出 logits。

## 5. 前置实验二：`L_focus` 逻辑实验

这个实验回答的是另一个问题：

> 为什么 `description` 里的更新必须抑制 template drift，而不能只看总漂移大小？

### 5.1 实验逻辑

最新 summary 明确把问题改写成“template drift 是否和 semantic instability 耦合”，见 [l_focus_logic_summary.md](<__REPO_ROOT__/docs/experiment2_l_focus_logic_full_20260806/l_focus_logic_summary.md>)。

结果显示：

- `All Tasks` 的 harmful template-drift support rate = `0.500`
- `Early Tasks` = `0.000`
- `Late Tasks` = `0.750`
- `Task 4 / IconQA` 给出 `strong_support`

这说明真正需要抑制的不是“所有更新”，而是会污染跨模板语义一致性的 template-driven drift。

### 5.2 对 HiDESC 的映射

这直接解释了为什么 HiDESC 训练里要保留：

1. `description_focus_loss`
2. `description_energy_loss`

以及为什么它们仍然只放在固定的 `description layer` 上，而不是做成全层共享正则。

换句话说：

- `L_focus` 负责把更新集中到 content-bearing tokens
- `L_energy` 负责限制 description hidden 的整体漂移

## 6. 两个前置实验如何合起来支撑 HiDESC

可以把 HiDESC 的整体逻辑写成一条链：

1. Stage 1 边界实验告诉我们，模型内部确实存在稳定的 `early / middle / late` 过渡区。
2. `L_focus` 实验告诉我们，description 训练里真正有害的是 template-driven drift。
3. 因此，训练侧要用 `B1` 做对齐，`B2` 做结构约束。
4. 推理侧要用 `stage1 band schedule` 做 progressive route，并把 text 侧 prototype 放在 `activation index` 空间中统一建模。

## 7. 可直接写进论文的表述

> HiDESC is a hierarchical description-aligned continual learning framework for LVLMs. During training, it anchors each task to a fixed description cache and maintains both spectral image anchors and FFT-style text activation anchors; during inference, it performs task-agnostic progressive collaboration with role-aware task routing, where stage-specific route bases are interpolated by `b1/b2` band schedules. Two preliminary studies support this design: Stage 1 boundary analysis shows that the model exhibits a stable transition band around `b1 = 15-18` and a late structural band around `b2 = 29-31`, while the `L_focus` study shows that harmful drift is driven by template-level semantic instability rather than raw update magnitude.

## 8. 复现入口

- 训练：[`LLaVA/HiDESC/scripts/MCITlib/Train/full_HiDESC_from_HiDeTask1.sh`](<__REPO_ROOT__/LLaVA/HiDESC/scripts/MCITlib/Train/full_HiDESC_from_HiDeTask1.sh>)
- 离线 prototype 分析：[`scripts/analyze_hidesc_spectral_routing.py`](<__REPO_ROOT__/scripts/analyze_hidesc_spectral_routing.py>)
- 离线 prototype 注入：[`LLaVA/HiDESC/scripts/MCITlib/inject_hidesc_prototype_cache.py`](<__REPO_ROOT__/LLaVA/HiDESC/scripts/MCITlib/inject_hidesc_prototype_cache.py>)
- UCIT offline smoke eval：[`LLaVA/HiDESC/scripts/MCITlib/Eval_UCIT/run_offline_hidesc_ucit_smoke_eval.sh`](<__REPO_ROOT__/LLaVA/HiDESC/scripts/MCITlib/Eval_UCIT/run_offline_hidesc_ucit_smoke_eval.sh>)
- UCIT offline full eval：[`LLaVA/HiDESC/scripts/MCITlib/Eval_UCIT/run_offline_hidesc_ucit_full_eval.sh`](<__REPO_ROOT__/LLaVA/HiDESC/scripts/MCITlib/Eval_UCIT/run_offline_hidesc_ucit_full_eval.sh>)
- Stage 1：[`scripts/run_stage1_b1_multiseed.sh`](<__REPO_ROOT__/scripts/run_stage1_b1_multiseed.sh>)
- `L_focus`：[`LLaVA/HiDe/scripts/MCITlib/Analysis/experiment2_prelim_parallel.sh`](<__REPO_ROOT__/LLaVA/HiDe/scripts/MCITlib/Analysis/experiment2_prelim_parallel.sh>)
