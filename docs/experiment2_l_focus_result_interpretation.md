# Experiment 2: `L_focus` 设计逻辑与结果解读

## 1. 这份文档要回答什么

这份文档专门回答一个容易被追问的问题：

> 既然当前 Hi-DESC patch 方案已经不再依赖显式 `deep / middle / shallow` 分层，为什么还需要保留 `L_focus`？

我们的回答不应该再建立在“某一层更像中层，所以要在这一层做 `L_focus`”这种旧叙事上，而应该建立在更稳定的 description-space 逻辑上：

1. 持续学习中，description 空间需要保留的是 `content words` 的语义坐标一致性。
2. 真正有害的漂移，不是“更新幅度大”本身，而是会污染这套语义坐标系的 `template-driven drift`。
3. 因此需要 `L_focus` 去抑制模板词上的无意义扩散更新，把更新尽量限制在有语义价值的位置上。

这也是 [docs/experiment2_l_focus_description_drift.md](/mnt/lyaa/MCITlib/docs/experiment2_l_focus_description_drift.md) 想验证的核心命题。

## 2. 为什么旧版论据不够稳

更早的叙事倾向于把 `L_focus` 的必要性写成：

1. `key token drift > template token drift`
2. `focus_ratio = template / key < 1`

但这条论据并不稳，因为它默认“只要关键语义词漂移更大，就说明更新是合理的”。问题在于，**更新大小本身并不能区分“有意义适应”和“无意义扩散”**。

现有 `ce-only` 预实验也说明了这一点。只看 drift 均值时，多数任务的 `key_mean` 和 `template_mean` 并没有拉开稳定差距：

1. `Task 1`: `focus_ratio = 1.012`
2. `Task 2`: `focus_ratio = 1.010`
3. `Task 4`: `focus_ratio = 1.050`
4. `Task 6`: `focus_ratio = 1.013`

这些结果见 [docs/experiment2_prelim_ce_only/description_drift_report.json](/mnt/lyaa/MCITlib/docs/experiment2_prelim_ce_only/description_drift_report.json)。

这意味着我们不能再把论证重点放在“谁漂得更大”上，而应该改问：

> 当 template token 漂移变大时，它会不会破坏相同 content words 在不同 prompt 模板下的语义稳定性？

如果答案是会，那么 `L_focus` 就有明确意义，因为它抑制的是**有害模板漂移**，而不是简单压小所有更新。

## 3. 实验二真正要验证的假设

按照新的实验设计，Experiment 2 想支持的不是“key token 必然漂得更大”，而是下面这条更本质的假设：

> 对同一图像，只要视觉语义词保持不变，description space 就应该让这些 content words 在不同 prompt 模板下仍然保持可比；如果 template token 的训练前后漂移变大，这种跨模板语义稳定性就更容易被破坏。

这条假设可以拆成两部分：

1. 我们真正想保护的是 `content-word cross-template semantic stability`。
2. 我们真正想抑制的是 `template drift` 对这套稳定性的污染。

因此，`L_focus` 的作用不应理解为“强迫 key token 一定漂更多”，而应理解为：

1. 降低模板词主导 hidden-state 更新的机会；
2. 让 description 适应更集中在承载视觉语义的词上；
3. 维持不同任务专家相对于共享 description reference 的可比性。

## 4. 为什么这个实验设计能回答这个问题

这个实验刻意不依赖显式层级，只固定一个 `description layer`，然后对同一图像构造：

1. `canonical prompt`
2. `template variant A`
3. `template variant B`
4. `template variant C`

这些 prompt 只改变 instruction wording，不改变 `objects`、`attributes`、`shapes`、`colors`、`textures`、`scene context`、`visible text`、`spatial relations` 等内容词。

这样做的好处是：如果训练后的 description space 仍然稳定，那么**同样的 content words 即使包在不同模板里，也应该保持接近**。反过来，如果模板部分的漂移会污染语义空间，那么：

1. content-word alignment 会下降；
2. content-word retrieval 会变差；
3. 模板漂移越大的样本，越容易伴随更差的跨模板对齐。

所以实验二不是在问“模型有没有更新”，而是在问：

> 更新是否开始破坏同一语义词在不同表达模板下的坐标一致性？

这正是 `L_focus` 想干预的问题。

## 5. 这次结果应该怎么看

目前最新的完整结果目录是：

1. [docs/experiment2_l_focus_logic_full_20260806/](/mnt/lyaa/MCITlib/docs/experiment2_l_focus_logic_full_20260806/)

主结论写在：

1. [l_focus_logic_summary.md](/mnt/lyaa/MCITlib/docs/experiment2_l_focus_logic_full_20260806/l_focus_logic_summary.md)

这份 summary 已经把评价口径从“谁漂得更大”切换成了“template drift 是否和 semantic instability 耦合”，因此它和新的实验目标是一致的。

### 5.1 总体结论

最新结果给出的总评是：

1. `overall conclusion = mixed_but_meaningful_support`

这句话很关键。它表示：

1. 证据不是每个任务都一致成立；
2. 但支持信号并不是零散噪声，而是有可解释模式；
3. 这种模式足以支撑“`L_focus` 的优化目标是合理的”。

### 5.2 为什么说它是“有意义的支持”

从 group summary 看，最重要的现象出现在持续学习的后期任务，而不是前期任务。

1. `All Tasks` 的 harmful template-drift support rate 是 `0.500`
2. `Early Tasks` 的 support rate 是 `0.000`
3. `Late Tasks` 的 support rate 是 `0.750`

这说明：

1. 在早期任务中，训练有时会让表示更稳定，或者至少不会明显破坏跨模板语义可比性；
2. 但到了后期任务，template drift 与 semantic degradation 的耦合显著更常见；
3. 这符合持续学习的直觉，因为任务累积之后，description space 更容易受到历史偏移和新任务更新的共同扰动。

换句话说，**越接近我们真正关心的 continual-learning 场景，`L_focus` 想抑制的问题就越明显**。

### 5.3 单任务层面的证据模式

单任务结果不是整齐划一的，但它们呈现出清晰分层：

1. `Task 4 / IconQA` 给出 `strong_support`
2. `Task 3 / VizWiz` 和 `Task 5 / CLEVR` 给出 `partial_support`
3. `Task 1 / ImageNet-R` 是 `mixed_signal`
4. `Task 2 / ArxivQA` 和 `Task 6 / Flickr30k` 是 `weak_or_null`

其中最关键的是 `Task 4 / IconQA`：

1. `alignment drop mean = 0.0253`
2. `margin drop mean = 0.0152`
3. `template/alignment corr = 0.4497`

这说明在这一任务上，模板漂移越大，跨模板 content 对齐越容易下降，而且这种关系并不弱。

`Task 3` 和 `Task 5` 虽然没有达到同样强度，但也出现了“模板漂移与语义退化部分耦合”的信号，因此更适合被解释为 `partial_support`，而不是完全无关。

### 5.4 为什么 early tasks 不支持，反而不一定是坏事

summary 里专门强调了一个原则：

1. 不要求每个任务都出现统一退化；
2. early tasks 甚至可能让表征更锐化；
3. 后期任务在持续学习里是更强证据。

这点非常重要。因为如果我们错误地要求所有 task 都必须出现明显退化，就会把“持续学习前期还没积累足够干扰”误判成“假设不成立”。

因此，这份结果更合理的读法不是：

> 为什么只有部分任务支持？

而是：

> 在最容易出现持续学习干扰的后期阶段，是否稳定出现了 template drift 与 semantic instability 的耦合？

从这个角度看，答案是偏肯定的。

## 6. 这组结果到底支持了什么

这次结果支持的是下面这条逻辑链：

1. description 空间里真正需要被保留的是 content words 的跨模板语义一致性；
2. 单纯比较 `key drift` 和 `template drift` 的大小，不足以判断更新是否合理；
3. 更关键的问题是模板词漂移会不会破坏语义词在不同模板下的可比性；
4. 最新结果显示，这种破坏在一部分任务上确实存在，尤其在后期任务中更明显；
5. 因而模板驱动的漂移不是无害噪声，而是会污染 description space 的真实问题；
6. 所以需要一个机制去抑制这种模板漂移，把更新约束在更有语义价值的位置上；
7. `L_focus` 正是为此存在。

也就是说，这个实验支持的是：

> `L_focus` 的目标是合理的。

而不是：

> `L_focus` 已经被这组结果严格证明是不可替代且因果有效的。

## 7. 这组结果还不能支持什么

为了避免后面写 proposal 或论文时说得太满，需要明确这次 Phase A 的边界。

这组结果还**不能单独证明**：

1. 只要加上 `L_focus`，语义稳定性就一定提升；
2. `L_focus` 比其他任意正则都更好；
3. 观测到的 template-drift 耦合已经足以构成完整因果链。

原因很简单：当前结果本质上仍是观测型前置证据。它能证明“问题存在且值得管”，但还不能单独证明“只有 `L_focus` 能解决，而且一定解决得最好”。

所以更准确的说法应该是：

1. Experiment 2 Phase A 证明了 `L_focus` 要处理的问题具有经验基础；
2. 它为保留 `L_focus` 提供了合理动机；
3. 若要进一步证明有效性，仍然需要 `w_focus = 0` vs `w_focus > 0` 的轻量 ablation。

## 8. 为什么这些结果已经足够支撑“需要保留 `L_focus`”

在当前 patch 方案里，我们其实只需要回答一个比“完全证明有效”更弱但更实际的问题：

> 在取消显式层级之后，`L_focus` 还有没有明确、独立、非冗余的存在理由？

这次实验给出的答案是：有。

因为它已经表明：

1. description 更新的风险不只是“漂移变大”，而是“模板漂移污染语义可比性”；
2. 这种风险在 continual learning 的后期任务里更明显；
3. `L_focus` 的归纳偏置恰好针对这个风险，而不是泛泛地压缩所有更新。

因此，即使我们不再依赖显式层级边界，`L_focus` 仍然有一个独立成立的角色：

> 它不是用来定义层级的，而是用来维持 description-space adaptation 的语义集中性，抑制 template-driven drift 对共享语义锚点的污染。

这就是“为什么仍然需要 `L_focus`”最核心、也最稳的表述。

## 9. 建议在 proposal 里如何表述

如果要把这次实验写进 proposal，建议避免使用下面这种容易被攻击的说法：

1. “我们证明了 key token 一定比 template token 漂得更大。”
2. “因此 `L_focus` 是必要的。”

更稳的写法应该是：

> Preliminary evidence suggests that the harmful part of description drift is not update magnitude per se, but template-driven drift that destabilizes cross-template semantic alignment. This motivates retaining `L_focus`, whose role is to bias adaptation toward content-bearing tokens and suppress semantically uninformative template drift.

如果写成中文，可以用下面这版：

> 我们的前置实验表明，description 空间中真正有害的并不是更新幅度本身，而是会破坏跨模板语义一致性的 template-driven drift。基于这一观察，我们保留 `L_focus`，其作用不是简单缩小更新，而是将适应限制在承载视觉语义的 content-bearing tokens 上，并抑制模板词上的无意义漂移。

## 10. 一句话总结

Experiment 2 的最新结果并没有支持“key token 一定漂得更多”这种旧假设，但它以更稳健的方式支持了新的核心逻辑：

> 在持续学习中，模板词漂移会在后期任务中更明显地破坏 content-word 的跨模板语义稳定性，因此我们需要 `L_focus` 去抑制 template-driven drift，并把 description 更新集中到真正有语义价值的位置上。
