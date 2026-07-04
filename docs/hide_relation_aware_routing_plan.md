# HiDe + Relation-Aware Routing + Selective Transfer Loss

## 目标

如果下一阶段只能选择一个主方法，当前最推荐的组合是：

`HiDe + relation-aware routing + selective transfer loss`

如果论文想强调的是**任务协同**，而不只是**防止遗忘**，那么这条路线比 `HiDe + description loss` 更合适。

## 为什么它应该作为主方法

当前 HiDe 的框架本身就已经具备较强的“协同式建模”特征：

- 它维护了任务级的 expert；
- 它在训练过程中统计 task anchor；
- 它在推理时对 expert 进行加权。

这说明 HiDe 天然更接近一个**协同路由框架**，而不是一个单纯依靠正则项的 continual learning 框架。因此，新的方法更应该强化“路由”和“迁移”本身，而不是只额外加入一个表示保持损失。

`HiDe + description loss` 虽然有价值，但它的核心仍然偏向稳定性约束：

- 它主要用于减小表示漂移；
- 它更容易被解释为遗忘控制正则；
- 它并没有直接回答“哪个旧任务应该帮助哪个新任务”。

相比之下，`HiDe + relation-aware routing + selective transfer loss` 是一个更完整的方法组合：

- `relation-aware routing` 回答的是：**当前样本或当前任务最应该调用哪些旧任务 expert？**
- `selective transfer loss` 回答的是：**哪些旧任务知识值得迁移，以及应该如何迁移，而不是把所有旧任务一视同仁地强行对齐？**

这更符合论文希望强调的核心故事：

- 不是所有旧任务都会同等有帮助；
- 有价值的旧任务应该被选择性激活；
- 迁移应该是正向的、稀疏的、关系依赖的。

## 推荐的方法定义

### 1. Relation-Aware Routing

不要只依赖一个粗粒度的相似度分数，而是显式估计任务关系信号，再用它来调节 expert 权重。

路由分数可以由以下几部分组成：

- visual anchor 相似度；
- text anchor 相似度；
- 可选的 reasoning-style 相似度；
- 可选的旧 expert 在当前任务上的历史 utility。

目标行为应该是：

- 高相关任务获得更高的 expert 权重；
- 低相关任务被抑制；
- 推理时只调用少量真正相关的旧 expert，而不是把所有旧任务平均对待。

### 2. Selective Transfer Loss

迁移损失不应该鼓励模型向所有旧任务统一靠拢，而应该只对**相关任务**进行选择性约束。

这一点很重要，因为 UCIT 并不是一个“六个任务都应该被压进同一个统一表示空间”的 benchmark。它内部有些任务重叠明显，有些任务相关性则较弱。

期望的行为应该是：

- 当前特征向有帮助的旧任务 expert 靠拢；
- 避免对所有历史任务做全局统一对齐；
- 降低来自不相关任务的负迁移。

## 为什么它比 Description Loss 更适合作为主线

`description loss` 仍然值得保留，可以作为辅助 baseline 或消融项，但不适合单独作为论文主贡献。

原因是：

- 它更强调“保持”而不是“协同”；
- 只靠它较难有力解释跨任务正迁移；
- 它没有显式建模任务之间的可用性关系。

如果论文希望论证“先前任务可以主动提升后续任务性能”，那么“路由 + 选择性迁移”会比单纯的 description alignment 更直接，也更容易自洽。

## UCIT Benchmark 的适配性

UCIT 是可以支持协同型方法的，但这里的协同更适合被描述为**结构化协同**，而不是**均匀协同**。

六个任务分别是：

1. `ImageNet-R`
2. `ArxivQA`
3. `VizWiz`
4. `IconQA`
5. `CLEVR-Math`
6. `Flickr30k`

其中最合理的协同模式包括：

- `Flickr30k <-> VizWiz`：共享 caption 和场景描述能力；
- `CLEVR-Math <-> IconQA`：共享计数、符号化、组合式视觉推理能力；
- `CLEVR-Math / IconQA -> ArxivQA`：结构化推理和定量理解对图表/科研图问答的迁移；
- `ImageNet-R -> Flickr30k / VizWiz / ArxivQA`：鲁棒视觉语义和识别能力带来的底层支持。

因此，这个 benchmark 更适合支撑这样一种论文叙事：

`task collaboration is relation-dependent`

而不是：

`all tasks should always help each other equally`

## 推荐的论文定位

如果只能推进一个主方法，推荐的论文定位是：

**在 HiDe 上引入 relation-aware expert routing 与 selective positive transfer**

它比 `HiDe + description loss` 更完整，因为它同时包含：

- 一个协同机制；
- 一个迁移控制机制；
- 一个更清晰的“为什么某个旧任务会帮助某个新任务”的解释路径。

## 当前 `train_UCIT.sh` 的结果

当前认可的 HiDe 全流程 `train_UCIT.sh` baseline 结果如下：

| Metric Type | Image-R | ArxivQA | Viz-cap | IconQA | CLEVR | Flickr30k | Average |
| ----------- | ------- | ------- | ------- | ------ | ----- | --------- | ------- |
| Avg         | 85.70   | 92.70   | 54.10   | 66.87  | 59.12 | 55.15     | 68.94 (+4.4) |
| Last        | 80.50   | 89.83   | 48.78   | 62.90  | 47.97 | 55.15     | 64.19 (+5.8) |

后续的方法设计与实验比较，均可将上表作为当前工作基线。
