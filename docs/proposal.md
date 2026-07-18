# Hi-DESC思路


## 1. 背景

随着以Llava为代表的大型多模态模型（LVLMs）的广泛应用，持续学习（Continual Learning）成为提升模型在非平稳环境（Non-stationary Environments）下处理新任务能力的关键研究热点。在实际场景中，模型需要不断获取新能力，同时维护既有的知识体系，这对于实现类人通用智能具有重要意义。

## 2. 现状

近年来，针对大语言模型的高效微调（PEFT）技术，特别是 LoRA 及其变体，已被广泛应用于持续学习领域，以实现低成本的知识累积：  

1. 参数解耦与结构扩展：部分研究提出通过维护多个 LoRA 模块来存储不同任务的知识，例如 O-LoRA 通过正交限制来减少新旧任务间的参数干扰；MRLoRA 则通过低秩结构的动态调整来适应不同任务的规模变化；KeepLoRA 倾向于保留关键参数子空间以缓解遗忘。这些方法实现了各任务参数空间的有效物理隔离，避免了不同任务的LoRA模块之间的干扰，**却忽略了任务间的深层逻辑关联，缺乏统一的语义对齐机制。模型在处理多模态任务时，难以通过任务间的知识共性来增强迁移能力**。

2. 路由机制与专家协同：为了解决多任务下的路由问题，例如HiDE方法采用专家LoRA深层等权融合，表层路由稀疏化激活专家LoRA的方式，防止训练过程中的灾难性遗忘；RegLoRA 通过正则化限制锚点更新避免深层推理能力遗忘、多种答案模式训练解决表层答案风格遗忘问题。**但这些方法忽视了相关性强的任务推理层面的协同效应，在前向推理时，将之前所有任务的LoRA同等对待**。


## 3. 现状总结出来的问题

上述现有方法大多将持续学习视为“模块存取”问题，存在以下共性局限：  

1. 协同机制缺失：现有工作多关注专家激活的“排他性”，忽略了推理过程中通过“知识协同”，实现多能力重组的可能性。

2. 缺乏任务之间的语义一致性：这些方法忽略了专家LoRA之间语义空间的一致性，训练时单独训练当前任务专家LoRA容易导致各个专家LoRA之间语义空间的不一致，阻碍各个专家LoRA之间协同工作。


## 4. ⼀句话概括你的 Idea

针对上述问题，我们提出了一个统一的持续学习框架**Hi-DESC (Hierarchical Description-aligned Expert Collaborative Continual Learning)**。在训练阶段，Hi-DESC以固定的base model description cache作为跨任务共享语义锚点：在standard ce之外引入description focus loss和description energy loss，约束当前任务专家相对于base语义锚点的更新方式，从而维持各专家LoRA之间的语义一致性；前向推理时采用Role-aware Progressive Collaboration架构分层实现深层跨任务推理能力专家协同，以及浅层语言表达风格的任务差异化。

核心假设：**持续学习过程中，关键语义token处更新幅度较大，其他模板词处更新幅度较小**

## 5. 实现这个 Idea 的挑战

● 挑战1：
如何在持续学习的过程中，各个任务的专家LoRA之间语义空间对齐？

● 挑战2：
前向推理时，如何实现深层推理时正确激活任务相关的专家LoRA实现推理层面的协同，并在浅层保证输出语言风格正确？

## 6. 为了解决挑战⼀，你提出了什么技术？描述这个技术的

## Motivation

针对挑战1，我们提出了 HiDeCL (Hierarchical Description-based Continual Learning) 训练机制。从base model中抽取description hidden state作为所有后续任务共享的语义锚点。训练时，将当前模型在相同description输入上的hidden states与cache中的reference states进行对齐。这样做可以避免reference随任务推进不断漂移，使不同任务专家LoRA始终围绕同一个语义基准进行更新。

在loss设计上，当前训练目标由三部分组成：

1. `standard CE`：保证当前任务的主学习目标不被削弱。
2. `description focus loss`：约束description空间中的变化应尽量集中在key tokens上，而不是无差别扩散到全部token，提升专家的学习指向性。
3. `description energy loss`：约束description hidden states相对base reference的整体漂移幅度，避免新任务训练造成过强的全局偏移。


## 7. 为了解决挑战⼆，你提出了什么技术？描述这个技术的

核心假设：同一类型任务之间存在推理能力的相互促进，例如Math相关任务可以促进Physics相关任务的表现。

## Motivation

针对挑战2，我们提出了 Role-aware Progressive Collaboration 推理框架。该方法利用训练阶段维护的 latent roles（由 task anchors 聚类诱导得到），整体前向过程专家激活遵循“从粗到细”的模式：

1. 深层（1-16）：不同role的专家之间按照相关性评分加权fuse，实现通用推理能力的协同。  
2. 中层（16-32）：实现同一role内的专家加权激活，实现同一类型的专家LoRA之间任务特异性推理能力的协同。  
3. 浅层（32）：收敛至最相关（top k）的专家单元加权fuse，实现输出语言风格的精准匹配

## 8. 总结

1. 本研究的主要贡献在于：通过约束description 更新位置各个任务专家LoRA之间的语义空间，并结合动态latent role的维护，解决了之前方法中专家LoRA“排他性”过强、难以协同工作的问题。

2. 我们的创新点在于：通过base-anchored HiDeCL训练机制和层级推理架构，实现了训练阶段的跨任务语义可比性维护，以及推理阶段对分布式存储历史专家能力的精准调用。  

3. 未来，我们计划通过精细化description的保存形式，提升description cache的生成效率和空间储存效率。
