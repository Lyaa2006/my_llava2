## progressive collaboration 怎么和动态 role 结合

这个部分可以完全保留，只把输入从“固定 role”改成“动态诱导 role”。

### 浅层
浅层面对的是当前时刻已有的 role prototypes：
- 当前样本先对 role 做门控
- 这些 role 是动态形成的，不是人工定义的

### 中层
中层在被激活的 role 内部选择具体历史 task experts：
- 如果 role 是软分配形成的，那么某个 task 在多个 role 中都可能有贡献
- 如果 role 是硬分配形成的，就只在对应 role 内选择

### 深层
深层仍然做稀疏 task-level selection：
- 从中层留下的候选 expert 中选 `top-k`
- 完成最终融合

所以整体框架不变，只是把最前面的 role source 改成了**online discovered roles**。

---

## 我建议你把方法改成三阶段

### Stage 1: Task Bootstrap
当前新任务先做短暂 warm-up，收集：
- 当前任务 anchor
- 当前任务 early-layer response pattern

### Stage 2: Online Role Induction
将当前任务和历史 role prototypes 匹配：
- 若匹配成功，得到 role membership
- 若都不匹配，创建新 role

### Stage 3: Role-Aware Progressive Collaboration
再进入正式协同：
- 浅层 role-level
- 中层 intra-role
- 深层 sparse task-level

这个版本就合理很多，因为它符合 continual benchmark 的设定：
**新任务到来之前，方法并不知道它属于什么协同类别。**

---

## 还需要补一个机制：role birth / merge / update

为了让这个方法长期可用，role memory 最好支持三种操作：

- `birth`
  当前任务和所有 role 都不接近时，新建 role

- `update`
  当前任务属于某个已有 role，就更新该 role prototype

- `merge`
  两个 role 后续变得非常接近时，可合并成一个 role

初版其实可以先只做：
- `birth`
- `update`

先不做 merge，避免系统太复杂。

---

## 最终你应该怎么表述这个修正版

你可以把原方案升级成：

**Adaptive Role-Aware Progressive Collaboration**

核心定义变成：

- role 不是 benchmark-specific semantic category
- role 是历史任务中自动形成的 latent collaboration pattern
- 当前任务通过 online role induction 接入已有协同结构
- 若现有协同结构不足以解释新任务，则动态创建新 role

这样就彻底摆脱了 UCIT 特定划分的问题。

---

## 如果继续往下收敛，我建议你后面直接把 proposal 里的这几块替换掉

需要替换的只有概念层，不是整体框架：

- `固定 role 划分`  
改成  
`dynamic role discovery / online role induction`

- `task-role mapping 表`  
改成  
`role memory bank + task-to-role assignment`

- `人工定义 role prototype`  
改成  
`从历史 task anchor 在线聚合得到 role prototype`

---

如果你愿意，我下一条可以直接帮你写一版“**proposal 中哪些段落该怎么改**”的精简替换稿，但仍然只写思路，不动文件。