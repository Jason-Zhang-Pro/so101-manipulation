# S1 距离进步奖励与阶段优势值标准化

2026-09-30。本文记录 **process reward schema 2**，其配置现保存在 `config/act_ppo_stages_s1_progress_v2.json`。schema 3 保存在 `act_ppo_stages_s1_progress.json`；当前默认已升级为 [schema 4：S1抬升与接近](act_ppo_s1_lift_approach.md)。下文的参数和检查结果属于 schema 2。这里的 process schema 与 `pick_place_reward.json` 的完整任务 reward schema 5 是不同版本号。

本次局部调整降低九项姿态势函数的尺度，增加不与 PBRS 抵消的距离目标，并在联合训练时分开标准化 S0/S1 的 actor advantage。它针对目前发现的奖励与样本尺度风险，尚未通过新版本 Gazebo live 训练验证效果，不能将它视为已确定的停滞唯一原因或成功方案。

## 研究结论与适用范围

Ng、Harada、Russell 的原始论文讨论 `R'=R+gamma*Φ(s')-Φ(s)`：满足相应假设时，势函数塑形保留基础任务的最优策略。它能改变学习信号的时序，不能替代基础任务里欠缺的目标偏好。当前九项 progress 与 discount correction 正是这种形式；只增大它们，并不等价于让“更接近物体”在基础回报中更有价值。有限采样、近似 critic 和本项目的阶段截断还需要各自检查，不能直接引用最优策略定理来保证 PPO 收敛。[原始论文：Policy invariance under reward transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)

Trott 等人的研究展示，距离目标虽然具有正确的全局最优点，仍可能因可达状态、动力学与学习分布而形成局部最优。对本项目，距离、XY、高度、方向及开口经过加权和门控后，单项改善可以被别项退步抵消；这是需要用本地轨迹核对的风险，不是该论文已经证明 SO-101 的具体原因。论文的实验包括导航与构建任务，其 Sibling Rivalry 算法没有引入本项目；本次只保留“不能凭密集距离奖励排除局部停滞”的结论。[原始论文：Keeping Your Distance](https://papers.neurips.cc/paper/9225-keeping-your-distance-solving-sparse-reward-tasks-using-self-balancing-shaped-rewards.pdf)

Hessel 等人的多任务 PopArt 研究指出，任务奖励的尺度和密度不同会使共享学习器偏向部分任务。这里据本地 S0/S1 的 GAE 分布选择较小范围的分阶段 actor advantage 标准化，并未实现 PopArt 的价值网络或目标归一化；该论文支持检查跨任务尺度问题，不直接证明本次具体归一化方案有效。[原始论文：Multi-Task Deep Reinforcement Learning with PopArt](https://ojs.aaai.org/index.php/AAAI/article/view/4266)

Raffin、Kober、Stulp 研究指出，高频逐步独立 Gaussian 探索可能产生抖动，并被机器人物理低通效应相消；gSDE 通过状态相关噪声和定期重新采样改善若干机器人任务。当前 capstone 确实每个 30 Hz 步独立采噪声，但论文不能证明当前确定性均值停滞就是探索导致，更不能保证噪声相关化适用于 ACT。此次未引入 gSDE、OU 噪声、动作重复或控制频率变化；若后续验证相关噪声，必须同步处理采样、PPO 概率、KL 与恢复状态，不能只把同一噪声复制几步。[原始论文：Smooth Exploration for Robotic Reinforcement Learning](https://arxiv.org/pdf/2005.05719)

## 已观察到的阶段尺度差异

已有 schema 1 run `act_ppo_20260930_014209_from_checkpoints` 的 update 57，使用 episode 616–626 保存的旧奖励和旧 value 离线重算 raw GAE，再调用当前分组标准化实现。900 条 S0 和 1500 条 S1 transition 的结果如下，原始数值见 [离线信用分配报告](../codex_workspace/reports/2026-09-30_s1_credit_audit.json)：

| 阶段 | raw GAE 均值 | raw GAE 标准差 | 原全局标准化正/负数 | 新分组标准化正/负数 |
|---|---:|---:|---:|---:|
| S0 | 1.76448 | 0.59315 | 847 / 53 | 498 / 402 |
| S1 | 0.0161137 | 0.102754 | 0 / 1500 | 756 / 744 |

这只展示同一旧 batch 在两种方式下的信用分配变化，没有重训策略、执行新动作或证明行为改善。S0/S1 的价值估计误差或回报尺度会一起影响混合 batch 的中心与尺度。负 advantage 表示相对当前 baseline 较差，本身不等于错误；减均值后符号变化也是标准化的正常结果。这些数据支持单独检查跨阶段尺度，不能单凭“全负”断言 PPO 算错、所有 S1 动作都被错误惩罚，或确认它是停滞唯一原因。

该旧 run 截至本次检查，285 个 S1 episode 全部 TIMEOUT，没有一个 episode 的最小距离达到 2 cm。首 25 集与末 25 集的平均最小距离从 6.32 cm 变为 8.63 cm，而末状态 dz 从 4.81 cm 变为 0.571 cm、opening 从 1.516 rad 变为 0.940 rad、方向误差从 49.2° 变为 45.8°。这说明若干姿态指标改善时，主要距离指标仍可能退步，并非完全没有发生策略变化。

抽查 10 集共 3000 步，势函数未出现零值，`s1_evidence_guard` 为零；episode 626 的 coarse/fine/XY 净变化为负，opening/axis 净变化为正。该样本支持检查多目标的抵消关系，也使“这些步一直被资格门控清零”缺少证据；它不能排除其他 episode 的证据、动力学、探索或价值拟合问题。

## Process schema 2 的奖励

实现仍在 `src/so101_capstone/rewards/stage_process_reward.py`。距离粗/细、XY、高度、bearing、axis、opening、combined、hold 九项评分及软门控沿用 [schema 1](act_ppo_s0_s1_dense_reward.md)，各尺度和成功容差不改。变化如下：

| 配置项 | 新默认 | 行为 |
|---|---:|---|
| `s1_shaping_scale` | 0.1 | 九项加权势函数整体缩小，总势上限由 100 降为 10；progress 和 discount correction 同比例缩小 |
| `s1_distance_cost_per_s` | 0.4 | 增加独立的距离未完成成本 |
| `s1_best_distance_bonus` | 8.0 | 首次刷新本 episode 最好距离质量时给有界增益 |

使用当前 grasp 到目标的距离 d（米）：

```text
q_d = 0.1 / (0.1 + max(d - 0.02, 0))
s1_distance_remaining = -0.4 * dt_s * (1 - q_d)
s1_best_distance = 8 * max(q_d - 历史最好 q_d, 0)
```

距离达到现有 2 cm 容差后 q_d=1，不要求进一步压向物体中心。新距离成本独立于开口/方向分数，单独摆好开口或角度无法消除离目标较远的成本。目标失去源区、静止、支撑或无接触资格时，距离成本按 q_d=0 计算；真实认证的下游抓持抬升保留已有例外。重复证据使用缓存距离质量，不凭变化但未更新的测量获得成本改善。

最好距离增益有严格证据边界：reset 初始化 baseline 不领奖；只有当前新鲜、前一步具有有效资格、当前仍具有有效资格且非 terminal 时，才支付新增部分。新的证据端点遇到接触或长缺口时，仍可提高 baseline，但不支付跨缺口/无资格进步，恢复后不补领。没有刷新历史最好就不再发奖；从差状态恢复到已达到过的位置不能重复领同一分数。单 episode 该项总奖励最多 `8*(1-初始最好 q_d)`，不超过 8。

该 best bonus 与独立距离成本是**新的非势函数任务偏好**。一次往返如果首次到达了新最好位置，可能取得正的新增奖励；相同往返不会持续刷取。这不属于“整个奖励保留原最优策略”的 PBRS 保证，也不代表所有可能的奖励利用方式已排除。

其余 S1 分项保持：

- `s1_time=-0.03*dt_s`。
- `s1_remaining=-0.3*dt_s*(1-quality)`，quality 仍是前八项未缩放评分的加权平均，不含 hold；实现用缩放后势函数除以对应缩放权重之和还原，未被 0.1 再次缩小。
- 新鲜证据认证 S1 通过时一次 `s1_success=+10`，不增加完整任务成功率。
- 真实失败 `failure=-10` 且 terminal 势函数为零；总失败步回报还包含势函数损失及其他成本。
- stage success/timeout 保留 truncation 与价值 bootstrap；证据缺口的正势函数增量仍由 `s1_evidence_guard` 抵消。

三个持续成本的幅度每秒最多 `0.03+0.3+0.4=0.73`，按默认 10 秒预算计算最多 7.3；这个界限不包含 PBRS、证据保护、阶段/失败奖金。它也不保证任意成功轨迹都优于任意失败轨迹，仍需结合实际折现总回报和物理可达性判断。

## S0/S1 分组 advantage 标准化

`training/act_ppo_gazebo.py` 对 process schema 2 自动设置 `advantage_group_context_index`；`training/act_ppo.py` 从 critic task context 的 stage active 位区分 S0=0、S1=1。每组只使用 `actor_update_enabled=True` 的 transition，分别计算 raw GAE 的均值与标准差，再将其 actor advantage 标准化。组内少于 2 条或标准差不大于 `1e-8` 时，该组 actor advantage 置零；value-only 样本不进入 actor 统计。

GAE 的计算、episode 截断、critic 的 `return=raw_GAE+V` 不变；分组不复制样本、不改变每次 update 的阶段步数比例。联合 S0/S1 因而各自按组内相对质量提供 actor 信号。它会改变两个阶段对共享 actor 的相对尺度，是明确的训练选择，不是证明原全局标准化在数学上非法。单阶段时只有对应非空组；旧配置继续使用原全局标准化。

诊断提供 group 0/1 的 actor/value-only 数量、raw 均值/标准差与正负比例、normalized 均值/标准差与正负比例，以及退化组标记。应联合观察这些数据、critic 拟合、实际 KL/接受更新和确定性评估，而不是以某组是否有正 advantage 单独判断学会任务。

## Context、恢复与启动

process schema 2 在原 12 维 stage reward context 上增加最好距离质量和前一步有效资格，合计 14 维，context schema 为 2。当前完整任务 reward schema 5 的 critic task context 是 `93+33+14=140` 维；S0 追加 14 个零。ACT actor 的输入未增加这些奖励历史。

旧 `act_ppo_stages.json` 和 `act_ppo_stages_dense.json` 保留旧奖励、context 及标准化配置。新 reward 配置/hash、140 维 critic 和分组标准化策略进入恢复校验；旧完整 PPO checkpoint 不能 resume 到新配置。使用导出的 ACT 权重新建 PPO run，重新初始化 critic、optimizer 与奖励历史；已有 run 若继续原目标，应保持原 JSON。

shell 启动器当前默认选择 schema 3。若要明确使用本文的 process schema 2，需要设置：

```bash
export STAGE_CONFIG="$PROJECT_ROOT/config/act_ppo_stages_s1_progress_v2.json"
```

这只是配置片段，不启动训练。`TRAINING_STAGES=full` 不使用这套阶段奖励。S0/S1 的 5/10 秒预算、3/0.15 秒保持时间、40%/60% 步数比例及各成功阈值保持。`final_good_act` 的独立 Gaussian 探索、脚本 actor/critic/探索学习率 2e-4、backbone 2e-5、KL 0.02 与回滚策略、30 Hz 步进、模型/控制器/reset/相机几何均未在本轮调整。

## 已执行的验证

2026-09-30，使用 conda `lerobot`，source ROS Humble，并设置 `PYTHONNOUSERSITE=1`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、`PYTHONDONTWRITEBYTECODE=1` 和项目/LeRobot 的 `PYTHONPATH`，执行：

```bash
python -m pytest -q -p no:cacheprovider \
  tests/unit/test_stage_process*.py \
  tests/unit/test_act_ppo*.py \
  tests/unit/test_pick_place*.py \
  tests/unit/test_training_stages.py \
  tests/unit/test_reward_return_ordering.py \
  tests/unit/test_reach_env_core.py
```

结果 **1323 passed in 11.74s**。覆盖毫米级距离信用、首次奖励上限、停留/往返/缺证据/接触恢复、真实失败与阶段截断、S0 原分项与通过边界、两版 critic context/恢复隔离，以及分组 advantage 对 actor/value-only/退化组的处理。`bash -n scripts/finetune_act_ppo_so101.sh` 与 `git diff --check` 通过。

另按旧 run 保存的配置重建其实际多世界 execution signature（含原图像增强环境变量），与 `act_ppo_20260930_014209_from_checkpoints/config.json` 中保存的旧签名完全一致。该检查只验证旧目标的配置契约，没有载入训练状态或恢复训练。

## 尚待验证

本次没有执行 process schema 2 的新 live 训练或成功率对照。新最好奖励、尺度调整与分组标准化各自的贡献仍需区分；已有记录中肘关节均值越界/执行饱和对停滞的贡献也仍待核验，本轮未改变关节限位或以放宽限位绕过问题。离线测试通过不代表机械臂已突破 8.54 cm 停滞，原 update 57 模型权重也没有改变。
