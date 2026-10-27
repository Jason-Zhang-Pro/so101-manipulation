# S0/S1 连续过程奖励（历史 process schema 1）

本文保留 2026-09-29 的 **process reward schema 1** 配置 `config/act_ppo_stages_dense.json`，用于解释与复现旧 run。当前 shell 默认使用 `config/act_ppo_stages_s1_lift_approach.json` 的 process schema 4，详见 [S1 抬升与接近](act_ppo_s1_lift_approach.md)。下文的势函数上限 100、12 维阶段 context/总 138 维及回归结果均属于历史 schema 1，不能当作新默认说明。

Schema 1 中，S0 保留原 schema 5 HOME 奖励及原有其他分项；S1 使用独立 `src/so101_capstone/rewards/stage_process_reward.py` 的 `StageReachReward`，完全替换完整任务奖励总和。当时没有改变探索 profile、学习率、KL、30 Hz 步进、动作映射、reset、相机几何或阶段成功阈值。

更早的 `config/act_ppo_stages.json` 保持兼容，不包含 `process_reward`；已有 run 应继续使用其原 JSON。切换到不同 process schema 是新的训练目标，需要以导出的 ACT 权重开始新 PPO run，不能把旧完整 PPO 状态直接切换到新奖励。

## 为什么拆出 S1

S1 的目标是到达可抓取的开口姿态并稳定保持，而完整任务的 schema 5 reach 奖励还包含历史最好距离、角度退步、近处闭爪等逻辑。旧公式在 d=0.2 m 停滞时，仅距离与步成本就是 -4.5/步；gamma=0.997 下停滞 30 步折现约 -129.29，真实失败项为 -10。这已经构成可从公式确认的激励风险，但没有观测到 rollout 主动利用该风险。

新 S1 不再叠加旧 `reach_*` 历史改善或惩罚、闭爪 gate、完整任务 PBRS、完整任务 remaining-task cost 或其他完整任务过程成本。它使用当前相邻状态的有界连续势函数差，保留阶段认证，并增加较小时间成本、独立的 S1 未完成程度成本和一次阶段通过奖金。S0 未采用这套替代公式，其完整 HOME 奖励保持原样。

## 势函数与尺度

九个分量按下表的固定顺序组成 `s1_potential_weights`。每项评分位于 [0, 1]，乘权重后求和为 Φ，默认 `0 ≤ Φ ≤ 100`。尺度位于新 stage JSON 的嵌套 `process_reward`，不会改写阶段通过阈值。

普通连续评分为 `quality = scale / (scale + excess)`。距离、XY 和角度的 excess 是超过允许容差的部分；高度和夹爪开口的 excess 是离允许区间的距离。已在容差或区间内时 excess=0、quality=1，不奖励继续向区间某个点挤压。

| 分量 | 权重 | excess / 评分来源 | 当前尺度 |
|---|---:|---|---|
| `distance_coarse` | 20 | `max(距离 - 0.02m, 0)` | 0.10 m |
| `distance_fine` | 20 | 同一距离误差，保留更细的近距离梯度 | 0.02 m |
| `xy` | 15 | `max(XY距离 - 0.01m, 0)` | 0.03 m |
| `height` | 10 | dz 离 [0, 0.02m] 的距离，再乘 XY 软门控 | 0.02 m |
| `bearing` | 5 | `max(4cm参考射线对准误差 - 20°, 0)`，再乘 XY/高度软门控 | 0.35 rad |
| `axis` | 5 | `max(局部Z轴与向下轴误差 - 20°, 0)`，再乘 XY/高度软门控 | 0.35 rad |
| `opening` | 10 | 相对 HOME 开口离 [0.45, 0.90rad] 的距离，再乘 XY/高度软门控 | 0.20 rad |
| `combined` | 10 | fine、XY、高度、bearing、axis、opening 六个原始 quality 的几何平均 | — |
| `hold` | 5 | `min(已认证连续reach保持时间 / 0.15s, 1)` | — |

粗、细距离使用相同的通过容差，但尺度不同。其他量固定、且尚未进入对应容差时，任意细微改善都会提高相应连续评分；已达标的单项会饱和。多个指标同时变化时，软门控或其他分项的退步可以抵消改善，不能保证每次动作总奖励为正。

软门控为 `g(q) = 0.25 + 0.75*q`：height 乘 `g(q_xy)`，bearing/axis/opening 各乘 `g(q_xy)*g(q_height)`。因此相邻目标之间有优先顺序，但有效源区状态下，远离 XY/高度容差不会把后续项完全关闭。`combined` 使用六个未乘门控的原始评分，不把 height 等门控再乘一遍。

上述分量使用当前 `StageEvaluator` 测量：grasp 参考点、4 cm 方向参考射线、向下轴与开口定义均沿用 capstone。没有移植旧控制间隔、20 cm 参考点、增量动作上限或旧夹爪几何。

## 有效场景与通过条件

普通 reach 势函数要求目标仍在源区、受到源平面支撑、静止且不在目标放置区，同时没有左右接触、有效抓持、正在进行的抓持认证或已经抬升事件。不满足时九项势函数均为零，不能通过把目标推离源区或提前闭爪接触而继续积累普通 reach 势能。

已由真实同时接触、抓持及抬升时长等证据认证的 `stage_downstream_grasp_lift` 是例外：九项取满权重，作为比开爪 reach 更强的阶段通过证据。仅闭爪或一次接触不满足这个例外。

S1 普通通过条件仍为距离 ≤2 cm、XY ≤1 cm、dz 在 0–2 cm、方向误差 ≤20°、相对 HOME 开口 0.45–0.90 rad，连同上述场景条件同时持续 0.15 秒；最长预算仍为 10 仿真秒。S0 仍为 5 秒观察窗、末尾连续 HOME 保持 3 秒，位置/速度阈值不变。40%/60% 有效步数调度与完整 episode wave 规则也不变。

## 单步总奖励与日志

新鲜有效证据下，先记录每一项加权势函数的真实差值：

```text
s1_<分量>_progress = Φ分量(next) - Φ分量(previous)
s1_discount_correction = (gamma - 1) * Φ(next)
s1_time = -0.03 * dt_s
s1_quality = sum(前八项 Φ(next)) / 95
s1_remaining = -0.3 * dt_s * max(1 - s1_quality, 0)
s1_success = 10，仅新鲜证据认证的 S1 阶段通过步领取一次
```

各 progress 与 discount correction 相加恰为 `gamma * Φ(next) - Φ(previous)`。默认 gamma 仍为 0.997，时间成本按仿真秒计算，30 Hz 正常一步约 -0.001。`hold` 已是九项势函数中的 5 分部分，没有另外每步加一次 hold 奖励。

`s1_remaining_cost_per_s=0.3` 是这个独立 S1 目标的配置；quality 使用前八项的加权和，除以它们的权重总和（默认 95），不包含 hold。越接近有效目标姿态，未完成程度成本越小，因此未成功的部分完成状态也受到目标偏好，而不只是改变 PBRS 的反馈时机。quality 位于 [0, 1]，所以 `s1_time+s1_remaining` 的成本幅度最多为每秒 0.33，按默认 10 秒计算最多 3.3。这个界限只包含这两项持续成本，不包括折扣势函数、证据保护或 terminal 势函数归零损失。

进步日志区分距离粗/细、XY、高度、bearing、axis、opening、combined 和 hold。哪一项改善可直接从对应 `s1_*_progress` 查看；总奖励还包含折扣修正、时间成本、未完成程度成本和证据保护，所以单个 progress 为正不代表总和为正。上式中的 `s1_quality` 是说明用中间量，实际奖励分项记录为 `s1_remaining`。

在一致有效证据、相同起终势函数的闭合往返中，T 步折现 shaping 总和为 `(gamma^T - 1) * Φ起点 ≤ 0`，加上非正的时间和未完成程度成本更低。退步再恢复可以获得“恢复”这一步的正 progress，但不能仅靠往返重复刷出正的折现 shaping 总回报。这是公式性质，不是仿真收敛保证。

阶段 `SUCCESS` 或 `TIMEOUT` 仍是 `truncated=True, terminated=False`，保留末状态势函数与价值 bootstrap，GAE 不跨 reset。S1 的 +10 是阶段奖金，不是完整任务 +100，也不增加完整任务成功率。真实任务失败保留 terminal 边界，`Φ(next)=0`，另加 `failure=-10`；失败步总回报还包含势函数归零造成的损失，因此并非总和固定 -10。

## 证据与 Reset

Reset 用有效初始快照建立九项势函数基线，不发首次姿态奖金，并清除本段历史。后续控制仿真时间、task identity 和目标 identity 必须连续；无效或过时传感器证据报错，按现有采集错误路径处理，不伪装成策略任务失败。

重复证据保持缓存势函数不变，各项 progress 为零，不增加 hold 或阶段奖金；折扣修正、时间和未完成程度成本仍按当前步计算。证据时间出现超过 `max_evidence_gap_s`（当前 0.12 秒）的缺口时，用新的有效端点重新建立缓存，但通过 `s1_evidence_guard = -sum(max(各progress, 0))` 抵消跨缺口的全部正增量，负增量保留。日志中可能仍显示正 progress，必须结合 evidence guard 判断实际奖励。

## Critic、配置与恢复

stage reward 的缓存与已用时间影响后续回报，因此历史 schema 1 的 critic 追加 12 维：active、initialized、九个加权缓存势函数，以及 `min(stage_elapsed_s / s1_budget_s, 1)`。S0 的追加部分全为零。schema 5 原有 93 维任务状态与 33 维奖励历史保留，因此这一版 task context 合计 `93+33+12=138`；S1 不执行旧 reward 的 step，其原 33 维保持 reset 基线。ACT actor 输入不增加这些状态。

新 stage JSON 仅比旧 JSON 多出嵌套 `process_reward`；旧 JSON 的缺省字段在序列化时仍省略，保留原 hash 和恢复语义。`resolved_stages.json` 保存实际配置，新执行签名包含 stage process reward 配置、hash、context schema/维度与 S0 原奖励/S1 替代奖励的边界标识。修改这些值需要新 run，旧完整 PPO optimizer/critic 状态不能恢复到新签名。

仅在复现历史 schema 1 时使用下面的 stage JSON；已有显式 `STAGE_CONFIG` 环境变量会覆盖脚本的新默认：

```bash
export STAGE_CONFIG="$PROJECT_ROOT/config/act_ppo_stages_dense.json"
```

已有 run 应保留旧 JSON 和原参数；转到当前 schema 2 应按 [新配置说明](act_ppo_s1_progress_reward.md) 另开 PPO run。选择 `TRAINING_STAGES=full` 不使用 S1 替代模块。`final_good_act` 探索、actor/critic/探索学习率 2e-4、backbone 2e-5、KL 0.02 及回滚机制在两版之间保持。

## 验证边界

当前阶段阈值仍标记 `candidate_unvalidated`。离线公式与测试能够检查分项、门控、折现回报、重复证据、终止边界和恢复兼容，不能证明实际 reach 姿态最优、HOME 稳定性或抓放成功率提高。本次尚未开展 Gazebo live 调参、正式训练或真实硬件验证；具体测试结果以本次实际执行报告为准，不沿用旧奖励的历史通过数。

2026-09-29 本次实际离线回归：加载 ROS 2 Humble 后，在 conda `lerobot` 解释器中使用 `PYTHONNOUSERSITE=1`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、`PYTHONDONTWRITEBYTECODE=1` 及项目 `src`/`lerobot/src` 的 `PYTHONPATH`，执行以下范围，结果为 **1253 passed**：

```text
tests/unit/test_stage_process*.py
tests/unit/test_act_ppo*.py
tests/unit/test_pick_place*.py
tests/unit/test_training_stages.py
tests/unit/test_reward_return_ordering.py
tests/unit/test_reach_env_core.py
```

启动脚本 `bash -n` 与 `git diff --check` 通过。对已有 `act_ppo_20260929_203125_from_checkpoints` 的保存配置及原图像增强开关进行只读重算后，旧配置的序列化执行签名完全一致，critic context 仍为 126 维；该检查没有连接或改变正在运行的训练。
