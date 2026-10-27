# ACT PPO：final_good_act 探索与 schema 5 奖励

2026-09-30。本文记录 `final_good_act` 探索和 schema 5 完整任务奖励。**当前启动脚本的 S1 默认使用独立 [连续分段奖励（process schema 3）](act_ppo_s1_distance_bands.md)，下文 schema 5 reach 分项不叠加到这个 S1 总奖励；S0 仍保留 schema 5 HOME。** [连续过程 schema 1](act_ppo_s0_s1_dense_reward.md) 和 [距离进步 schema 2](act_ppo_s1_progress_reward.md) 保留作历史参考。这些记录不表示已完成新版本 Gazebo live 调参或证明成功率提升。阶段预算、40%/60% 有效步数目标、完整 episode wave 调度及终止规则见 [S0/S1 训练说明](act_ppo_s0_s1_training.md)。

## 入口与参数优先级

`scripts/finetune_act_ppo_so101.sh` 默认传入 `EXPLORATION_PROFILE=final_good_act`，通过项目 `config/pick_place_reward.json` 使用 schema 5，并选择 `config/act_ppo_stages_s1_progress.json` 中的 S1 替代奖励。探索、完整任务 reward schema 与 stage process reward 独立配置。Python 的 `ActPpoGazeboConfig` 与直接 CLI 默认探索仍为 `stage_log_std`；`PickPlaceRewardConfig()` 默认仍为 schema 4。旧版本由显式配置保留。

本次学习率变更只在 shell 默认值：actor、critic、探索参数均为 2e-4，视觉 backbone 为 2e-5。显式数值环境变量优先；actor/backbone 显式留空会继承 ACT checkpoint。`TARGET_KL=0.02`、2 epochs、clip 0.2、现有整轮回滚和最多 8 次缩小步长的规则未因本 profile 更改。

`final_good_act` 下，旧固定阶段的 `IDLE/MOVING/MANIPULATION/HOME_HOLD_*EXPLORATION_LOG_STD` 及 moving/manipulation 夹爪组数值不决定噪声；`EXPLORATION_PROFILE=stage_log_std` 才使用这些组值。通用 learned log-std 基础值和上下限仍有效，显式学习率环境变量也仍有效。

如果继续使用带有旧 `export` 的启动命令，应替换其中这些行，而不是只修改脚本默认值：

```bash
export REWARD_CONFIG="$PROJECT_ROOT/config/pick_place_reward.json"
export STAGE_CONFIG="$PROJECT_ROOT/config/act_ppo_stages_s1_progress.json"
export EXPLORATION_PROFILE=final_good_act
export ACTOR_LEARNING_RATE=2e-4
export ACTOR_BACKBONE_LEARNING_RATE=2e-5
export CRITIC_LEARNING_RATE=2e-4
export EXPLORATION_LEARNING_RATE=2e-4
export INITIAL_LOG_STD=-3
export MINIMUM_LOG_STD=-9
export MAXIMUM_LOG_STD=-2.5
export GRIPPER_MAXIMUM_LOG_STD=-1.5
```

这只是新 run 的参数片段，不会启动训练。保留其他已选定的运行参数，并以 ACT 权重开启新 PPO run；不能把这些参数直接套在旧 optimizer/critic 的完整 resume 上。已有使用旧 stage JSON 的 run 应继续保留原 JSON，不中途更换奖励。若新 run 仍 export 旧 `act_ppo_stages_dense.json`，它会覆盖脚本默认，需主动换成上面的路径才启用 process schema 3。

## 物理探索尺度

实现位于 `src/so101_capstone/training/act_ppo_exploration_profile.py`。对非 HOME 阶段，距离优先使用 `stage_reach_distance_m`，缺少时使用 `target_distance_m`。下表为完整任务原始分档：

| 当前距离 d | 六关节初始标准差 σ |
|---|---|
| d > 0.15 m | 0.12 rad |
| 0.08 m < d ≤ 0.15 m | 0.08 rad |
| d ≤ 0.08 m | 0.05 rad |

2026-09-30，按新的 S1 训练要求追加阶段专用覆盖：`training_stage=S1` 的非 HOME 阶段在 **d ≤ 0.08 m** 时改用 **0.12 rad**（约 6.88°），是原 0.05 rad 的 2.4 倍；8 cm 外沿用上表。覆盖 APPROACH、GRASP_LIFT、TRANSPORT、PLACE_RELEASE，后两者在正常 S1 流程中通常已触发抓持抬起的阶段结束。GRASP_LIFT 在 4.5 cm 内可能仅因接近而出现，因此不能只放大 APPROACH。S0、所有 HOME 阶段和完整任务模式不受此覆盖影响。

第 j 轴现有物理限位的半跨度为 `h_j = (upper_rad_j - lower_rad_j) / 2`，归一化 Gaussian 初始标准差为 `σ / h_j`，profile 返回 `log(σ / h_j) - initial_log_std`。学习中的物理标准差在触及已有上下限前为 `σ * exp(base_log_std_j - initial_log_std)`。每轴物理初始 σ 相同，但标定跨度不同，所以对应的归一化 log-std 可以不同。

ACT 仍输出绝对位置均值；前五轴的 policy 范围为 [-100, 100]，夹爪为 [0, 100]。PPO 保存并评分裁剪前 Gaussian 样本，随后裁剪执行命令。这些 σ 是裁剪前命令噪声的物理换算，既不是旧项目的增量动作上限，也不是实测关节位移。原动作边界、控制器和 30 Hz 确定性步进保留。

WAIT_HOME、RETURN_HOME、HOME_HOLD 取当前最大关节绝对误差 `e`（rad），按 0.08/0.15 **rad** 分档，再使用：

```text
σ_home = min(分档标准差(e), 0.001 + 0.5 * max(e - HOME位置容差, 0))
```

HOME 目标位置及容差来自正在使用的 reward 配置；当前位置容差为 1°，未修改。容差内的初始 σ 为 0.001 rad，仍受之后学习的尺度影响。默认基础 log-std 为 -3、下限 -9、上限 -2.5，夹爪有效上限 -1.5；配置校验要求上下限能够容纳上述初始尺度，不能容纳时直接报错。无效或缺失的必要测量也报错，不以虚构距离选择噪声。

探索 signature 升级为 schema 2，显式记录 `s1_near_override` 的阶段、距离和尺度，旧完整 PPO 状态因签名变化拒绝恢复；只加载导出的 ACT 权重起新 run 不受影响。现有进程需要重新启动才采用新探索，训练命令无需新增参数。

## Schema 5 的历史过程奖励（完整任务与旧 stage JSON）

实现位于 `src/so101_capstone/rewards/pick_place_reward.py`，数值由项目 reward JSON 决定。它保留现有 PBRS、任务剩余成本、HOME 位置/速度成本等，并追加下列分项。S0 仍使用该奖励；process reward schema 1/2 均在 S1 使用独立替代总奖励，不叠加本节 reach 历史罚分、闭爪 gate、完整任务 PBRS 或完整任务剩余成本。当前 S1 的距离成本与最好距离增益见新奖励文档。完整任务成功仍为 +100、真实任务失败仍为 -10；阶段成功和超时仍按已有 truncation 规则处理，不领取完整任务成功奖金，S1 独立阶段奖金为 +10。

reach 分项只在目标存在、任务处于 APPROACH 或 GRASP_LIFT、尚无有效抓持或进行中的抓持认证时生效。以下逐步成本以一个 30 Hz 步为单位；实际连续成本统一乘 `dt * 30`，新最好改善和一次性奖金不乘这个系数。

| 分项 | 当前规则 |
|---|---|
| 距离新最好 | `100 * (旧最好距离 - 当前距离)`；仅比历史最好更近时发放 |
| 距离停滞 | 本步没有距离新最好时 `-100 * d²`；与距离改善互斥 |
| reach 步成本 | 每步 -0.5 |
| X/Z 对准新最好 | 各自按 `floor(误差角度 / 0.5°)` 计算档位；刷新历史最好时，每改善一档 +0.1 |
| X/Z 对准退步 | 各自比历史最好多出的每一档，每步 -0.5 |
| 首次 Z 对准 | 首次达到 Z 误差 ≤3° 时 +5 |
| 近距离错误对准 | d ≤0.06 m 且 Z 误差 >10° 时，每步 -5 |
| 距离阈值 | 首次从阈值外跨入 0.22、0.19、0.15、0.12、0.09、0.06、0.03、0.02 m 内，依次 +0.1 至 +0.8；单步跨入至少两档，所获阈值奖金之和乘 1.5 |
| 夹爪新最好 | 当前开口/闭合评分分别维护历史；有效权重乘 `5 * (当前评分 - 旧最好评分)` |
| 夹爪退步 | 有效权重乘 `-5 * max(最好评分 - 当前评分, 0)²` |

角度继续从当前 FK 几何计算：X 轴误差是局部 X 轴相对世界水平面的倾角；Z 误差取当前 4 cm 参考射线对准误差与向下轴误差的较大值。没有搬入旧控制间隔、20 cm 方向回退点或旧夹爪几何。夹爪继续使用现有 `approach_opening_score`、`grasp_closure_score` 和 `grasp_close_gate`，开口和闭合权重分别为 `1-gate`、`gate`。首次进入有效闭合区先建立闭合评分基线，远处已经闭爪不会阻止随后真正靠近时的闭合改善；重新开闭或切换 gate 不清空历史。

HOME 的历史误差为六关节容差外误差的归一化 RMS：

```text
E = RMS(max(abs(q - q_home) - HOME位置容差, 0) / home_position_scales_rad)
```

各轴当前尺度为 0.5 rad。创造新最好时加 `10 * (旧最好 E - E)`，否则每步加 `-0.5 * E²`；比历史最好退步时另加 `-2 * max(E - 最好 E, 0)²`；尚未同时满足 HOME 位置和静止条件时，每步另加 -0.5。等待 HOME 与任务完成后回 HOME 各自记录历史。进入当前位置容差并保持速度合格后，E 为零，新增停滞、退步和未稳定成本也为零，不要求策略为刷“改善”离开安全保持区。

只有新鲜、有效的运行中证据才能更新最好值或发放正向过程奖励。Reset 清除全部历史，初始有效状态及稍后首次有效状态只建立基线；初始已经满足的距离阈值和 Z 对准不会补发奖金。重复/无效证据不更新最好值、不发改善和一次性奖励，持续成本沿用缓存有效值。每个 episode 的阈值及对准奖金只领取一次。

## Critic 与恢复

schema 5 通过 `critic_history_context()` 暴露历史最好值、最近有效测量、阈值已领取标记、开闭夹爪及两组 HOME 历史等 33 维状态。拼接到原 93 维 task context 后为 126 维；process schema 1 追加 12 维得到 138 维，process schema 2/3 追加 14 维得到 140 维。S0 追加部分为零，actor 观测输入保持原状。这些数值是 critic **任务 context** 维度，不是整个 critic 所有输入的总维度。process schema 2/3 还按 S0/S1 分组标准化 actor advantage，schema 1 保留原全局方式。

schema 3/4 的奖励历史 context 为空；未启用新 stage process reward 时，其原 93 维任务 context、奖励语义和配置 hash 保留。schema 5 的 reward hash、context 维度/版本、新探索 profile 的物理分档、HOME 规则和标定跨度，以及启用的 stage process reward 配置进入恢复校验。旧完整 PPO 状态不能直接用于这套新配置；应以导出的 ACT 权重开始新 PPO run。仅切换探索 profile 不会自动切换 reward schema 或阶段奖励。

## 验证范围与激励风险

静态检查和单元测试可以核对分档、数学公式、历史清理、重复奖励防护及恢复拒绝边界，不能证明训练已收敛、HOME 已稳定或真实抓放成功率提高。本配置尚未完成新的 Gazebo live 调参与硬件验证。

schema 5 的旧 reach 公式已确认存在奖励尺度导致的激励风险。例如只看 d=0.2 m 的距离停滞及 reach 步成本，就有每个 30 Hz 步 -4.5；以 gamma=0.997 计算，停滞 30 步的折现成本约 -129.29，尚未计入其他分项。保留物理失败 -10 时，某些失败轨迹可能比继续尝试再成功回报更高。这是源码公式能构造出的回报排序问题，不是已经观测到策略主动推物块下桌；具体轨迹可达性和频率尚未 live 验证。该旧公式仍供完整任务和旧 stage JSON 使用；新的默认 S1 独立替换奖励，避免继续叠加这些历史逐步成本，没有另行调大失败罚分或改变 terminal 语义。新方案效果仍须比较实际成功、停滞和失败轨迹的累计回报。

Actor 初始学习率由脚本原先的 3e-8 提到 2e-4，幅度变化很大。现有 KL 回退可能多次拒绝更新并降低有效学习率，不能把更大的起始值解释为更快收敛或更高成功率。日志记录的实际有效学习率、接受次数和 KL 比仅看启动值更有意义。

历史记录：引入本次独立 S1 stage process reward **之前**，指定 conda `lerobot` 环境下相关测试曾有 1176 项通过，启动脚本 `bash -n` 和 `git diff --check` 通过；这个历史结果不代表当前新奖励已经通过相同回归。本次新奖励的实际检查结果以当次验证报告为准，未进行 Gazebo live 调参或硬件验证。
