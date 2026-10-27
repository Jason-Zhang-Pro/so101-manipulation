# S1：8 cm 内连续分段奖励

2026-10-01 更新：以下 schema 3 配置原样保留。新默认 [schema 4](act_ppo_s1_lift_approach.md) 在完成8 cm抬升后，继续使用这些接近区间，并增加从上方接近的高度约束。

2026-09-30。当前 `config/act_ppo_stages_s1_progress.json` 使用 process reward schema 3，在 schema 2 的首次最好距离奖励上增加下列细分区间，并将时间成本提高十倍。旧 schema 2 原样保存在 `config/act_ppo_stages_s1_progress_v2.json`，与 `act_ppo_20260930_132641_from_update_000057` 的保存配置一致。

## 距离分段

距离指有效、时间对齐证据中的抓取点到目标中心的三维距离。每段 5 mm，越靠近目标奖励越高：

| 从较远端到较近端 | 完整走过该段的新增奖励 |
|---|---:|
| 8.0 → 7.5 cm | +1 |
| 7.5 → 7.0 cm | +1.5 |
| 7.0 → 6.5 cm | +2 |
| 6.5 → 6.0 cm | +2.5 |
| 6.0 → 5.5 cm | +3 |
| 5.5 → 5.0 cm | +3.5 |
| 5.0 → 4.5 cm | +4 |
| 4.5 → 4.0 cm | +4.5 |
| 4.0 → 3.5 cm | +5 |
| 3.5 → 3.0 cm | +6 |
| 3.0 → 2.5 cm | +7 |
| 2.5 → 2.0 cm | +8 |

分段新增奖励合计最多 **48**。它与原有 `s1_best_distance`、姿态势函数等奖励叠加，而非替代它们。

每段按进度线性发放，不必先碰到边界：例如刷新本集最好距离，从 7.2 cm 到 7.1 cm，该分段项增加 `1.5 × 1/5 = 0.3`。本步总 reward 仍是所有分项之和，不能把这笔增益当成净奖励。

奖励只支付超过本集历史最好距离的新增进度。停住、退回后重访已到达距离均不再领取；初始化已经位于区间内时，已完成进度也不补发。一步跨多个区间会分别结算新增部分。2 cm 以下不再增加距离分段奖励，成功仍需原有 XY、高度、方向、开口和保持时间全部达标，或可靠抓持抬起的已有例外。

## 证据与记录

沿用 schema 2 的同一历史最好距离质量和前一步资格，共 14 维 process critic context。连续、有效、无接触的源区测量才支付新奖励；重复测量、证据缺口、接触或真实 terminal 步不发分段奖励。新测量中已经到达的更近位置仍会推进历史记录，恢复后不补领跨无效区间的进步。

日志分项为 `s1_distance_band_01` 至 `s1_distance_band_12`，顺序与上表一致。各分项本身进入总 reward，没有另外重复加入合计项。

## 时间成本和结束条件

`s1_time_cost_per_s` 从 **0.03** 改为 **0.3**。30 Hz 下每步约 **−0.01**；实际 33/34 ms 步分别为 −0.0099/−0.0102，10 秒总计 −3。成本按真实仿真时间累计，避免改变确定性步进或浮点时间差带来的总额偏差。

`s1_remaining_cost_per_s=0.3`、`s1_distance_cost_per_s=0.4`、`s1_success_bonus=10` 和完整任务 failure cost=10 保持原值。三个持续成本在默认 10 秒内合计绝对值最多 10；其折现值更小。这仅是持续成本的上界，不能保证任意成功、失败、停留轨迹的总回报排序，因为总回报还有分段增益、势函数、成功奖励和 bootstrap。

S1 成功仍提前结束当前 episode；10 秒未达标仍为 TIMEOUT。上述奖励改动未修改 S0、探索、学习率、KL、30 Hz、关节限位、相机几何或世界生命周期。细分奖励不代表已经验证模型能突破停滞，需要下一轮训练和关闭探索的评估确认。

后续按用户要求单独增强了 S1 近距离探索：非 HOME 阶段距离 ≤8 cm 时六轴初始标准差从 0.05 rad 增至 0.12 rad，HOME 不变。该改动属于探索 profile schema 2，不改变本文件的 reward schema 3 或成功阈值，详见 [当前探索与学习起点](act_ppo_s0_s1_training.md#当前探索与学习起点)。

## 配置与恢复

原启动命令中的 `STAGE_CONFIG="$PROJECT_ROOT/config/act_ppo_stages_s1_progress.json"` 将读取新配置。显式指定旧 run 的 `resolved_stages.json` 则继续旧奖励。

schema 3 的 process context 长度仍为 14，但版本为 3；完整 schema-5 任务的 critic context 仍为 140 维、版本变为 6。S0/S1 actor advantage 继续按阶段分别标准化。旧 schema 1/2 的配置序列化、哈希与各自恢复契约保持不变；新旧奖励即使维度相同，也必须由完整 execution signature 拒绝混用 PPO 状态。

可使用 update 22 的 `pretrained_model` 作为 ACT 起点，新建 critic、optimizer 和奖励状态。不能把旧完整 PPO training state 直接 resume 到新奖励。

## 本次验证

2026-09-30，在 conda `lerobot`（Python 3.10.18、torch 2.7.1+cu128、pytest 8.4.2）中加载 ROS 2 Humble 与当前 workspace 后执行：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/src:$PWD/lerobot/src${PYTHONPATH:+:$PYTHONPATH}" \
/home/tiger/miniconda3/envs/lerobot/bin/python -m pytest -q -p no:cacheprovider \
 tests/unit/test_stage_process*.py tests/unit/test_act_ppo*.py \
 tests/unit/test_pick_place*.py tests/unit/test_training_stages.py \
 tests/unit/test_reward_return_ordering.py tests/unit/test_reach_env_core.py
```

结果：**1377 passed in 10.26s**。覆盖分段连续增益、总额上限、历史最好记录、重复/缺口/接触证据、成功和失败语义、S0 保持原行为，以及相同 critic 维度下新旧完整 PPO 状态仍由 execution signature 拒绝混用。合成轨迹验证的是相同接近前缀下及时成功、延迟成功、超时、失败的回报排序，不代表所有可能轨迹。

`git diff --check` 与启动脚本 `bash -n` 通过。本次未启动训练、Gazebo 或真实硬件，尚无新奖励的实际训练效果结论。
