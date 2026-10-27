# 从不准确 reach 学习完整抓取放置

这版按「无 target 基本会 HOME，有 target 只能粗略 reach」设计。PPO 直接更新 ACT；部署仍使用普通 ACT 的图像、关节输入及绝对位置输出，五个手臂轴为 `[-100, 100]`，夹爪为 `[0, 100]`。课程、任务状态、critic 和随机探索只在训练侧使用。

## 奖励与成功条件

新训练默认奖励配置是 schema 4，仍支持显式 schema 3，成功定义保持版本 2。入口名 `pick_place_v2` 保留，用于区别更早的 legacy reward。

- 接近使用粗、细两种指数距离尺度，远处和最后几厘米都有进展信号。进入原 4.5 cm 抓取候选区不会失去几何信号，也不会自动获得阶段奖金。
- 朝向同时考虑目标相对方向和独立的工具轴方向。默认期望 FK 抓取参考系的 local +Z 朝世界向下；它是可配置的桌面抓取候选，不是指尖接触认证。方块中心与末端重合，也不能把侧向或倒置算成对齐。
- 远处鼓励合适开口；只有有效目标、近距离且朝向合适，才在 2 cm 到 8 mm 之间连续转向闭合引导。闭爪自身不计抓取成功。真实双侧接触的持续证据逐步接管到抓取和抬升势能。
- 回 HOME 使用连续的多关节距离分数，修复远离 HOME 时的奖励平台；运输距离和下降到支撑面的高度同样保留远距离信号。
- 无目标和 HOME 保持仍惩罚位置误差、实际速度；抓取、抬升、有效搬运、稳定放置、松爪分离和连续 HOME 保持的硬条件均保留。

势能塑形仍为 `gamma * Phi(next) - Phi(current)`。真正终止时势能为零，时间截断保留 bootstrap。schema 4 另加真实的任务代价，明确改变优化目标；精确 reach 仍只是一项训练进度指标，完整成功必须完成全部流程。

schema 4 将开口偏差与闭合程度的线性截断分数改为指数分数。schema 3 中，开口超过 `0.9 + 0.8 = 1.7 rad` 后，适度收窄不会提高开口分；在闭合门控完全开启时，开口达到 `0.9 rad` 以上也会出现闭合分恒为零的平台。新分数让这些位置仍有收窄信号。开口在合法区间内仍不受开口偏差惩罚，HOME 六轴参考、2 cm / 20° / 0.15 s 精确接近要求，以及真实接触、抬升、放置和静止认证均未放宽。

目标实际出现后，schema 4 每步加入 `remaining_task = -c * dt * (1 - progress / 10)`，默认 `remaining_task_cost_per_s = 0.3`。`progress` 复用有界的物理阶段分数：接近最高 2，真实接触/抬升至 4，运输至 6，放置/松爪至 8，回 HOME/保持至 10。它独立于 `potential_max` 与 `shaping_enabled`；单独关闭势能塑形不会关闭这项任务代价，消融实验应显式将该系数设为 0。代价始终非正，停在接近阶段仍持续付出代价；闭爪不能代替双侧接触认证，也没有抵达阶段就可反复领取的正奖金。

这项代价读取下一状态的有效物理证据。目标出现前以及两个纯 HOME 场景不计此项；目标出现后，暂时无效、陈旧或缺失的证据沿用最近有效进度，不会因为丢失观测而免除代价。有效的物理退步会增加代价。完整成功当步此项为 0，时间截断照常计费并 bootstrap，失败仍付原失败惩罚。episode 重置清空相关记忆。它不证明不存在策略钻空子的方式，仍需检查接触、抬升、完整成功及逐项成本。

增加直接任务代价的原因是：势能塑形可帮助有限样本学习，但在精确价值函数下，`V_shaped = V_base - Phi`，优势不会凭空多出持续的距离偏好。原训练目标出现后的基础回报几乎只有时间代价，52 个抓放 episode 均未发现成功。离线数学检查验证了这一恒等式，**没有证明实际 critic 已经收敛或已经抵消了塑形**。上述直接代价提供独立于势能差的偏好，同时明确放弃“只改变学习速度、不改变最优策略”的保证。理论参考：[Ng 等人的势能塑形论文](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)。

显式载入历史 `resolved_reward.json` 的 schema 3 时，继续使用原来的线性数学表达式，且不添加直接任务代价，配置哈希保持不变，可用于旧 ACT 与更新后 ACT 的公平配对评估。不能把 schema 3 rollout 或 PPO 完整状态直接用于 schema 4。这些改动不能据此解释全部失败：已检查的训练中，最佳接近距离约 2.88 cm、开口约 1.65 rad，仍未进入闭合门控；该帧任务证据的旧开口分为 0.078，同一行关节数据重建约为 0.059，两者存在采样时间差，均未完全饱和。

指数替换保证远处仍有信号，不保证每个开度的信号都更强。例如在上述状态，收窄 0.01 rad 的离线势能增量从约 0.00145 变为 0.00057；下降 1 mm 则从约 0.01317 变为 0.01380。这些是控制其他状态不变的局部计算，不是可执行轨迹或学习率标定，也不证明 PPO 已获得足够探索。

schema 4 同时移除 HOME 基础位置惩罚的平台。先减去原关节容差，令 `x = max(abs(q-q_home)-tolerance, 0)/scale`，每轴使用 `x²/(1+x²)` 后取均值，仍乘原 HOME 权重及 `dt*30`。实现通过 `qerr/hypot(scale,qerr)` 避免直接平方极大误差。旧版 `clip(x,0,1)²` 在约 30° 以上恒定，无法区分实测夹爪 78° 附近的小幅收拢；新版静止空场仅夹爪误差为 78°、76°、73° 时，每步位置成本分别约为 -0.0073201、-0.0072723、-0.0071944。容差内仍为零，实际速度成本、HOME 目标和稳定保持要求保持原值。schema 3 使用原公式；这项改动同样明确改变基础目标，不只依靠 HOME 势能塑形。

## 目标位置课程

每次从真实 HOME 和桌上方块开始，训练中不伪造夹持状态。以下几何范围仍需实际仿真标定：

| 等级 | 目标半径 / 方位角 | 晋级证据 |
| --- | --- | --- |
| 0 | 0.24–0.28 m / ±10° | 距离 ≤2 cm、方向误差 ≤20°、开口合适，连续 0.15 s；真实抓起或完整成功也可满足 |
| 1 | 0.22–0.31 m / ±25° | 同一 episode 的有效抓取与抬升，或正式完整成功 |
| 2 | 0.20–0.35 m / ±45° | 不再扩大；统计完整成功 |

每关至少收集 20 个当前范围的有效抓放 episode，最近窗口至少 70% 达标才晋级。已过关后 20% 概率回到较小范围练习，这些回顾数据仍用于 PPO，但不计当前关晋级。无目标、已完成场景、无效采集、proxy、重复记录也不能推动晋级。没有按训练轮数强行升级。

所有等级都继续执行完整任务，reach 达标不会提前结束。目标位置按 scene seed 确定，采样排除与实测目标区的任何足迹交叠。单 world、多 world、立即生成和延迟生成共享同一场景接口。课程等级、窗口、已计 episode 和 attempt 计数保存进 PPO checkpoint 并恢复；这不表示能恢复正在运行的 Gazebo 物理状态。

## 探索与闭环反馈

六轴保存独立的阶段 offset；采样、旧概率核对、PPO 更新、解析 KL 都使用同一个有效标准差计算。

启动脚本的候选默认值：

| 阶段 | 手臂 log std | 夹爪 log std | 初始未裁剪标准差（ACT 单位） |
| --- | --- | --- | --- |
| 无目标 / HOME 保持 | -7 | -7 | 手臂 0.091，夹爪 0.046 |
| 接近 / 运输 / 回程 | -4 | -3 | 手臂 1.83，夹爪 2.49 |
| 抓取 / 放置 | -4 | -2.3 | 手臂 1.83，夹爪 5.01 |

基础可学习 log std 的范围为 `[-9, -3.5]`；加阶段 offset 后，手臂有效上限仍为 -3.5，夹爪可到 -1.5。夹爪的大上限不会放宽 HOME 的基础参数范围。HOME 降噪有助于保持，但能否满足实际速度阈值必须看仿真反馈；这是未标定的默认值，不是保证。

`ACT_MAX_ACTION_STEPS=5` 将原生执行队列限制到最多 5 步：30 Hz 下每约 0.167 s 重新看图。原有 1 步 temporal ensembling 保持 1 步；预测 chunk 长度不变。新队列长度写进导出 ACT 的 `config.json`，没有只在训练里使用特权阶段信息强刷动作块。

脚本默认 episode 上限为 1200 steps（40 s），rollout 最低阈值 2400，最多 4 epochs；仍等完整 episode / world wave 才更新，实际 batch 可以超过阈值。既有环境变量覆盖继续生效，例如用户指定的 `ROLLOUT_STEPS=1800` 不会被改写。KL 超限后完整恢复本 epoch，脚本默认最多以更小策略步长重试 5 次；仍超限则停止剩余 epochs。详见[更新幅度与回滚](act_ppo_numerical_stability.md)。

## 配置、记录和恢复

入口仍为 `scripts/finetune_act_ppo_so101.sh`。本次代码修改没有启动训练。

- `REACH_CURRICULUM=true/false`；`CURRICULUM_WINDOW_SIZE`、`CURRICULUM_PASS_RATE`、`CURRICULUM_REHEARSAL_PROBABILITY` 控制晋级和回顾。
- `ACT_MAX_ACTION_STEPS` 控制原生队列上限；显式空值继承 checkpoint。
- `MOVING_GRIPPER_EXPLORATION_LOG_STD`、`MANIPULATION_GRIPPER_EXPLORATION_LOG_STD`、`GRIPPER_MAXIMUM_LOG_STD` 控制夹爪探索；原 HOME / moving / manipulation 设置仍适用于手臂。
- 上述参数均有 Python CLI 对应项；直接调用 Python 默认不主动打开课程或覆盖队列，启动脚本启用本方案。
- `REWARD_CONFIG` 可指定 schema 3 或 4 JSON，默认 schema 4 配置见 `config/pick_place_reward.json`，包括 `desired_grasp_axis_world` 和细接近门限。历史公平评估应明确指定该次运行保存的 schema 3 配置。
- episode JSON 记录课程场景和 `stage_reached.precise_reach`；`metrics.jsonl` 记录课程进度。TensorBoard 记录课程等级、实际动作噪声、精确接近、抓取、抬升、放置和完整成功，不能用累计 reward 代替完成率。

critic task context 升至 schema 2 / 93 维；采样 schema 仍为 3。旧 PPO 完整状态与本版奖励/critic/探索配置不兼容，会明确拒绝。可把旧的 `pretrained_model` 作为 ACT 权重起点，通过脚本的 `PRETRAINED_MODEL` 指定并重新收集 rollout；直接调用 Python 时不要传入 `--resume-checkpoint`。不要混入旧 rollout、critic 或优化器状态。

课程的具体半径、角度、采样规则也包含在恢复契约中，改动后不能静默沿用旧进度。

实现入口对应：

| 文件 | 职责 |
| --- | --- |
| `src/so101_capstone/rewards/task_state.py` | 实测几何分数、精确 reach 证据、critic context |
| `src/so101_capstone/rewards/pick_place_reward.py`、`config/pick_place_reward.json` | 版本化势能、直接任务代价与默认 schema 4 参数 |
| 新增 `src/so101_capstone/training/act_ppo_curriculum.py` | 位置课程、明确坐标、恢复契约 |
| `src/so101_capstone/training/act_ppo.py` | 六轴探索与一致的 PPO 概率回放 |
| `src/so101_capstone/training/act_ppo_gazebo.py`、`act_ppo_multiworld_gazebo.py`、`act_ppo_multiworld_worker.py` | 两条采集路径、显式生成协议、队列和 checkpoint 集成 |
| `src/so101_capstone/training/act_ppo_tensorboard.py`、`scripts/finetune_act_ppo_so101.sh` | 诊断指标与训练默认配置 |

新增回归文件为 `test_pick_place_pregrasp.py`、`test_act_ppo_curriculum.py`、`test_act_ppo_learning_integration.py`；已有 PPO、buffer、数值、两条采集入口测试同步更新。原项目文件没有删除。

## 验证边界

单元测试和离线真实 ACT 检查覆盖奖励连续性、循环塑形、真实成功门槛、六轴概率重算、KL、课程升级/恢复、显式目标协议及原生配置导出。它们不证明机器人已学会抓放，也不验证接触、碰撞、夹爪几何门限和学习率适合实际物理系统。

另外对三个范围共 27 个代表点做了离线 FK / 关节限位求解，均能满足 2 cm / 20° 门槛；这只证明这些点的几何可达性，不代表无碰撞轨迹或真实抓持可行。

2026-09-27 的独立受控仿真进一步验证了中心场景 `(0.26, 0)` 的下降、收口与抬升：1040 步预设轨迹产生真实双侧接触、通过抓取认证，并将方块中心抬高约 29.73 mm。它没有使用 ACT 或 PPO，也未尝试搬运、放置和回 HOME，因此不能视为 ACT 已学会任务，或所有课程场景已经完成物理标定。现有参考点、HOME 与成功阈值保留。

当前课程只调整目标位置，没有演示回放、教师动作或抓取阶段的人工初态。如果真实 rollout 始终没有双侧接触/抬升，后续应先校准抓取几何和探索，再考虑增加成功抓取示范或经过物理验证的阶段初态；增加 epoch 本身不能补出未经历过的技能。
