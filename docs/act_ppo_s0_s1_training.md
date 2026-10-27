# ACT PPO：仅 S0 / S1

2026-10-01。当前只启用 S0/S1，以同一个 ACT 的 PPO 微调学习 HOME 保持和精确 reach。S2 及以后未启用。启动脚本默认选择 `final_good_act` 探索，以及 `act_ppo_stages_s1_lift_approach.json` 的 process reward schema 4：S0 保留 schema 5 HOME 奖励；S1 在同一10秒窗口内先抬升至桌面上方8 cm，再从上方接近，并全程引导转腕90°及对准。接近段保留原8→2 cm细分奖励，详见 [S1抬升与接近](act_ppo_s1_lift_approach.md)。动作、30 Hz 步进、reset、当前几何和真实任务认证沿用 capstone，不添加 BC 或教师动作。

## 默认运行

```bash
bash scripts/finetune_act_ppo_so101.sh
```

默认权重为 `codex_workspace/checkpoints/act_0005000_te_off_zdtwineg`：原 0005000 ACT，`temporal_ensemble_coeff=null`，每步重新规划。原始 checkpoint 不被修改。旧全任务 PPO 状态与新阶段采样签名不兼容；第一次应使用 ACT 权重开始，不恢复旧 optimizer/critic。

- `TRAINING_STAGES=s0_s1`，也支持 `s0`、`s1`；`full` 保留旧训练方式。
- `STAGE_CONFIG=config/act_ppo_stages_s1_lift_approach.json`。旧 `act_ppo_stages_s1_progress.json`（process schema 3、update105基线）、`act_ppo_stages_s1_progress_v2.json`（schema 2）、`act_ppo_stages_dense.json`（schema 1）及更早的 `act_ppo_stages.json` 保留各自路径，供已有 run 使用；显式 export 旧路径会覆盖新默认。
- `WORLD_COUNT` 是并行上限，当前脚本默认值为 1；设置为 4 时，每个 wave 仍可以只启动 1、2、3 或 4 个 world。
- `S0_STEP_FRACTION=0.4`：正常采样批的有效 ACT 步数目标为 S0 40%、S1 60%。`STAGE_RATIO_TOLERANCE=0.05` 允许绝对偏差 5 个百分点，即 S0 实际占比在 35%–45%。这不是 episode 抽取概率。2026-10-03 新增的阶段成功跳过规则可排除本批的部分更新数据，并临时改变下一批的采样比例，见下文。
- `STAGE_MAX_ROLLOUT_MULTIPLIER=4`：限制为满足比例而持续采样的规模，具体边界见下文。
- 旧 `S0_PROBABILITY` 仍表示单 episode 的随机选择概率，multi-world `s0_s1` 新调度不使用它。旧启动命令应改用 `S0_STEP_FRACTION`；不会把旧数值静默解释为步数比例。
- `ENVIRONMENT_RANDOMIZATION=false`：不随机改变场景、灯光、相机位姿、关节测量、HOME起点或目标区。
- `IMAGE_AUGMENTATION=true`：只处理两路相机图片。配置为 `config/image_augmentation_s0_s1.json`，包含亮度、对比度、颜色、白平衡、噪声、轻微模糊、平移、畸变和裁切。top 输入仍经过现有相机处理链。PPO 重放使用采集时存下的增强图片，不在 epoch 内重新随机增强。
- 两种随机化开关不能同时开启，避免重复图像处理。
- `EXPLORATION_PROFILE=final_good_act` 仅影响探索分布。项目 `config/pick_place_reward.json` 的 schema 5 为 S0 和完整任务提供奖励；阶段 JSON 的 `process_reward` 单独选择 S1 替代奖励。切换探索 profile 不会切换奖励。
- 默认 rollout 阈值 2400、2 epochs、minibatch 16、KL 上限 0.02；启动脚本的 actor/critic/探索参数初始学习率均为 2e-4，backbone 为 2e-5。KL 超限整轮回滚并最多 8 次缩小步长的机制保持。它们是待仿真验证的起始配置，实际有效学习率记录在日志中。
- 一次 update 等待当前 wave 各 world 完整结束，再同时检查总步数和阶段比例。达到 episode 数量目标时，如果已有未完成的 batch，继续采样使其达标，再 update 收尾。`full` 模式保留原有尾批处理。

## 阶段达标后跳过更新与下一批采样

2026-10-03：shell 启动器在 S0/S1 模式下默认启用 `STAGE_SUCCESS_SKIP_ENABLED=true`、`STAGE_SUCCESS_SKIP_THRESHOLD=0.8`。保存的 `codex_workspace/commands/finetune_good40_posture_distance.sh` 也使用这两个可覆盖参数。Python multi-world API/CLI 为兼容旧调用默认关闭；显式使用 `--stage-success-skip --stage-success-skip-threshold 0.8` 启用。该入口也支持 `WORLD_COUNT=1`。当前阶段 JSON 满足要求；没有二元阶段 context 的旧 process schema 0/1 必须关闭此功能，启动前会校验。

每次完整采样批达标后，按阶段分别计算本批的 `成功有效 episode 数 / 全部有效 episode 数`。使用正式 `stage_success`，不按总 reward、`moving/manipulation` 或局部抬升事件判断。有效的阶段失败/超时计入分母；相机超时、Gazebo crash 等无效采集不计入。成功率 **大于等于 80%** 时：

1. 本批该阶段的所有 transition 都从 PPO 输入移除，actor、critic、探索参数和辅助损失均不使用它们。episode 和 step 原始记录仍保留。
2. 下一完整采样批跳过该阶段，用仍启用的阶段采满 `ROLLOUT_STEPS`，临时不要求被跳过阶段的原比例。完整 episode 可使实际步数超过门槛，不复制样本或 padding。
3. 再下一批恢复该阶段并重新评估成功率。没有新样本的跳过阶段不能凭旧成功率再次触发；另一阶段仍独立按其新采样结果判断。

例如设 `ROLLOUT_STEPS=2400`，某批收集 S0 900 步、S1 1500 步，S0 成功率达到 80%、S1 未达到：当前 PPO 使用 1500 步 S1；下一批仅采 S1，仍至少 2400 步；之后 S0 恢复。**补满步数指下一采样批，当前批过滤后不会再补采。** 若两阶段均被跳过，或唯一配置阶段被跳过，没有剩余阶段可补：明确记录一次无世界、无 PPO 的空批，随后恢复采样；不推进 PPO update 计数，也不永久停练。

调整门槛或关闭功能：

```bash
STAGE_SUCCESS_SKIP_THRESHOLD=0.9 bash codex_workspace/commands/finetune_good40_posture_distance.sh
STAGE_SUCCESS_SKIP_ENABLED=false bash codex_workspace/commands/finetune_good40_posture_distance.sh
```

门槛必须在 `(0, 1]` 内。更改门槛/开关会改变严格的 PPO checkpoint 恢复签名；旧完整训练状态不能静默恢复到新规则，导出的 ACT 权重仍可作为新 run 起点。新 checkpoint 保存下一批待跳过阶段和独立采样批编号，恢复时不会把旧批成功统计混入新批。尚未保存的部分 rollout 不会恢复。

`metrics.jsonl` 新增 `stage_success_skip_decision`，记录各阶段成功数、分母、成功率、当前排除阶段、下一批跳过阶段，以及采集/更新/排除步数；整批不更新时记录 `stage_update_skipped`，全阶段跳过的空批记录 `stage_sampling_skipped`。TensorBoard 的 `stage_skip/` 横轴为从 0 开始的采样批编号，和 PPO update 编号分开。`summary.json` 单独累计 `success_gated_transition_count`，不会算作 crash 或无效采集的浪费步数。

跳过 S0 数据不会冻结共享 ACT 的 S0 行为：S1 优化仍会改变同一个网络。当前规则是暂停使用达标阶段的数据，恢复采样后再检查其能力；不是防止遗忘的保证。此改动已做 CPU 单元与模拟调用链测试，未执行 Gazebo live 或真实硬件验证。

## 当前 S0 的阶段成功标准

2026-10-03：当前启动器默认的 `act_ppo_stages_s1_lift_approach.json`，以及保存命令使用的 `act_ppo_stages_s1_lift_completion.json`，均启用以下 S0 标准：

- 六个关节（含夹爪）各自相对 HOME 的绝对误差严格小于 **8°**，配置字段 `s0_home_joint_tolerance_rad=0.13962634015954636`，恰好8°不通过。
- **速度不参与 S0 成功判断**。速度和静止占比仍如实记录，不把位置保持成功标记成已经静止。
- 位置条件连续满足 **1 秒**，`s0_home_hold_s=1.0`、`s0_success_mode=position_hold`，立即判定 `stage_success` 并结束本段采样；最长预算仍为5秒。任一关节位置达到或超出阈值会清零连续计时，重复或失效的证据不能用于补足保持时间。

此条件只定义 S0 阶段通过；完整任务 HOME 奖励、探索噪声收窄、初始化确认和 S1 使用的 HOME 阈值仍读取原 reward 配置（1°、0.03 rad/s）。初始化仍须满足原严格条件；准备过程不计入阶段的1秒。S0通过是采样截断，保留价值 bootstrap，不冒充完整抓放成功。

`finetune_good40_posture_distance.sh` 已设置 `S0_HOME_HOLD_S=1.0`。手动粘贴旧命令时要把 `S0_HOME_HOLD_S=3.0` 改为 `1.0`，或取消该覆盖以继承 JSON。只要本批正式 S0 episode 成功率达到80%，现有机制就排除本批 S0 更新、下一批跳过 S0 并由 S1 补满步数；单次通过或平均位置误差小于阈值不足以触发。

旧 JSON 未配置这些新增字段时保留原语义：阈值继承 reward，`s0_success_mode=window_end`，在预算结束时检查末尾位置和速度保持时长。上一版 `home_hold` 模式仍要求位置和速度同时达标再提前成功。新字段进入阶段配置和 checkpoint 恢复签名，改变标准需使用匹配的训练状态，或从 ACT 权重开始新 run。

## 当前探索与学习起点

这些学习率是 **shell 启动脚本默认值**，没有统一改写 Python 配置类、直接 CLI 或 checkpoint 的默认值。显式数值环境变量仍优先：

| 启动参数 | 脚本默认值 | 覆盖规则 |
|---|---|---|
| `ACTOR_LEARNING_RATE` | 2e-4 | 显式数值覆盖；显式空字符串继承 ACT checkpoint |
| `ACTOR_BACKBONE_LEARNING_RATE` | 2e-5 | 显式数值覆盖；显式空字符串继承 ACT checkpoint |
| `CRITIC_LEARNING_RATE` | 2e-4 | 非空环境变量覆盖 |
| `EXPLORATION_LEARNING_RATE` | 2e-4 | 非空环境变量覆盖 |

`final_good_act` 按当前 reach 距离设置六个关节的初始物理噪声标准差：大于 0.15 m 为 0.12 rad，0.08–0.15 m（不含 0.08）为 0.08 rad。2026-09-30 增强 S1 近距离探索：当 `training_stage=S1` 且处于非 HOME 任务阶段时，不大于 0.08 m 的标准差由 0.05 rad 提高到 **0.12 rad**，即幅度 2.4 倍、方差 5.76 倍。完整任务模式的近距离分档仍为 0.05 rad；S0/HOME 的分档和收窄规则不变。S1 覆盖由距离触发的 `GRASP_LIFT`，避免进入 4.5 cm 后提前退回旧噪声。

每个关节使用现有标定的绝对关节限位半跨度，将物理标准差换算到 ACT 归一化动作空间；没有使用旧项目的增量动作上限。ACT 均值仍为绝对位置动作，现有 sampler 在裁剪前 Gaussian 上计算 PPO 概率，再裁剪执行命令。实际关节运动幅度仍受边界、控制器及动力学影响。

WAIT_HOME、RETURN_HOME、HOME_HOLD 使用当前最大关节误差（rad）分档，并将初始标准差收窄为 `min(分档标准差, 0.001 + 0.5 * max(误差 - HOME容差, 0))`。HOME 目标和当前 1° 位置容差读取现有 reward 配置，未改变。进入容差后初始标准差为 0.001 rad；这不等于关闭探索，也不保证实际保持已经稳定。

噪声仍逐 30 Hz 控制步采样，可学习的基础 log-std 继续调整这些初始尺度。`INITIAL_LOG_STD=-3`、`MINIMUM_LOG_STD=-9`、`MAXIMUM_LOG_STD=-2.5` 及夹爪有效上限 -1.5 保留；若显式边界不能容纳该 profile 的初始物理尺度，启动校验会报错。`PPO_EPOCHS=2`、`TARGET_KL=0.02`、clip 0.2、全局 entropy bonus 0 和现有 KL 回滚机制保持。

以上 KL 是脚本默认值；最近训练命令显式覆盖为 `TARGET_KL=50`，本次探索修改不调整它或学习率。当前上限可容纳六轴初始 0.12 rad，无需增大通用 log-std 上限。探索 signature 现为 schema 2，保存 S1 覆盖条件与幅度；旧完整 PPO 状态不允许直接恢复到新探索，使用 ACT 权重开始新 run 即可。上一条从 5000 开始的完整命令无需修改，重新启动才读取新代码；已有进程不会自动切换。

直接调用 Python CLI 或 `ActPpoGazeboConfig`，默认探索仍为 `stage_log_std`；通过 shell 启动时才默认传入 `final_good_act`。新 profile 不使用旧 `IDLE/MOVING/MANIPULATION/HOME_HOLD_*EXPLORATION_LOG_STD` 及 moving/manipulation 夹爪组参数。需要旧固定分组行为时显式设置 `EXPLORATION_PROFILE=stage_log_std`。学习率环境变量在两种 profile 下都继续生效。

## S0/S1 奖励及恢复边界

S0 沿用项目 schema 5 的 HOME 过程奖励和原有其他分项：按容差外误差的历史最好值给改善奖励，停滞或退步时给成本；位置、速度都满足 HOME 保持要求时，新增 HOME 停滞和未稳定成本为零。HOME 目标、1° 位置容差、速度阈值及保持时间未改变。完整公式见 [schema 5 说明](act_ppo_final_good_act_profile.md)。

S1 的 `rewards/stage_process_reward.py` 完全替换完整任务奖励总和，不叠加 schema 5 的历史距离/角度罚分、近处闭爪 gate 或完整任务过程奖励。process schema 2/3 保留九项连续势函数和软门控，但把它们整体乘 0.1，使总势上限从 100 降为 10。当前 schema 3 将时间成本从 0.03/秒增为 0.3/秒，S1 未完成程度成本和阶段通过 +10 保留。

距离质量 `q_d=0.1/(0.1+max(d-0.02,0))`，每步距离成本为 `-0.4 * dt * (1-q_d)`；首次刷新 episode 最好距离质量时加 `8 * max(q_d-历史最好,0)`，只在前后有效资格、新鲜证据且非 terminal 时支付。schema 3 另在 8→2 cm 每 5 mm 一段追加线性进步信用，12 段总预算 48，只支付历史最好以外的新增部分。初始化不奖，接触或证据缺口只提高历史基线、不补发奖励，重复往返不能持续刷同一进步。开口仍引导到 0.45–0.90 rad，细微改善可增加对应项，但总奖励不保证为正。当前公式见 [S1 连续分段奖励](act_ppo_s1_distance_bands.md)，研究和历史 schema 2 见 [S1 距离进步奖励](act_ppo_s1_progress_reward.md)。

当前 grasp 参考点、4 cm 方向参考几何、向下轴要求与 stage 通过条件保持。目标仍需满足源区、支撑、静止和无接触的 reach 门槛；真实认证抓持并抬升仍可作为更强的 S1 通过证据。S1 阶段通过奖金为 +10，不是完整任务成功 +100，真实失败项仍为 -10；stage success/timeout 保留 truncation 与价值 bootstrap，真实失败使用 terminal 势函数零值。重复证据不给进步，跨证据缺口不补发正向增量。

schema 5 原 93+33=126 维 critic task context 保留；process schema 2/3 均追加 14 维阶段奖励状态，当前合计 140 维。它在 schema 1 的九项缓存势函数、active/initialized 标记和已用预算比例之外，再增加最好距离质量和前一步有效资格；schema 3 复用该历史计算已领取分段，task context schema 为 6。S0 的 14 维全为零。ACT actor 的图像和关节输入不扩展。

process schema 2/3 自动启用 S0/S1 分组的 actor advantage 标准化，只用各组参与 actor 更新的样本统计均值/标准差，保留原 GAE 和 critic return。它处理已观察到的阶段尺度差异，不保证 S1 必然进步；负 advantage 本身也不是错误。schema 1 的全局标准化语义保留，具体诊断见新奖励说明。

旧 `act_ppo_stages.json` 没有 `process_reward`，`act_ppo_stages_dense.json` 保留 process schema 1，`act_ppo_stages_s1_progress_v2.json` 保留 process schema 2；它们保留各自的奖励与恢复签名语义。已有 run 应继续用原 JSON，不能中途换成新奖励。完整任务 reward schema 3/4 语义和配置 hash 也保留。新配置与旧完整 PPO 状态不兼容；可以加载导出的 ACT 权重开始新 PPO run，重新初始化 critic、optimizer 和奖励历史。

旧 schema 5 reach 的持续成本已经确认存在回报排序风险，例如 d=0.2 m 时仅距离停滞和步成本就为 -4.5/步，30 步折现约 -129.29，可能比失败 -10 更差。它是源码公式可构造的激励问题，尚未观察到 rollout 主动利用；这也是默认 S1 独立替换奖励的原因。新方案尚无 Gazebo live 调参或成功率提升结论，不能把离线数学性质或测试等同于训练效果。

## 同阶段 wave 与每批采样的比例

一个 wave 内，**本 wave 选中的 world 运行同一个阶段、使用同一阶段预算**。例如选中 S0 时都运行 HOME 观察窗；下一 wave 可以全部切到 S1。`WORLD_COUNT` 是并行上限，每个 wave 只启动被选中的 world，接近 batch 达标时只启动 1 或 2 个 world 是正常行为。S1 提前成功或发生真实物理失败仍会提前结束，因此各 world 的实际 episode 长度不保证相等。

调度器根据**当前 PPO batch 已收集的有效、完整 episode 步数**，以及各阶段已采集有效完整 episode 的平均长度，规划下一 wave 的阶段与 world 数量；数量不超过 `WORLD_COUNT` 和当前可用数量。某阶段还没有实际长度数据时，先用其时间预算乘控制频率估计。每个 wave 完成后按真实结果重新规划，补充不足的阶段。预测步数只用于规划，不保证实际 episode 一定跑到预测长度。

初始化动作、采集错误造成的无效片段、wall-clock 等待时间都不计入比例。完成一个 wave 后，必须同时满足：

1. 有效总步数 `N >= ROLLOUT_STEPS`；
2. `abs(S0有效步数 / N - S0_STEP_FRACTION) <= STAGE_RATIO_TOLERANCE`；
3. 目标占比非零的阶段都已贡献有效样本。

满足后，先执行启用的阶段成功过滤，再将剩余数据交给一次 PPO update；关闭过滤时使用整个 batch。不通过裁断 episode、复制样本或 padding 凑比例。每个用于训练的有效 transition 只归属一个 update，update 内仍按 `PPO_EPOCHS` 做优化。下一 batch 重新统计占比，所以保证的是每批采样的构成，而非长期累计平均；存在跳过阶段时使用本批剩余阶段的比例。

`ROLLOUT_STEPS` 是最低门槛，完整 episode 的离散长度和比例约束仍可能增大实际 batch。在显式设置 `WORLD_COUNT=4`、30 Hz、S0/S1 各跑满 5/10 秒、无采集失败、`ROLLOUT_STEPS=2400` 时，单个 S0/S1 episode 分别为 150/300 步。S0 目标 40%、绝对容差 0.05 时，预计可按下列顺序采样：

| Wave | 阶段与选中 world 数 | 新增步数 | 累计 S0 / S1 步数 | 累计总步数 |
|---|---|---|---|---|
| 1 | S1 × 4 | 1200 | 0 / 1200 | 1200 |
| 2 | S0 × 4 | 600 | 600 / 1200 | 1800 |
| 3 | S0 × 2 | 300 | 900 / 1200 | 2100 |
| 4 | S1 × 1 | 300 | 900 / 1500 | 2400 |

最终 S0/S1 占比为 37.5%/62.5%，满足 40%±5 个百分点；未触发成功过滤时全部 2400 步进入同一次 update。相同满预算条件下，容差收紧为 0.01 后，因完整 episode 长度离散，最小达标 batch 为 3000 步，即 S0 1200、S1 1800，比例恰为 40%/60%。

这些是基于完整预算的规划示例，不是实际采样长度或 wave 顺序的保证。提前结束、采集失败或可用 world 数变化后，调度器会使用新的实际计数与均长重新规划。容差更小可能需要更多数据，且并非所有比例都能在有限个完整 episode 内达到。减少尾 wave 的 world 数不裁断 episode，也不复制样本或 padding。

为避免无限等待，在启动下一 wave 前，如果当前 batch 已达到 `STAGE_MAX_ROLLOUT_MULTIPLIER * ROLLOUT_STEPS`，却仍未达标，则报错停止；不会提交比例不合格的 update。最后一个已经开始的 wave 会完成，允许步数超过这个软上限；如果它使 batch 达标，正常 update。

这些调度参数进入 checkpoint 恢复签名。旧随机 episode 混合方式保存的 PPO 状态不能直接续训到新调度；可使用其导出的 ACT 权重开始新的 PPO 训练。更改比例、容差或上限也应按恢复校验处理。

## 从启动命令调整阶段时间

下面四个环境变量可以加入现有启动指令，单位都是**仿真秒**，不包含初始化准备时间：

| 环境变量 | Python CLI | 当前 JSON 默认值 | 含义 |
|---|---|---|---|
| `S0_BUDGET_S` | `--s0-budget-s` | 5 | S0 最大观察时长；当前配置满足保持条件可提前成功 |
| `S0_HOME_HOLD_S` | `--s0-home-hold-s` | 1 | 当前 S0 六轴位置同时满足条件的持续时间；旧 window_end 模式检查末尾位置和速度保持 |
| `S1_BUDGET_S` | `--s1-budget-s` | 10 | S1 最长时间；达到通过条件可提前结束 |
| `S1_REACH_HOLD_S` | `--s1-reach-hold-s` | 0.15 | S1 所有 reach 条件同时满足的连续时间 |

例如将 S0 调到 6 秒、S1 调到 15 秒，保持时间分别为 3 秒和 0.15 秒：

```bash
env TRAINING_STAGES=s0_s1 \
  S0_BUDGET_S=6 S0_HOME_HOLD_S=3 \
  S1_BUDGET_S=15 S1_REACH_HOLD_S=0.15 \
  S0_STEP_FRACTION=0.4 STAGE_RATIO_TOLERANCE=0.05 \
  ENVIRONMENT_RANDOMIZATION=false IMAGE_AUGMENTATION=true \
  bash scripts/finetune_act_ppo_so101.sh
```

优先级为：非空启动参数 > `STAGE_CONFIG` JSON > 内置默认值。变量不设置或留空均继承 JSON，不会改写 JSON 文件；四项必须为有限正数，保持时间不能超过对应阶段总时长。`TRAINING_STAGES=full` 不接受这四项覆盖。

调整时长后仍直接设置 `S0_STEP_FRACTION=0.4` 表示实际步数目标 40%/60%，无需换算 episode 概率。调度器按真实采样步数补充不足的阶段；S1 提前成功也计入当前 batch 的实际统计。

| 调度环境变量 | Multi-world Python CLI | 默认值 |
|---|---|---|
| `S0_STEP_FRACTION` | `--s0-step-fraction` | 0.4 |
| `STAGE_RATIO_TOLERANCE` | `--stage-ratio-tolerance` | 0.05，绝对比例偏差 |
| `STAGE_MAX_ROLLOUT_MULTIPLIER` | `--stage-max-rollout-multiplier` | 4 |

实际生效时间写入 `resolved_stages.json` 和 checkpoint execution signature。改变时间后，原 PPO checkpoint 的恢复校验会拒绝不一致配置；可以从导出的 ACT 权重开启新训练。`MAX_STEPS_PER_EPISODE` 不控制 S0/S1 时长。

## 场景及边界

| 阶段 | 起点 | 通过条件 | ACT 仿真预算 |
|---|---|---|---|
| S0 | 无目标、按原严格阈值初始化到HOME稳定 | 各轴位置误差<8°连续≥1秒立即成功，速度不参与判定 | 最多5秒；30Hz下连续保持约30个控制间隔 |
| S1 | HOME稳定、目标在源区支撑静止、相机已看到生成后的场景 | grasp参考点距离≤2cm、XY≤1cm、高于物体中心0–2cm、方向误差≤20°、相对HOME夹爪开口0.45–0.90rad、目标未移出源区且静止，全部同时连续0.15秒 | 最多10秒，30Hz通常300步 |

HOME与物体运动/支持等阈值读取统一reward配置。新增reach几何阈值仍标记 `candidate_unvalidated`，尚未由可靠抓取轨迹标定为最优可抓姿态。真实接触认证的当前抓持并抬升可以作为更强的S1通过证据。

S1从现有中心扇区采样目标（半径0.24–0.28m，角度±10°），排除目标区重叠。旧按抓取/完整成功晋级的位置课程被禁用。此处目标采样是任务本身的训练分布，不开启环境随机化。

初始化仅由HOME控制和物理传感器确认完成，时间不计入5/10秒，动作不加入PPO。目标生成服务ACK不足以开始S1；必须等待新的目标、关节、接触、两路相机证据，再重建首帧与奖励基准。保留最近连续的目标pose历史仅用于估计首帧速度，不继承任何准备过程任务事件。

`SUCCESS`和`TIMEOUT`都是本轮采样段结束：`truncated=True, terminated=False`，保留实际末状态势函数和价值bootstrap，GAE不跨重置。新 S1 奖励可在阶段 `SUCCESS` 发放一次 +10，但这不改变其 truncation 边界。真实任务失败保持terminal；无效传感器、初始化失败按采集错误另报，不构造策略失败transition。

阶段成功不会领取完整抓放100分奖金，也不会增加完整任务成功率。新默认 S1 引导满足 reach 开口区间，不叠加完整任务近目标闭爪奖励；旧阶段 JSON 仍保留原反馈。成功边界之后暂缺后续训练数据，不能据此认为后续抓放已经学会，或把局部成功率当作最终能力。

## 记录与复现

`resolved_stages.json` 保存阶段阈值、时间及选定的 `process_reward`。checkpoint execution signature 保存有效阶段配置、调度比例、容差、采样上限、图像增强配置，以及 reward 配置 hash、critic context 维度、schema 5 奖励历史版本和启用的 stage process reward 签名。新探索 profile 还写入物理分档、HOME 收窄规则及关节标定换算的签名，恢复时校验一致性。`stage_attempt_count` 随 checkpoint 恢复，使目标与增强种子继续前进。

每个episode的JSON记录 `training_stage`、`stage_result`、`stage_elapsed_s`、`start_certificate`、`stage_initialization`、`stage_metrics`。准备耗时、丢弃的控制步单独保存。TensorBoard使用 `stage/S0/` 和 `stage/S1/`：显示成功、超时、逐轴全窗最大/RMS误差与速度、HOME稳定占比，以及reach距离/XY/高度/方向/开口。

每个 episode 还记录 `wave_index`。`metrics.jsonl` 的 `stage_wave_sampling` 记录每个 wave 后当前 batch 的实际计数和达标状态；每个 `ppo_update` 的 `stage_sampling` 保存最终构成。TensorBoard 的 `stage_batch/` 显示 S0/S1 步数、实际 S0 比例、目标比例、误差与容差。

RMS为不同证据时间戳的样本RMS；稳定时间占比分母为整段实际仿真时间，重复证据与证据缺口不能计入稳定时间。`task_success`与阶段成功分开；辅助后续起点不在本轮范围。

代码入口为 `training/act_ppo_multiworld_gazebo.py`，worker 负责初始化和环境测量，主进程负责 ACT、critic、PPO 更新与保存。阶段评估在 `rewards/training_stages.py`；ACT 不接收隐藏阶段命令，沿用前五轴 -100 到 100、夹爪 0 到 100 的绝对位置动作映射。

## S1 近距离探索变更验证（2026-09-30）

在 conda `lerobot` 环境中加载 ROS 2 Humble 与 workspace，设置 `PYTHONNOUSERSITE=1`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、`PYTHONDONTWRITEBYTECODE=1`，并将 `src`、`lerobot/src` 加入 `PYTHONPATH`。执行 `python -m pytest -q -p no:cacheprovider tests/unit/test_act_ppo*.py tests/unit/test_stage_process*.py tests/unit/test_training_stages.py tests/unit/test_reach_env_core.py`，结果 **1242 passed in 10.12s**。

新增测试检查近距离阶段切换、HOME 和其他模式隔离、六轴实际有效标准差、相同随机种子的命令 RMS 增至 2.4 倍、双向 Gaussian 概率重放，以及新旧完整 checkpoint 契约拒绝混用。未启动 Gazebo 或训练；上述分布验证不能等同于实际末端移动幅度或成功率提升。
