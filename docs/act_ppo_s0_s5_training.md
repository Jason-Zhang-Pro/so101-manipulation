# S0–S5 分阶段训练与自动准备

完整六阶段启动器是 `scripts/finetune_act_ppo_s0_s5.sh`，内部调用下述通用入口。它使用较好 `act_ppo_20261001_182324_from_checkpoints/checkpoints/update_000040/pretrained_model` 的 ACT 权重，建立全新 critic、optimizer、探索状态与输出目录，从 update 1 开始。

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone"
bash scripts/finetune_act_ppo_s0_s5.sh
```

默认最多 4 个无 GUI 世界、rollout 2400、minibatch 30、2 个 PPO epoch、KL 50、actor LR `3.125e-6`、backbone LR `3.125e-7`、critic LR `2e-4`、探索 LR `3.125e-6`。学习率采用较好 40 结束时的有效值作为新基准；KL 最小缩放比例为 `0.0064`（由原 `1e-4 / 0.015625` 换算），对应 ACT 最小有效学习率 `2e-8`。训练图像增强开启、TE 关闭、物理环境随机化关闭，沿用当前有界动作与原始均值越界辅助损失。每 5 次 update 保存中间 checkpoint，TensorBoard 每次 update 记录。

当前 S1 在抬升与接近两部分都设置 `process_reward.s1_lift_approach.wrist_rotation_reward_enabled=false`：关闭顺时针转腕目标、夹指平行桌面潜力、转腕分段奖励、未转到位扣分及这两项的历史退步扣分；成功判断也不要求转腕到位。朝向 target 和抓取轴向下的对准奖励继续保留。此开关不锁定关节、不更改动作范围或随机探索。默认 S0/S1 配置和 `finetune_good40_posture_distance.sh` 使用的当前配置也同步关闭；历史 `s1_posture_guard` 配置及保存的旧训练配置保留原行为。开关写入奖励配置哈希，旧完整 PPO 状态不会被静默当作相同奖励恢复，ACT 权重初始化仍可使用。

只查看完整生效参数而不启动任何进程：

```bash
bash scripts/finetune_act_ppo_s0_s5.sh --print-config
```

此完整启动器允许显式覆盖 `PRETRAINED_MODEL`、`WORLD_COUNT`、`GAZEBO_GUI`、`EPISODE_COUNT`、`STAGE_STEP_WEIGHTS`；其余参数固定在脚本中，避免上次终端的配置污染。需要 GUI 时使用 `GAZEBO_GUI=true bash scripts/finetune_act_ppo_s0_s5.sh`。完整启动器会清除 `RESUME_CHECKPOINT`；六阶段完整 checkpoint 的精确恢复应通过下述通用入口显式配置。

正式入口仍是 `scripts/finetune_act_ppo_so101.sh`。`TRAINING_STAGES` 支持 `full`、单阶段 `s0` 至 `s5`、原来的 `s0_s1`，以及同时训练六个阶段的 `s0_s5`。使用 `WORLD_COUNT=1` 可在同一套受管理的世界生命周期下训练单世界；无需调用旧的、由外部维护 Gazebo 的 Python 单世界入口。

选择 S2–S5 或 `s0_s5` 时，脚本默认读取 `config/act_ppo_stages_s0_s5.json`。显式 `STAGE_CONFIG` 仍优先。原 `s0_s1` 默认配置和 40%/60% 实际步数调度不变。脚本没有因此调整 KL、学习率、相机、物理随机化或动作限位设置。

## 阶段起点

| 阶段 | 自动准备的起点 |
|---|---|
| S0 | 无目标，原 HOME 准备流程。 |
| S1 | 源区生成目标并确认反馈，原 HOME 起点。 |
| S2 | 空夹爪移动到源区目标附近，确认可抓取的几何起点，ACT 开始学习关闭夹爪和抬起。 |
| S3 | 自动真实闭夹并抬起，双指接触及有效抓持、抬升证据成立后交给 ACT 搬运。 |
| S4 | 先真实抓持、抬起并搬运至放置区上方，持物保持悬空，交给 ACT 降低和释放。 |
| S5 | 先真实抓持、搬运、释放并确认物体稳定放置，空夹爪撤开，再交给 ACT 回 HOME。 |

准备控制器只发送限位内的关节目标，并通过 30 Hz 确定性物理步推进；首次抓持和最终放置必须由实际传感反馈认证。六阶段启动器现在明确启用仿真刚性抓持辅助，初次有效抓持后将物体连接到固定夹指（见下节），不修改物体位姿或伪造接触。准备移动、等待和认证都不进入 PPO buffer，也不消耗阶段训练预算。成功交接后，阶段时钟从零开始。

自动准备确认有效抓持后，直接保持当时的夹爪指令角度，不再额外减去 2°；抬升和搬运期间保持该指令，直到释放。这不改变 ACT 在正式采样阶段输出的动作。执行契约中的 `stage_preparation_grasp_hold` 记录该行为；旧的额外闭合版本完整 PPO checkpoint 不会被静默当成相同准备流程恢复，ACT 权重初始化仍可使用。

自动准备的每步 IK 可接受位置误差为 1 cm。求解仍优先寻找 0.5 mm 内的解，精细求解未达标时再采用 1 cm 内的最佳位置解，避免每步约 2 mm 的目标被宽容差直接判为无需移动。轴向约束、关节限位、速度限制及各阶段实际反馈成功条件继续独立检查。

## S0–S5 起点目视检查

入口是 `codex_workspace/scripts/show_stage_preparation_gui.py`。它在一个 Gazebo GUI 世界中依次展示 S0–S5 起点，只生成一个目标，中间不重建世界，也不加载 ACT 或创建 PPO optimizer。检查面板在各起点暂停世界，点击“继续下一段”后继续自动准备，结束按钮关闭本次检查世界。默认 ROS domain 为 126，partition 每次独立生成；占用冲突会拒绝启动。

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
source /opt/ros/humble/setup.bash
source "$PWD/ros2_ws/install/setup.bash"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PWD/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
python codex_workspace/scripts/show_stage_preparation_gui.py --execute
```

省略 `--execute` 只打印检查计划，不启动 GUI、仿真或写报告。现场报告在 `codex_workspace/reports/stage_gui_时间_标识/`，保存准备步骤、连接角度与状态、阶段快照、错误和清理结果。目视检查展示的是自动准备流程，不是模型阶段成功率测试；它处于准备模式，因此不会启用正式 S2/S3 采样的持续连接规则。可复用的无 GUI 物理验证入口为 `check_stage_preparation_live.py` 和 `check_grasp_attachment_live.py`，同样必须显式提供 `--execute`。

## 分别测试 S0–S5 的 ACT policy

使用 `scripts/evaluate_act_stages.sh`。与只展示起点的 GUI 工具不同，它在每次自动准备并认证起点后，运行 checkpoint 的 ACT 动作，使用正式训练阶段的成功/超时/失败判定。默认按 S0→S5 各测试一次，也可选择任意阶段或重复次数。每个 stage/episode 使用新世界，结束记录保存后关闭；没有 PPO 更新或探索噪声。

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone"

# 将此路径设为要测的模型；示例是较好40。
TEST_CHECKPOINT="$PWD/output_simulation/finetuned/act_ppo_20261001_182324_from_checkpoints/checkpoints/update_000040/pretrained_model"

# 只打印完整测试计划，不启动仿真、不加载权重、不创建输出目录。
bash scripts/evaluate_act_stages.sh --checkpoint "$TEST_CHECKPOINT"

# 默认 GUI，六个阶段分别测试一次。
bash scripts/evaluate_act_stages.sh --checkpoint "$TEST_CHECKPOINT" --execute

# 只测 S2，重复3次；也可写 --stages S2 S3 S4。
bash scripts/evaluate_act_stages.sh --checkpoint "$TEST_CHECKPOINT" \
  --stages S2 --episodes-per-stage 3 --execute
```

S0 从 HOME 开始检测保持；S1 从 HOME 与已生成的源区目标开始；S2–S5 使用前表的真实自动准备起点。准备时间不计入模型动作时间。原 `evaluate_act_home_reach.sh` 的 ZERO→自主回 HOME→3 秒后生成目标的连续测试仍保留。

默认继承 checkpoint 内保存的完整六阶段 `resolved_stages.json` 和 `resolved_reward.json`，因此新六阶段模型保存的 S2–S4 8 秒预算会被沿用。普通监督模型或仅有 S0/S1 配置的旧模型，默认使用当前 `config/act_ppo_stages_s0_s5.json` 与 `config/pick_place_reward.json`，计划中的 `config_source` 会注明；这时 JSON 中 S2–S4 默认是 10 秒。可以通过 `--stage-config`、`--reward-config` 显式选择文件，或用 `--s0-budget-s` 至 `--s4-budget-s` 覆盖时间，S5 始终跟随 S0。评估调用当前项目的阶段与准备实现，不是恢复历史代码。

ACT 原有 chunk、执行步数、TE 和有界输出配置从 checkpoint 原样加载，每个 episode 清空历史。图像增强和物理环境随机化关闭。刚性抓持默认沿用可读取的六阶段源 run 签名；普通监督/旧 S0/S1 模型默认开启当前六阶段辅助抓持。复制到其他位置且缺少源 run 的模型会在计划中说明默认值，可用 `--assisted-grasp true` 或 `false` 明确指定。开启时 S2/S3 保持连接，S4 正常按实测张开释放。

目标固定为 `(0.26, 0, 0.768)`，可用 `--target-position X Y Z` 更换；各阶段相同重复编号使用相同 seed。`--headless` 可关闭 GUI，`--ros-domain-id` 可调整默认独立 domain 125。资源占用时拒绝启动，不清理其他训练世界。默认准备允许 600 秒墙钟时间，使用 `--preparation-timeout-s` 调整。

结果保存在新建的 `output_simulation/evaluation/stages_时间_标识/`，可用 `--output-directory` 指定新的目录：

- 总 `summary.json`：各阶段尝试数、有效评估数、成功/超时/失败数、运行或准备错误数，以及明确分母的成功率。
- `S*/episode_*/summary.json`：权威阶段结果、HOME 六关节误差/速度、目标最近和最终距离、S1 抬升完成及子目标、刚性连接/释放记录。
- `steps.jsonl`：全部已确认 policy 步的动作、实际关节反馈、奖励分项与阶段指标；自动准备不计入。
- `initialization.json`、`events.jsonl`、初始/最终/最近目标图像，以及每个世界的 `gazebo.log`；错误保留部分结果，完成记录后再清理本次世界。

准备失败不算策略超时，也不算成功。正常超时或任务失败是有效评估结果；基础设施错误单独报告。即使多个独立阶段均成功，也不能据此宣称一个连续抓放任务已成功，固定目标重复测试也不代表随机场景泛化成功率。

## 仿真刚性抓持辅助

`ASSISTED_GRASP_ENABLED=true` 导出运行时开关 `SO101_ASSISTED_GRASP_ENABLED`。完整六阶段启动器以及起点 GUI 检查默认开启；通用 PPO 启动器默认关闭，以保留既有自然摩擦实验的行为。

只有 TaskStateTracker 通过真实双指接触、抓取距离、相对稳定性和持续时间确认有效抓持后，才在当前暂停边界创建 Gazebo DetachableJoint：父端 `so101/gripper_link`（固定夹指），子端当前 `target_cube_*/cube_link`。固定关节保留连接瞬间的相对位姿，不把方块吸到指定位置。记录确认时的夹爪指令角度和实测角度，两者独立保留。

**S2/S3 正式采样期间，已经建立的刚性连接持续保持，夹爪指令和实测角度均不能触发解除。** ACT 可以自由输出张开或闭合指令，但张开不会让 target 因主动解除连接而掉落。阶段结束后按原世界生命周期关闭世界；若同一世界切换至 S4，则恢复下述释放规则。真实约束丢失、反馈失效等仍按采样错误处理。

自动准备过程、S4 和其他阶段仍把“新指令允许分开”定义为：新夹爪张开指令达到原确认**实测角度 +2°**，并且夹爪已经实际张开到这个角度。先保持连接执行张开动作，实测达到后在暂停边界解除并等待删除确认；保持夹爪指令、闭合或其他关节运动不会主动解除。新指令改回闭合时，不延续之前未完成的释放请求。每次抓持 attempt 只连接一次，解除后必须有新的真实抓取认证才能重新连接。准备期间不启用 S2/S3 的持续连接模式，保证 S4/S5 的放置起点仍能正常准备。

S2/S3 新增独立的 `s2_attachment_gripper_command` / `s3_attachment_gripper_command` 奖惩。参考值是连接确认瞬间冻结的**实测有效抓持角度**，比较对象是本步实际执行的限位内夹爪指令，而非原始 policy 输出或后续实测角度。指令严格小于参考值时给正分：每秒最多 `+0.05`，在额外闭合 2° 时达到上限；指令等于参考值时每秒 `−0.05`，大于参考值时逐渐加罚，张开超过参考 2° 后封顶每秒 `−0.2`。2° 只是奖惩幅度的归一化尺度，不会修改 ACT 指令或强制额外闭合。配置分别位于 `S2RewardConfig` / `S3RewardConfig` 的 `attachment_close_margin_scale_rad`、`attachment_close_reward_per_s`、`attachment_open_cost_per_s`。

此项只对当前目标、当前抓持 attempt 的有效连接以及新鲜反馈覆盖的时间计分；复用反馈不重复付分。新连接在当前物理步结束后确认，不重算已记录的奖励，后续有效 step 开始计分。日志记录参考角度、指令角度、闭合余量、奖励速率和计分时间。默认正分小于现有每秒 `0.1` 的时间惩罚，不会仅靠原地保持夹爪而取得正的持续净奖励；S2/S3 的原有过程奖励、子目标和阶段成功判定保留。

使用实测角度作为基准是为了处理碰撞限位造成的“指令角度小、实际角度大”。DART 焊接后可能过滤夹爪与方块间的接触，因此解除前必须确认移动夹指已真正张开，避免恢复碰撞时把方块推出去。释放判定只使用解除确认时间之后的新传感反馈，不能用焊接时被过滤为空的旧接触帧证明物体已释放。

辅助模式的自动准备在已确认合法释放、连接解除且实际开夹完成后，允许抬空夹爪来消除残余单侧接触；不把最终的无接触条件提前用来阻挡这次抬升。最后的 S4/S5 起点认证仍检查双侧均无接触、参考点与物体分离、区域和稳定放置。自然摩擦模式的准备顺序不变。

连接与释放均在暂停边界进行，不插入额外物理 step，不删除 target。每个世界有独立控制器和请求身份；约束丢失、状态不一致或服务超时会中止本次采样，不用缓存连接状态冒充有效抓持。有效连接可维持原 attempt，但不覆盖传感数据缺失，不虚构双指接触；放置认证要求约束已解除。

每个完成的控制步从 ROS `/so101/grasp_attachment_state` 等待时间戳至少到达该步末尾的约束状态，不逐步启动命令行查询进程。插件在暂停边界立即发布精确仿真时间，暂停期间继续 heartbeat；创建/解除连接仍经由 Gazebo 服务确认。旧消息不能冒充本步状态。

这是仿真辅助任务，成功不能解释为已学会自然摩擦防滑。日志保留 `grasp_attachment_*`、`grasp_attachment` 的请求/事件/角度及 `success_physics`；`unassisted_success` 区分未使用辅助的成功。连接日志还记录 `training_stage`、`release_mode`、`release_allowed`。critic 的现有辅助状态和执行指令已包含上述奖惩所需信息，本次不增加维度。执行契约用 `stage_grasp_attachment` 记录 S2/S3 对默认释放规则的覆盖，用 `attachment_command_reward` 记录参考角度和比较规则；因此旧释放/奖励规则的完整 PPO checkpoint 不会被静默当作相同任务恢复。仅加载 ACT 权重开始新训练仍可使用，包括监督训练 0005000。

S2/S3 持续连接与指令奖惩已做定向离线测试，本次未启动 Gazebo 或训练验证。

通用入口的 S2–S4 默认训练预算分别为 10 秒，可用 `S2_BUDGET_S`、`S3_BUDGET_S`、`S4_BUDGET_S` 覆盖；当前完整六阶段启动器的配置表将三者均设为 8 秒。S5 HOME 预算继承 `S0_BUDGET_S`；S5 成功仍要求“放置持续有效 + S0 HOME 成功”。

准备失败记录为 `stage_initialization`，保留阶段、控制步骤和原因；此时 policy step、discarded transition 都为零。发生 Gazebo 崩溃时另保留崩溃诊断。`STAGE_INITIALIZATION_TIMEOUT_S` 可覆盖父进程的等待上限：原 S0/S1 默认 180 秒，含后续阶段默认 600 秒。准备控制器自身仍有独立的阶段与总时限。

## 六阶段采样与更新

`s0_s5` 每个 wave 只采一个阶段；按各阶段已收有效步数相对于其目标占比的完成程度补充，必要时减少该 wave 的世界数量。`STAGE_STEP_WEIGHTS` 按 `S0:S1:S2:S3:S4:S5` 顺序配置六个正数，允许小数，不允许零、负数或非有限值。完整六阶段启动器默认 `1:3:2:2:1:1`，对应 10%、30%、20%、20%、10%、10%；可以在脚本配置表中修改，也可以在启动命令中覆盖：

```bash
STAGE_STEP_WEIGHTS="1:3:2:2:1:1" bash scripts/finetune_act_ppo_s0_s5.sh
```

若 `ROLLOUT_STEPS=2400`，六阶段基准目标分别是 240、720、480、480、240、240 步；实际总量和各阶段数量可因完整 episode 与比例容差而超过或偏离基准值。默认允许各阶段占比偏差 ±5 个百分点，且每个参与阶段都必须实际采过。完整 episode 不截断、不复制、不补假 step。达到 `ROLLOUT_STEPS` 且比例满足后才运行 PPO；超出 `STAGE_MAX_ROLLOUT_MULTIPLIER` 仍不满足比例会报错停止，不更新不满足比例的 batch。这些比例控制成功阶段过滤前的采样数据，不改变奖励系数。

通用入口和 Python CLI 默认仍为 `1:1:1:1:1:1`，保留旧等分行为。Python 参数为 `--stage-step-weights`；配置和采样日志保存原始权重或归一化目标比例。续训签名记录实际比例：等价倍数如 `1:3:2:2:1:1` 与 `2:6:4:4:2:2` 兼容，更改比例会拒绝直接恢复原完整 PPO 状态；仅加载 ACT 权重开始新训练不受此限制。旧 S0/S1 的 `S0_STEP_FRACTION` 行为保持不变。

`STAGE_SUCCESS_SKIP_THRESHOLD=0.8` 对每个选中的阶段生效。某阶段本 batch 有效 episode 的成功率达到阈值，其全部样本从本次 actor、critic、探索及辅助损失更新中排除，下一采样 batch 跳过该阶段一次。其余阶段按原有权重重新归一化并采满 `ROLLOUT_STEPS`。例如 `1:3:2:2:1:1` 跳过 S0 后，S1–S5 按 `3:2:2:1:1` 分配，目标约为 33.33%、22.22%、22.22%、11.11%、11.11%。被跳过的阶段在再下一批重新参与，按当批实际参与阶段重新归一化；其他阶段仍可能因新达标而跳过。若所有阶段都达标，只消费一个空跳过 batch，然后恢复采样；空 batch 不创建世界、不增加 PPO update。

目标始终先在源区生成；各 world/episode 的 seed 独立且可复现。目标在 episode 内不删除。每个 episode 完整保存后关闭世界，需要下一个 wave 才创建替代世界。PPO 计算期间不保留等待使用的 Gazebo 世界。

S2–S5 使用自己的奖励、成功判定、阶段 ID 和日志，不会被当成 S1。TensorBoard 继续按每个 update 记录，checkpoint 间隔独立配置。

## 旧模型与恢复

含 S2–S5 的训练追加了固定尺寸的 critic 状态，包括六阶段 ID、阶段计时与奖励历史。S0/S1 在同一 `s0_s5` run 中也使用这个尺寸。旧的 S0/S1 完整 PPO checkpoint 与新 critic 形状不同，不能直接 `RESUME_CHECKPOINT` 续训；可将其 `pretrained_model` 作为 ACT 起点，重新创建 critic 和 optimizer。相同六阶段契约的完整 checkpoint 才能恢复；奖励、准备参数、调度或维数不一致会拒绝。

配置、默认奖励、准备参数和 context 字段会写入本次训练的配置及恢复契约。2026-10-07 的独立 Gazebo 检查已完成一条辅助抓持准备链：真实抓取、连接、抬升、搬运、实际开夹后解除、放置、空爪抬升、S5 起点交接及交接后两步。报告位于 `codex_workspace/reports/grasp_attachment_live_20261007_204247_81dbbfd2/summary.json`。这是单场景自动准备验证，不代表 ACT 已学会这些阶段，也不能代替多世界和多目标位置的成功率评估。

上述 Gazebo 记录来自取消额外闭合 2° 之前；取消后的指令保持行为已做定向离线测试，本次未重跑 Gazebo。
