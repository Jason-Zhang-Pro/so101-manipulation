#!/usr/bin/env bash
# S0–S5 同一 ACT 的分阶段并行采样/PPO。仅加载较好40的ACT权重；
# critic、optimizer、探索和update计数重新建立，输出到全新训练目录。
# 学习率采用较好40结束时的有效值；KL最小缩放按新基准换算以保留绝对下限。
# 支持环境覆盖：PRETRAINED_MODEL、WORLD_COUNT、GAZEBO_GUI、EPISODE_COUNT、STAGE_STEP_WEIGHTS。
# 其余设置固定在此脚本中，避免之前终端的S0/S1配置意外污染本次训练。
set -euo pipefail

ppo_print_config=false
case "$#:$*" in
    0:) ;;
    1:--print-config) ppo_print_config=true ;;
    1:--help|1:-h)
        printf '%s\n' \
            'Usage: bash scripts/finetune_act_ppo_s0_s5.sh [--print-config]' \
            'Overrides: PRETRAINED_MODEL, WORLD_COUNT, GAZEBO_GUI, EPISODE_COUNT, STAGE_STEP_WEIGHTS.' \
            '--print-config validates required files and prints shell settings only; it never starts training.'
        exit 0
        ;;
    *) printf 'Unknown arguments: %s\n' "$*" >&2; exit 2 ;;
esac

ppo_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ppo_project_root="$(cd -- "$ppo_script_dir/.." && pwd -P)"

# 只允许上述五项显式覆盖；固定使用本脚本所在项目和六阶段配置。
declare -A ppo_launch_config=(
    [PROJECT_ROOT]="$ppo_project_root"
    [PRETRAINED_MODEL]="${PRETRAINED_MODEL:-$ppo_project_root/output_simulation/finetuned/act_ppo_20261001_182324_from_checkpoints/checkpoints/update_000040/pretrained_model}"
    [WORLD_COUNT]="${WORLD_COUNT:-4}"
    [GAZEBO_GUI]="${GAZEBO_GUI:-false}"
    [EPISODE_COUNT]="${EPISODE_COUNT:-100000000}"
    [DEVICE]=cuda
    [PYTHONNOUSERSITE]=1
    [ACT_MAX_ACTION_STEPS]=1

    [REWARD_PROFILE]=pick_place_v2
    [REWARD_CONFIG]="$ppo_project_root/config/pick_place_reward.json"
    [TRAINING_STAGES]=s0_s5
    [STAGE_CONFIG]="$ppo_project_root/config/act_ppo_stages_s0_s5.json"
    [TASK_MODE]=mixed
    [IDLE_EMPTY_PROBABILITY]=0.1
    [IDLE_COMPLETED_PROBABILITY]=0.1
    # 六阶段有效采样步数权重：S0:S1:S2:S3:S4:S5。跳过阶段后按剩余权重重新分配。
    [STAGE_STEP_WEIGHTS]="${STAGE_STEP_WEIGHTS-1:3:2:2:1:1}"
    # S0_STEP_FRACTION仅兼容旧S0/S1入口，不参与六阶段比例。
    [S0_STEP_FRACTION]=0.4
    [STAGE_RATIO_TOLERANCE]=0.05
    [STAGE_MAX_ROLLOUT_MULTIPLIER]=4
    [STAGE_SUCCESS_SKIP_ENABLED]=true
    [STAGE_SUCCESS_SKIP_THRESHOLD]=0.8
    [S0_BUDGET_S]=5.0
    [S0_HOME_HOLD_S]=1.0
    [S1_BUDGET_S]=10.0
    [S1_REACH_HOLD_S]=0
    [S2_BUDGET_S]=8.0
    [S3_BUDGET_S]=8.0
    [S4_BUDGET_S]=8.0

    # S0：HOME 保持  S1：抬升后接近  S2：抓住并稳定抬起  S3：持物搬运  S4：放下并撤离  S5：回 HOME

    # S5的HOME成功判定与预算直接继承S0。物理准备不占阶段时间或PPO样本。
    [STAGE_INITIALIZATION_TIMEOUT_S]=600.0

    # 接触确认有效抓持后刚性连接固定夹爪与方块；S2/S3采样期间持续连接。
    # 其他阶段与自动准备：新指令和实际角度均达到确认时实测角度+2°才解除。
    # 此辅助物理模式会写入训练签名，与原自然摩擦抓持实验区分。
    [ASSISTED_GRASP_ENABLED]=true

    [ENVIRONMENT_RANDOMIZATION]=false
    [ENVIRONMENT_RANDOMIZATION_CONFIG]="$ppo_project_root/config/environment_randomization.json"
    [ENVIRONMENT_RANDOMIZATION_SEED]=0
    [IMAGE_AUGMENTATION]=true
    [IMAGE_AUGMENTATION_CONFIG]="$ppo_project_root/config/image_augmentation_s0_s1.json"
    [REACH_CURRICULUM]=false
    [CURRICULUM_WINDOW_SIZE]=20
    [CURRICULUM_PASS_RATE]=0.7
    [CURRICULUM_REHEARSAL_PROBABILITY]=0.2
    [TARGET_DELAY_S]=0
    [FINAL_TARGET_ABSENT_S]=0
    [SEED]=0

    [BASE_ROS_DOMAIN_ID]=20
    [GZ_PARTITION_PREFIX]=gz_env_
    [WORLD_STARTUP_MAX_ATTEMPTS]=3
    [WORLD_RESTART_MAX_ATTEMPTS]=0
    [WORLD_DEFERRED_RESTART_MAX_ATTEMPTS]=3
    [PREFLIGHT_CLEANUP_ENABLED]=true
    [PREFLIGHT_CLEANUP_TIMEOUT_S]=5.0
    [STARTUP_FEEDBACK_TIMEOUT_S]=30.0
    [MAX_CONSECUTIVE_EPISODE_FAILURES]=3
    # 仅full模式使用此上限；六阶段使用上面的秒数预算。
    [MAX_STEPS_PER_EPISODE]=1200
    [ROLLOUT_STEPS]=2400
    [MINIBATCH_SIZE]=30
    [PPO_EPOCHS]=2

    [GAMMA]=0.997
    [GAE_LAMBDA]=0.95
    [CLIP_RATIO]=0.2
    [TARGET_KL]=50
    [KL_BACKOFF_MAX_RETRIES]=10
    [KL_BACKOFF_FACTOR]=0.5
    # 最小缩放比例为0.0064；ACT基准3.125e-6对应最小有效LR为2e-8。
    [KL_BACKOFF_MIN_SCALE]=0.0064
    [VALUE_LOSS_COEFFICIENT]=0.5
    [ENTROPY_COEFFICIENT]=0
    [MEAN_BOUNDS_COEFFICIENT]=0.1
    [RAW_MEAN_BOUNDS_COEFFICIENT]=0.1
    [MAX_GRADIENT_NORM]=0.5
    [CRITIC_MAX_GRADIENT_NORM]=0.5
    [EXPLORATION_MAX_GRADIENT_NORM]=0.5
    [ACTOR_LEARNING_RATE]=3.125e-6
    [ACTOR_BACKBONE_LEARNING_RATE]=3.125e-7
    [CRITIC_LEARNING_RATE]=2e-4
    [EXPLORATION_LEARNING_RATE]=3.125e-6
    [CRITIC_IMAGE_HEIGHT]=96
    [CRITIC_IMAGE_WIDTH]=128
    [CRITIC_HIDDEN_DIM]=256

    # 保持当前有界动作采样及近目标探索缩小；不恢复旧单侧探索。
    [EXPLORATION_PROFILE]=final_good_act_near_target
    [INITIAL_LOG_STD]=-3.0
    [MINIMUM_LOG_STD]=-9.0
    [MAXIMUM_LOG_STD]=-2.5
    [GRIPPER_MAXIMUM_LOG_STD]=-1.5
    [CHECKPOINT_EVERY_UPDATES]=5
    [TENSORBOARD_PORT]=6006
)

# 与实际启动同一套清理操作；--print-config也输出这些unset，便于只读核对。
ppo_unset_parameters=(
    RESUME_CHECKPOINT S0_PROBABILITY
    SO101_ASSISTED_GRASP_ENABLED
    IDLE_EXPLORATION_LOG_STD MOVING_EXPLORATION_LOG_STD
    MANIPULATION_EXPLORATION_LOG_STD HOME_HOLD_EXPLORATION_LOG_STD
    MOVING_GRIPPER_EXPLORATION_LOG_STD MANIPULATION_GRIPPER_EXPLORATION_LOG_STD
)
for ppo_legacy_group in STATE1 STATE2 STATE3 STATE4_NOT_HOME STATE4_HOME; do
    for ppo_legacy_suffix in EXPLORATION_ENABLED ACTOR_UPDATE_ENABLED EXPLORATION_LOG_STD; do
        ppo_unset_parameters+=("${ppo_legacy_group}_${ppo_legacy_suffix}")
    done
done
for ppo_parameter in "${ppo_unset_parameters[@]}"; do
    unset "$ppo_parameter"
done

for ppo_parameter in WORLD_COUNT EPISODE_COUNT; do
    if [[ ! "${ppo_launch_config[$ppo_parameter]}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'Error: %s must be a positive integer.\n' "$ppo_parameter" >&2
        exit 2
    fi
done
case "${ppo_launch_config[GAZEBO_GUI]}" in
    true|false) ;;
    *) printf 'Error: GAZEBO_GUI must be true or false.\n' >&2; exit 2 ;;
esac

# 仅使用标准库共享解析器；打印配置时也验证权重，不启动conda、ROS或训练。
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    python3 - "${ppo_launch_config[STAGE_STEP_WEIGHTS]}" \
    "$ppo_project_root/src/so101_capstone/training/act_ppo_stage_schedule.py" <<'PY'
import runpy
import sys
# 直接载入独立调度模块，避免training包初始化时导入PyTorch。
parse_stage_step_weights = runpy.run_path(sys.argv[2])["parse_stage_step_weights"]
try:
    parse_stage_step_weights(sys.argv[1])
except ValueError as error:
    print(f"Error: STAGE_STEP_WEIGHTS: {error}", file=sys.stderr)
    sys.exit(2)
PY

for ppo_parameter in "${!ppo_launch_config[@]}"; do
    export "$ppo_parameter=${ppo_launch_config[$ppo_parameter]}"
done
for ppo_required_file in \
    "$STAGE_CONFIG" "$REWARD_CONFIG" "$ENVIRONMENT_RANDOMIZATION_CONFIG" \
    "$IMAGE_AUGMENTATION_CONFIG" "$PRETRAINED_MODEL/config.json" \
    "$PROJECT_ROOT/scripts/finetune_act_ppo_so101.sh"; do
    if [[ ! -s "$ppo_required_file" ]]; then
        printf 'Error: required file missing or empty: %s\n' "$ppo_required_file" >&2
        exit 2
    fi
done
if [[ ! -s "$PRETRAINED_MODEL/model.safetensors" && ! -s "$PRETRAINED_MODEL/pytorch_model.bin" ]]; then
    printf 'Error: no ACT weights in %s\n' "$PRETRAINED_MODEL" >&2
    exit 2
fi
bash -n "$PROJECT_ROOT/scripts/finetune_act_ppo_so101.sh"

if [[ "$ppo_print_config" == true ]]; then
    printf '# Resolved S0-S5 launcher settings; no training or simulation started.\n'
    for ppo_parameter in "${ppo_unset_parameters[@]}"; do
        printf 'unset %q\n' "$ppo_parameter"
    done
    while IFS= read -r ppo_parameter; do
        printf 'export %s=%q\n' "$ppo_parameter" "${ppo_launch_config[$ppo_parameter]}"
    done < <(printf '%s\n' "${!ppo_launch_config[@]}" | LC_ALL=C sort)
    exit 0
fi

exec bash -e -o pipefail "$PROJECT_ROOT/scripts/finetune_act_ppo_so101.sh"
