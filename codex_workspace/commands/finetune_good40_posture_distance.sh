#!/usr/bin/env bash
# 较好40的ACT权重 + 结束时有效学习率/KL + 最新S1奖励；新建训练状态。
(
    set -e
    cd "/media/tiger/New Volume/ubuntufile/capstone"

    for ppo_legacy_group in STATE1 STATE2 STATE3 STATE4_NOT_HOME STATE4_HOME; do
        for ppo_legacy_suffix in EXPLORATION_ENABLED ACTOR_UPDATE_ENABLED EXPLORATION_LOG_STD; do
            unset "${ppo_legacy_group}_${ppo_legacy_suffix}"
        done
    done
    unset RESUME_CHECKPOINT S0_PROBABILITY
    unset IDLE_EXPLORATION_LOG_STD MOVING_EXPLORATION_LOG_STD
    unset MANIPULATION_EXPLORATION_LOG_STD HOME_HOLD_EXPLORATION_LOG_STD
    unset MOVING_GRIPPER_EXPLORATION_LOG_STD MANIPULATION_GRIPPER_EXPLORATION_LOG_STD

    export PROJECT_ROOT="/media/tiger/New Volume/ubuntufile/capstone"
    export DEVICE=cuda
    export PYTHONNOUSERSITE=1
    export PRETRAINED_MODEL="$PROJECT_ROOT/output_simulation/finetuned/act_ppo_20261001_182324_from_checkpoints/checkpoints/update_000040/pretrained_model"
    # 继承40：chunk=15，执行1步，TE关闭。
    export ACT_MAX_ACTION_STEPS=1

    # 最新S1：先抬升8cm保持0.1秒，再距离严格小于2cm立即成功。
    # 抬升保持首次完成额外+5；保留对准target/路径引导，关闭转腕和平行桌面奖惩。
    export REWARD_PROFILE=pick_place_v2
    export REWARD_CONFIG="$PROJECT_ROOT/config/pick_place_reward.json"
    export TRAINING_STAGES=s0_s1
    export STAGE_CONFIG="$PROJECT_ROOT/config/act_ppo_stages_s1_lift_completion.json"
    export TASK_MODE=mixed
    export IDLE_EMPTY_PROBABILITY=0.1
    export IDLE_COMPLETED_PROBABILITY=0.1

    export S0_STEP_FRACTION=0.4
    export STAGE_RATIO_TOLERANCE=0.05
    export STAGE_MAX_ROLLOUT_MULTIPLIER=4
    # 每阶段成功率达到阈值：本批不更新，下一批跳过一次，其余阶段补满步数。
    export STAGE_SUCCESS_SKIP_ENABLED="${STAGE_SUCCESS_SKIP_ENABLED:-true}"
    export STAGE_SUCCESS_SKIP_THRESHOLD="${STAGE_SUCCESS_SKIP_THRESHOLD:-0.8}"
    export S0_BUDGET_S=5.0
    # S0：六轴HOME误差<8度连续1秒即成功，不检查速度；最长5秒。
    export S0_HOME_HOLD_S=1.0
    export S1_BUDGET_S=10.0
    export S1_REACH_HOLD_S=0

    # 继承40：环境随机化关闭，训练图像增强开启。
    export ENVIRONMENT_RANDOMIZATION=false
    export ENVIRONMENT_RANDOMIZATION_CONFIG="$PROJECT_ROOT/config/environment_randomization.json"
    export ENVIRONMENT_RANDOMIZATION_SEED=0
    export IMAGE_AUGMENTATION=true
    export IMAGE_AUGMENTATION_CONFIG="$PROJECT_ROOT/config/image_augmentation_s0_s1.json"
    export REACH_CURRICULUM=false
    export CURRICULUM_WINDOW_SIZE=20
    export CURRICULUM_PASS_RATE=0.7
    export CURRICULUM_REHEARSAL_PROBABILITY=0.2
    export TARGET_DELAY_S=0
    export FINAL_TARGET_ABSENT_S=0
    export SEED=0

    # 继承40：最多4个世界，无GUI；需要看画面可将false改为true。
    export WORLD_COUNT=4
    export GAZEBO_GUI=false
    export BASE_ROS_DOMAIN_ID=20
    export GZ_PARTITION_PREFIX=gz_env_
    export WORLD_STARTUP_MAX_ATTEMPTS=3
    export WORLD_RESTART_MAX_ATTEMPTS=0
    export WORLD_DEFERRED_RESTART_MAX_ATTEMPTS=3
    export PREFLIGHT_CLEANUP_ENABLED=true
    export PREFLIGHT_CLEANUP_TIMEOUT_S=5.0
    export STARTUP_FEEDBACK_TIMEOUT_S=30.0

    export EPISODE_COUNT=100000000
    export MAX_CONSECUTIVE_EPISODE_FAILURES=3
    # 仅full模式使用步数上限；S0/S1使用上述5秒/10秒。
    export MAX_STEPS_PER_EPISODE=1200
    export ROLLOUT_STEPS=2400
    export MINIBATCH_SIZE=30
    export PPO_EPOCHS=2

    export GAMMA=0.997
    export GAE_LAMBDA=0.95
    export CLIP_RATIO=0.2
    export TARGET_KL=50
    export KL_BACKOFF_MAX_RETRIES=10
    export KL_BACKOFF_FACTOR=0.5
    # 40结束时scale=0.015625。新基准已取有效LR，保留旧绝对LR下限：
    # 1e-4 / 0.015625 = 0.0064；actor/exploration最低2e-8，backbone最低2e-9。
    export KL_BACKOFF_MIN_SCALE=0.0064

    export VALUE_LOSS_COEFFICIENT=0.5
    export ENTROPY_COEFFICIENT=0
    export MEAN_BOUNDS_COEFFICIENT=0.1
    # 直接约束限幅前 ACT 原始输出；任务奖励与动作采样保持当前配置。
    export RAW_MEAN_BOUNDS_COEFFICIENT=0.1
    export MAX_GRADIENT_NORM=0.5
    export CRITIC_MAX_GRADIENT_NORM=0.5
    export EXPLORATION_MAX_GRADIENT_NORM=0.5

    # 继承较好40结束时optimizer中的实际学习率。
    export ACTOR_LEARNING_RATE=3.125e-6
    export ACTOR_BACKBONE_LEARNING_RATE=3.125e-7
    export CRITIC_LEARNING_RATE=2e-4
    export EXPLORATION_LEARNING_RATE=3.125e-6
    export CRITIC_IMAGE_HEIGHT=96
    export CRITIC_IMAGE_WIDTH=128
    export CRITIC_HIDDEN_DIM=256

    # 最新近目标探索缩小；探索参数重新初始化，保留原上下限。
    export EXPLORATION_PROFILE=final_good_act_near_target
    export INITIAL_LOG_STD=-3.0
    export MINIMUM_LOG_STD=-9.0
    export MAXIMUM_LOG_STD=-2.5
    export GRIPPER_MAXIMUM_LOG_STD=-1.5

    export CHECKPOINT_EVERY_UPDATES=5
    export TENSORBOARD_PORT=6006

    # 启动：自动激活lerobot、加载ROS，创建全新输出目录，从update1计数。
    bash -n "$PROJECT_ROOT/scripts/finetune_act_ppo_so101.sh"
    bash -e -o pipefail "$PROJECT_ROOT/scripts/finetune_act_ppo_so101.sh"
)
