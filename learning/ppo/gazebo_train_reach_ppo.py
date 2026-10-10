
import argparse
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/so101_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from so101_gazebo_ros_env import SO101GazeboROSEnv


class StepStatsCallback(BaseCallback):
    def __init__(self, print_freq=1000, total_timesteps=500000):
        super().__init__()
        self.print_freq = print_freq
        self.total_timesteps = total_timesteps
        self.last_print_step = 0
        self.start_time = None

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_distances = []
        self.episode_successes = []
        self.episode_gripper_open = []
        self.reward_component_window = {}

    def _on_training_start(self):
        self.start_time = time.time()
        print(f"[train] ROS + Gazebo PPO training start, target_timesteps={self.total_timesteps}")
        print("[train] Make sure robot_state_publisher, Ignition Gazebo, and controllers are already running.")

    def _on_step(self):
        infos = self.locals.get("infos", [])

        for info in infos:
            distance = float(info.get("distance_to_cube", np.nan))
            gripper = float(info.get("gripper_open_norm", np.nan))

            if "episode" in info:
                ep = info["episode"]
                self.episode_rewards.append(float(ep["r"]))
                self.episode_lengths.append(int(ep["l"]))
                self.episode_distances.append(distance)
                self.episode_successes.append(float(info.get("success", False)))
                self.episode_gripper_open.append(gripper)

            for name, value in info.get("reward_components", {}).items():
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                self.reward_component_window.setdefault(name, []).append(value)

        if self.num_timesteps - self.last_print_step >= self.print_freq:
            self.last_print_step = self.num_timesteps

            if self.episode_rewards:
                window = min(len(self.episode_rewards), 20)

                mean_reward = np.mean(self.episode_rewards[-window:])
                mean_len = np.mean(self.episode_lengths[-window:])
                mean_dist = np.nanmean(self.episode_distances[-window:])
                success_rate = np.mean(self.episode_successes[-window:])
                mean_gripper = np.nanmean(self.episode_gripper_open[-window:])

                print(
                    f"[train] step={self.num_timesteps} "
                    f"ep_rew_mean={mean_reward:.3f} "
                    f"ep_len_mean={mean_len:.1f} "
                    f"dist_mean={mean_dist:.4f} "
                    f"gripper_mean={mean_gripper:.3f} "
                    f"success_rate={success_rate:.3f}"
                )

                self.logger.record("custom/ep_rew_mean_window20", mean_reward)
                self.logger.record("custom/ep_len_mean_window20", mean_len)
                self.logger.record("custom/dist_mean_window20", mean_dist)
                self.logger.record("custom/success_rate_window20", success_rate)
                self.logger.record("custom/gripper_mean_window20", mean_gripper)

                if self.reward_component_window:
                    component_means = {
                        name: float(np.mean(values))
                        for name, values in self.reward_component_window.items()
                        if values
                    }
                    for name, value in component_means.items():
                        self.logger.record(f"reward_components/{name}", value)

                    negative_components = sorted(
                        (
                            (name, value)
                            for name, value in component_means.items()
                            if value < 0.0
                        ),
                        key=lambda item: item[1],
                    )[:5]
                    if negative_components:
                        component_text = " ".join(
                            f"{name}={value:.3f}"
                            for name, value in negative_components
                        )
                        print(f"[train] reward_components_mean {component_text}")

                    self.reward_component_window.clear()
            else:
                print(f"[train] step={self.num_timesteps}, waiting for first finished episode...")

        return True


def make_env(max_steps=180, env_id=0, ros_namespace=""):
    def _init():
        env = SO101GazeboROSEnv(
            gui=False,
            max_steps=max_steps,
            action_scale=0.030,
            use_done_action=True,
            control_dt=0.07,
            env_id=env_id,
            ros_namespace=ros_namespace,
            use_tf_kinematics=True,
            randomize_cube=True,
        )
        return Monitor(env)

    return _init


def make_training_env(max_steps, num_envs, ros_namespaces, use_subproc):
    env_fns = [
        make_env(
            max_steps=max_steps,
            env_id=env_id,
            ros_namespace=ros_namespaces[env_id],
        )
        for env_id in range(num_envs)
    ]

    if num_envs == 1:
        return env_fns[0]()

    if use_subproc:
        return SubprocVecEnv(env_fns, start_method="spawn")

    return DummyVecEnv(env_fns)


def evaluate(model, episodes=3, max_steps=180, ros_namespace=""):
    # Do not run this in parallel with training. It controls the same Gazebo robot.
    env = make_env(max_steps=max_steps, env_id=0, ros_namespace=ros_namespace)()

    rewards = []
    distances = []
    successes = []

    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            info = {}

            for step in range(max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    break

            rewards.append(total_reward)
            distances.append(float(info.get("distance_to_cube", np.nan)))
            successes.append(float(info.get("success", False)))

            print(
                f"eval episode={ep + 1} "
                f"steps={step + 1} "
                f"reward={total_reward:.3f} "
                f"distance={distances[-1]:.4f} "
                f"success={bool(successes[-1])} "
                f"cube_move={info.get('cube_move', 0):.4f} "
                f"gripper={info.get('gripper_open_norm', 0):.3f}"
            )

        print(
            f"[eval summary] "
            f"reward_mean={np.mean(rewards):.3f} "
            f"distance_mean={np.nanmean(distances):.4f} "
            f"success_rate={np.mean(successes):.3f}"
        )

    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--ros-namespaces",
        type=str,
        default="",
        help="Comma-separated ROS namespaces, e.g. /env_0,/env_1. Required when --num-envs > 1.",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--total-timesteps", type=int, default=500000)
    parser.add_argument("--print-freq", type=int, default=1000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--pretrained",
        type=str,
        default="",
        help="Path to a PyBullet PPO .zip model to warm-start Gazebo sim2sim fine-tuning.",
    )
    parser.add_argument(
        "--dummy-vec",
        action="store_true",
        help="Use DummyVecEnv instead of SubprocVecEnv. Mostly useful for debugging.",
    )
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def parse_ros_namespaces(raw_namespaces, num_envs):
    if num_envs < 1:
        raise ValueError("--num-envs must be at least 1")

    if not raw_namespaces.strip():
        namespaces = [""] * num_envs
    else:
        namespaces = [item.strip() for item in raw_namespaces.split(",")]

    if len(namespaces) != num_envs:
        raise ValueError(
            f"--ros-namespaces must contain exactly {num_envs} entries, got {len(namespaces)}"
        )

    if num_envs > 1 and any(namespace == "" for namespace in namespaces):
        raise ValueError(
            "--num-envs > 1 requires one independent Gazebo/ROS instance per env. "
            "Pass unique namespaces with --ros-namespaces, for example /env_0,/env_1."
        )

    if num_envs > 1 and len(set(namespaces)) != num_envs:
        raise ValueError("Each parallel environment must use a unique ROS namespace.")

    return namespaces


def main():
    args = parse_args()
    ros_namespaces = parse_ros_namespaces(args.ros_namespaces, args.num_envs)

    models_dir = Path("models")
    log_dir = Path("ppo_reach_ros_gazebo_tensorboard")

    model_path = models_dir / "so101_reach_ppo_ros_gazebo"
    best_model_path = models_dir / "so101_reach_ppo_ros_gazebo_latest.zip"
    models_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    env = make_training_env(
        max_steps=args.max_steps,
        num_envs=args.num_envs,
        ros_namespaces=ros_namespaces,
        use_subproc=not args.dummy_vec,
    )

    total_timesteps = args.total_timesteps

    policy_kwargs = dict(
        log_std_init=-2.0,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    if args.pretrained:
        pretrained_path = Path(args.pretrained).expanduser()
        print(f"Loading PyBullet pretrained model from {pretrained_path}")
        model = PPO.load(
            str(pretrained_path),
            env=env,
            device="cpu",
            tensorboard_log=str(log_dir),
        )
    elif args.resume and best_model_path.exists():
        print(f"Loading existing ROS + Gazebo model from {best_model_path}")
        model = PPO.load(
            str(best_model_path),
            env=env,
            device="cpu",
            tensorboard_log=str(log_dir),
        )
    else:
        print("Creating new ROS + Gazebo PPO model.")
        print(
            f"[train] envs={args.num_envs} "
            f"vec={'SubprocVecEnv' if args.num_envs > 1 and not args.dummy_vec else 'Dummy/single'} "
            f"namespaces={ros_namespaces}"
        )
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            gamma=0.98,
            gae_lambda=0.95,
            ent_coef=0.002,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="cpu",
            tensorboard_log=str(log_dir),
        )

    callbacks = [
        StepStatsCallback(
            print_freq=args.print_freq,
            total_timesteps=total_timesteps,
        )
    ]

    # Do not create a second evaluation environment while training against a
    # single Gazebo robot. It would use the same ROS namespace/node name and
    # publish commands to the same controllers, causing duplicate-node warnings
    # and corrupting both training and evaluation. Evaluation is run serially
    # after the training environment has been closed below.
    eval_env = None
    if not args.skip_eval:
        print(
            "[train] Periodic EvalCallback disabled for the shared Gazebo robot; "
            "evaluation will run after training."
        )

    callback = CallbackList(callbacks)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            reset_num_timesteps=False,
            callback=callback,
            tb_log_name="so101_ros_gazebo_run",
        )

    except KeyboardInterrupt:
        print("\n[train] interrupted, saving ROS + Gazebo model...")

    model.save(str(model_path))
    model.save(str(best_model_path).replace(".zip", ""))
    print(f"saved model to {model_path}.zip")
    print(f"saved latest copy to {best_model_path}")

    env.close()
    if eval_env is not None:
        eval_env.close()

    if not args.skip_eval:
        evaluate(
            model,
            episodes=args.eval_episodes,
            max_steps=args.max_steps,
            ros_namespace=ros_namespaces[0],
        )


if __name__ == "__main__":
    main()
