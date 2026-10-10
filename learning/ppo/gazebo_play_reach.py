import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO

from so101_gazebo_ros_env import SO101GazeboROSEnv


def find_model(model_path=""):
    if model_path:
        path = Path(model_path).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"Model not found: {path}")

    candidates = [
        Path("models/best_so101_reach_ppo_ros_gazebo/best_model.zip"),
        Path("models/so101_reach_ppo_ros_gazebo_latest.zip"),
        Path("models/so101_reach_ppo_ros_gazebo.zip"),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("No saved Gazebo PPO model found under models/")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--ros-namespace", type=str, default="")
    parser.add_argument("--control-dt", type=float, default=0.07)
    parser.add_argument("--no-scripted-grasp", action="store_true")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic PPO actions instead of deterministic mean actions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    env = SO101GazeboROSEnv(
        gui=False,
        max_steps=args.max_steps,
        action_scale=0.030,
        use_done_action=True,
        control_dt=args.control_dt,
        ros_namespace=args.ros_namespace,
        use_tf_kinematics=True,
        randomize_cube=True,
    )

    model_path = find_model(args.model)
    modified_at = datetime.fromtimestamp(model_path.stat().st_mtime).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print(f"Loading Gazebo model from: {model_path} (modified: {modified_at})")
    print(
        f"episodes={args.episodes} max_steps={args.max_steps} "
        f"deterministic={not args.stochastic} ros_namespace='{args.ros_namespace}'"
    )
    model = PPO.load(str(model_path), env=env, device="cpu")

    try:
        for episode in range(args.episodes):
            obs, _ = env.reset()
            episode_reward = 0.0

            for step in range(env.max_steps):
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if step % 20 == 0:
                    print(
                        f"episode={episode + 1} "
                        f"step={step + 1} "
                        f"reward={episode_reward:.3f} "
                        f"distance={info.get('distance_to_cube', 0):.4f} "
                        f"success={info.get('success', False)} "
                        f"cube_move={info.get('cube_move', 0):.4f} "
                        f"gripper={info.get('gripper_open_norm', 0):.3f}"
                    )

                if terminated or truncated:
                    if info.get("success", False) and not args.no_scripted_grasp:
                        print("[scripted grasp] success=True, closing gripper and lifting...")
                        env.execute_scripted_grasp()

                    print(
                        f"\n[EPISODE END] "
                        f"episode={episode + 1} "
                        f"steps={step + 1} "
                        f"reward={episode_reward:.3f} "
                        f"distance={info.get('distance_to_cube', 0):.4f} "
                        f"success={info.get('success', False)} "
                        f"early_fail={info.get('early_fail', False)} "
                        f"cube_move={info.get('cube_move', 0):.4f} "
                        f"gripper={info.get('gripper_open_norm', 0):.3f}"
                    )
                    print("-" * 60)
                    break

    finally:
        env.close()


if __name__ == "__main__":
    main()
