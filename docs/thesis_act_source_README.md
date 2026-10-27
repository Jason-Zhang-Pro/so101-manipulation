# SO-101 基于 ACT 的视觉抓取与放置

## 项目简介

本项目使用 SO-101 机械臂与 Action Chunking Transformer（ACT），实现红色方块的视觉抓取和放置，并研究数据覆盖、图像增强、KL 权重与时间集成对模型性能的影响。

## 系统配置

- 机械臂：SO-101 follower
- 摄像头：固定相机 + 手眼相机
- 图像尺寸：640 × 480
- 训练设备：NVIDIA GPU
- 软件：Ubuntu、PyTorch、LeRobot

## 数据集

主要数据集包含：

- 150 个演示 episode
- 32,233 帧
- 15 FPS
- 15 个方块起始位置
- 6 维关节状态和动作
- `fixed` 与 `handeye` 两路图像

原始数据、训练输出和 checkpoint 不保存在 GitHub 仓库中。

## ACT 配置

| 参数 | 值 |
|---|---:|
| Vision backbone | ResNet18（ImageNet 预训练） |
| Batch size | 16 |
| Chunk size | 50 |
| Action steps | 1 |
| Temporal ensembling | 0.005 |
| KL weight | 0.5 |
| Learning rate | 1e-5 |
| Weight decay | 1e-4 |
| AMP | 开启 |
| Training steps | 100,000 |
| Checkpoint interval | 5,000 |

训练使用轻度亮度、对比度、饱和度和清晰度增强，不使用翻转、裁剪或 Hue 变化。

## 主要改动

- 为训练流程加入 TensorBoard 日志。
- 记录总损失、L1 损失、KL 损失、梯度范数和学习率。
- 记录 ACT 潜变量 `z`、`mu`、对数方差与后验标准差。
- 支持潜变量标量曲线和直方图。
- 提供 SO-101 checkpoint 实机运行脚本。

## TensorBoard

```bash
python -m tensorboard.main \
  --logdir=outputs/train/<RUN_NAME>/tensorboard \
  --port=6006 \
  --bind_all
```

浏览器打开：

```text
http://localhost:6006
```

## 实机测试

指定模型目录和 checkpoint：

```bash
MODEL_RUN_DIR="outputs/train/<RUN_NAME>" \
EPISODE_SECONDS=20 \
MAX_RELATIVE_TARGET=3 \
./run_act_red_v2.sh 015000
```

首次测试应使用训练分布内的位置，并确保机械臂周围没有障碍物。

## 评估方法

每个 checkpoint 在相同位置和相机条件下测试，记录：

- 抓取成功率
- 完整任务成功率
- 抓取失败、掉落和放置失败次数
- 已见位置与未见位置的性能差异

建议重点比较 `020000`、`040000`、`060000`、`080000` 和 `100000`。

## 仓库说明

本项目基于 [Hugging Face LeRobot](https://github.com/huggingface/lerobot) 开发。当前毕设代码位于 `thesis-act-so101` 分支。
