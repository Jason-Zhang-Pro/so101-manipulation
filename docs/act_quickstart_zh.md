# ACT 流程使用说明

统一入口：`learning/act/workflow.py`。
配置文件：`configs/act_workflow.json`。
原始数据、训练输出和模型权重均保留在外部，不提交到仓库。

## 准备环境和配置

在兼容的 LeRobot 环境中运行以下命令，并切换到本仓库目录。
配置中的数据路径可以填写电脑上现有数据的绝对路径。
要继续使用现有黑桌面 150 条数据，把 `dataset.repo_id` 设置为
`jasonz/so101_pick_red_black_15points_v1`，把 `dataset.root` 设置为实际目录。
设备编号可能变化，请先确认 fixed、handeye 和两条机械臂串口。

完整机器环境差异记录在 `environment.md`；ACT 模型来自 LeRobot，仓库整理的是
你的采集、训练、部署、诊断和评估流程。没有复制网络实现或模型权重。

## 采集

```bash
python learning/act/workflow.py collect
python learning/act/workflow.py collect --episodes 10 --execute
python learning/act/workflow.py collect --append --episodes 10 --execute
```

不加 `--execute` 只显示命令；`--append` 追加，不会删除旧数据。
每条演示请记录采样点 P0-P14、光照、抓取是否成功和动作是否有修正。

## 检查数据

```bash
python evaluation/check_dataset.py --root /实际的数据目录
```

检查时间戳、关节数据和视频完整解码。它不能自动证明抓取成功、标注正确或
同点示范风格一致，这些仍需看视频。

## 训练和查看 TensorBoard

```bash
python learning/act/workflow.py train --run act_black15_baseline
python learning/act/workflow.py train --run act_black15_baseline --execute
python learning/act/workflow.py tensorboard --run act_black15_baseline --execute
```

默认匹配已检查的配置：100000 步、batch 8、ImageNet、chunk 50、KL 0.5、TA 0.005，
每2000步保存一次。是否开启图像增强在配置中控制。
训练输出目录不要提前创建，也不自动删除同名目录。

## 实机部署

```bash
python learning/act/workflow.py deploy --checkpoint /实际模型/pretrained_model --run seen_eval
python learning/act/workflow.py deploy --checkpoint /实际模型/pretrained_model --run seen_eval --execute
```

先预览，再执行。执行后显示双摄并保存评估视频、qpos、预测 action 和运行配置。
评估目录位于 `outputs/evaluation/seen_eval`。复位时间用于手动重新摆放物体，
不等于自动回到初始姿态。部署沿用旧接口的无限幅设置，运行前必须确认机械臂姿态。

## 评估

```bash
python learning/act/workflow.py trial-sheet --trial-file outputs/evaluation/seen_eval/trials.csv --execute
python learning/act/workflow.py evaluate --trial-file outputs/evaluation/seen_eval/trials.csv
```

创建试验表后，填入真实观察结果：抓取成功填 `1`，失败填 `0`。
只测抓取时放置结果留空。测完整任务时填放置结果，并记录失败方向。
未见点测试使用 `position_set=unseen`。工具分别统计抓取成功率、完整任务成功率
和置信区间，不会凭空生成实验结果。

## 当前验证范围

统一命令已做离线测试，训练参数已由本机 LeRobot 解析检查。
现有150条数据的完整性检查结果见 `verification.md`。
本次整理没有采集新数据、训练模型、控制机械臂或推送 GitHub。
