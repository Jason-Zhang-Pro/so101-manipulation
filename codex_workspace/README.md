# 项目辅助源码

本发布副本保留 checks、commands、scripts、tests、patches 和 diagnostics 顶层 Python 诊断源码。
运行报告、checkpoint、诊断输出、临时文件不包含在本副本中。

- `scripts/show_stage_preparation_gui.py`：S0–S5 自动准备起点目视检查。
- `scripts/check_stage_preparation_live.py`、`check_grasp_attachment_live.py`：Gazebo live 检查，需显式手动运行。
- `scripts/check_s0_s5_configuration.py`：配置检查。
- `scripts/evaluate_latest_stages.py`：历史 S0/S1 对比辅助工具；正式分阶段评估使用根目录 `scripts/evaluate_act_stages.sh`。
- `commands/finetune_good40_posture_distance.sh`：保留已有命令及测试依赖，其中模型路径需要用户自行提供。

部分历史诊断脚本默认依赖未随源码发布的模型、报告或本机路径；它们不是新克隆后自动执行的任务。
LeRobot 定制代码已直接包含在 `lerobot/`。已有 patches 用于溯源，不要重复应用。
