# 整合验证记录

验证日期：2026-10-27。环境为 Windows、Python 3.12.14、CPU PyTorch 2.7.1，
不是原项目的 Ubuntu 22.04 / ROS Humble / Gazebo Fortress 环境。

## 已通过

- 源码 smoke：597 个 Python 文件可编译，16 个机器人 XML 可解析。
- 原主体 ACT workflow 回归：采集和训练命令生成、无执行副作用、抓取/放置结果计数。
- 所有 `scripts/*.sh` 经 Git Bash `bash -n` 语法检查。
- 项目包及共享 LeRobot 的 `--no-deps --no-build-isolation -e` 安装。
- 核心回归 102 项全部通过：ACT 有界输出、PPO、梯度累积、潜变量监测及配置入口。
  日志开关前后的训练损失和梯度一致，默认内部返回接口不变。

核心测试命令：

```bash
python -m pytest tests/unit/test_integration_contract.py \
  tests/unit/test_act_bounded_output.py tests/unit/test_act_ppo.py \
  tests/unit/test_supervised_training_updates.py -q
```

## 扩展测试的实际结果

对可收集的测试运行后，2671 项通过、89 项失败、4 项跳过。
另有 33 个文件因缺少 `rclpy` 不能收集；Linux 六阶段 shell 启动器测试未运行。
没有通过添加 ROS 伪模块或修改原测试来消除环境限制。

89 项失败归因：

| 原因 | 数量 |
|---|---:|
| 运行中才导入 ROS 模块，缺少 rclpy | 28 |
| Windows 不支持当前目录 fsync/O_DIRECTORY 操作 | 50 |
| Windows 换行、路径表示与 POSIX 预期不同 | 3 |
| Windows shell/编码差异 | 8 |

失败涉及的 10 个原测试文件与 Capstone 来源版本逐字节一致。
分组清单、错误摘要和不可收集文件见 [integration_validation.json](integration_validation.json)。
原始 pytest 日志及 JUnit 报告保存在本地 `outputs/integration_validation/`，不纳入 Git。
这不是全量测试通过记录，也不是 ROS/Gazebo 可运行认证。

## 未验证与外部输入

- 原主体模型有 121 处缺失的外部 mesh 引用；smoke 明确报告
  `runtime_assets_complete=false`，不将其算作可运行资源完整。
- 未启动 ROS、Gazebo、多世界训练或机械臂，未运行模型训练或 sim-to-real。
- 原数据集、ACT 权重、PPO optimizer/critic 状态均需另行提供。
- 需在目标 Ubuntu/ROS 环境执行完整 `python -m pytest tests/unit`、ROS 构建、
  reset/相机/控制器 live smoke，再做单世界 PPO 更新及实机回归。

这里的“核心测试”验证源码接口和小型 CPU 模型；不能证明真实旧 checkpoint
在两套场景中的行为相同，更不能证明 PPO 已提升实机成功率。
