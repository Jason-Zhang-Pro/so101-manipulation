# 本地 LeRobot 源码说明

> 本文保留 Capstone 原快照的来源记录。整合版本在此基础上加入了主体相机修复和
> 毕设潜变量/TensorBoard 监测；当前整合说明见 [integration_zh.md](../integration_zh.md)。

本发布副本直接保存当前项目的 `lerobot/` 运行源码，不是 Git submodule，不含嵌套 `.git`，无需再用上游最新版替换。

| 来源 | 标识 |
|---|---|
| 上游 | <https://github.com/huggingface/lerobot> |
| 版本声明 | 0.3.4 |
| 比较基线 | `882c80d446a63a44868c67ae535467af32ce0e80` |
| 原本地 LeRobot HEAD | `feaea6a02218846e36fe034c2fabab82aa52eabb` |
| 导出方式 | 当前工作区快照，含本地提交与未提交源码修改 |

HEAD 本身不足以重现该快照，因为还包含工作区修改。来源记录见 [`source_provenance.json`](source_provenance.json)，基线到当前源码差异见 [`lerobot_local_changes.patch`](lerobot_local_changes.patch)。

**相关修改已经包含在 `lerobot/`，不要重复应用 patch。** 此 patch 用于审阅与溯源；原 `codex_workspace/patches/act_bounded_output.patch` 是较早、范围更小的历史补丁，不能代表当前所有修改。

## 相对基线的八个源码变化

路径均相对于 `lerobot/`：

| 文件 | 变化范围 |
|---|---|
| `src/lerobot/cameras/utils.py` | Windows OpenCV 相机后端选择 |
| `src/lerobot/configs/policies.py` | 梯度累积配置、特征选择、临时配置加载兼容处理 |
| `src/lerobot/motors/motors_bus.py` | 同步读写默认重试与失败日志 |
| `src/lerobot/policies/act/configuration_act.py` | 可选动作上下界、平滑参数与校验 |
| `src/lerobot/policies/act/modeling_act.py` | 原生 ACT 输出限位、队列/TE 结果处理和有界监督输出 |
| `src/lerobot/policies/act/bounded_actions.py` | 新增平滑动作限位与舍入越界保护 |
| `src/lerobot/scripts/train.py` | 按 microbatch 样本数加权的梯度累积与计数 |
| `src/lerobot/utils/logging_utils.py` | 日志显示完整 step 数值 |

这些是现状记录，不代表本次发布新增，也不代表各平台均已验证。保留 Windows 相机后端改动不等于 ROS/Gazebo 流程支持 Windows。动作限位是可选 checkpoint 配置；不能仅因源码支持有界输出，就断言任意旧模型已启用限位。

## 许可证与排除项

保留运行源码、配置、文本说明、相关测试代码及 [LeRobot Apache-2.0 许可证](../../lerobot/LICENSE)。原作者与版权头保留。

源码包不含嵌套 Git、缓存、生成文件、权重、视频、上游非 Python 二进制测试 fixtures、宣传媒体 `media/`、上游 `.github/` 工作流及针对已排除资源的 `.gitattributes` LFS 规则。因此不是上游仓库的完整镜像。部分上游测试和图片依赖额外资源，需要从对应基线另行取得，不能直接覆盖本地定制运行代码。

发布整理另有两类仅文档变化：

- 上游 README 一条已过期的带签名视频 URL 改为上游 README 永久链接。
- 11 个在原磁盘上表现为“只含相对路径的普通文本”的文档别名，展开为其指向的文档正文，避免发布后只显示一个路径。

上述不改运行源码。主 Capstone 来源 HEAD 为 `c63540392a6d3de1ec059ffe4e5b59b24ef740b4`，与本页 LeRobot 比较基线及本地 HEAD 是三个不同标识。纯源码目录无 `.git`；主仓历史另保存在准备包的 `local_history/capstone.bundle`，不属于本次纯源码上传内容。
