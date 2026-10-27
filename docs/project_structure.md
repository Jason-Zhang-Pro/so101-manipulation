# Capstone 项目结构

本文档说明 SO-101 Capstone 项目的正式目录结构、各目录职责、当前使用状态、生成文件边界和维护原则。

项目根目录：

```bash
/media/tiger/New Volume/ubuntufile/capstone
```

本文档用于回答：

- 正式代码放在哪里
- ROS 2 与 LeRobot 如何分工
- 哪些目录是当前 active
- 哪些目录只是 scaffold
- 哪些目录是生成或缓存内容
- 哪些目录是只读参考资料
- 哪些文件不应提交 Git
- 新功能应放到哪里

---

## 1. 顶层结构

当前项目顶层结构：

```text
capstone/
├── configs/
├── data/
├── docs/
├── hardware/
├── huggingface/
├── lerobot/
├── notebooks/
├── outputs/
├── references/
├── ros2_ws/
├── scripts/
├── src/
├── tests/
├── tools/
├── .gitignore
├── pyproject.toml
└── README.md
```

当前最主要的 active areas：

```text
README.md
docs/
hardware/
ros2_ws/
lerobot/
references/
```

其中正式 ROS 2 / Gazebo 代码集中在：

```text
ros2_ws/
```

真实机器人与 LeRobot runtime 集中在：

```text
lerobot/
hardware/
```

---

## 2. 根目录文件

### `README.md`

路径：

```text
capstone/README.md
```

用途：

- 项目总览
- 研究目标
- 当前状态
- 主要目录
- 快速入口
- 系统边界
- 常用命令
- 文档索引

根 README 应保持高层，不应塞入所有实现细节。

详细内容应链接到：

```text
docs/
ros2_ws/README.md
hardware/README.md
```

---

### `.gitignore`

路径：

```text
capstone/.gitignore
```

当前用于排除：

- ROS 2 build outputs
- LeRobot source / external tree
- Hugging Face cache
- Official reference archive
- Raw / processed data
- Outputs
- Machine-specific hardware notes
- Python cache
- Editor backup
- Migration backup

不要使用 `git add -f` 绕过忽略规则，除非已经明确决定改变 repository policy。

---

### `pyproject.toml`

当前文件存在，但尚未建立正式 project-level Python packaging configuration。

在定义以下内容前，不应随意填写：

- Project package name
- Build backend
- Python dependencies
- Tool configuration
- Editable installation
- Test configuration
- Formatting / linting configuration

当顶层 Python source architecture 明确后，再集中完善。

---

## 3. `ros2_ws/`

路径：

```text
capstone/ros2_ws/
```

状态：

```text
Active
```

用途：

- ROS 2 Humble workspace
- Gazebo Fortress simulation
- SO-101 robot description
- `ros2_control`
- Cameras
- World
- Controllers
- Target cube manager
- Formal simulation launch

结构：

```text
ros2_ws/
├── build/
├── install/
├── log/
├── README.md
└── src/
    ├── so101_description/
    ├── so101_gazebo/
    └── so101_bringup/
```

正式说明：

```text
ros2_ws/README.md
docs/gazebo_simulation.md
```

---

## 4. ROS 2 generated directories

以下目录由 `colcon` 生成：

```text
ros2_ws/build/
ros2_ws/install/
ros2_ws/log/
```

用途：

| Directory | Purpose |
|---|---|
| `build/` | Build intermediates |
| `install/` | Installed package resources and executables |
| `log/` | Build logs |

这些目录：

- 不应手动编辑
- 不应提交 Git
- 可以在 clean rebuild 时删除
- 可以从 `src/` 重新生成

正式 source 只位于：

```text
ros2_ws/src/
```

---

## 5. `so101_description`

路径：

```text
ros2_ws/src/so101_description/
```

状态：

```text
Active
```

用途：

- SO-101 Xacro
- Mesh assets
- Links
- Joints
- Joint limits
- Cameras
- Materials
- `ros2_control`

关键文件：

```text
urdf/so101.urdf.xacro
meshes/
CMakeLists.txt
package.xml
README.md
```

不要在该 package 中加入：

- Gazebo world
- Target cube manager
- Complete simulation launch
- Real robot serial configuration

---

## 6. `so101_gazebo`

路径：

```text
ros2_ws/src/so101_gazebo/
```

状态：

```text
Active
```

用途：

- Gazebo world
- Table model
- Target cube model
- Controller YAML
- Target cube YAML
- Target cube manager node

关键目录：

```text
config/
models/
scripts/
worlds/
```

关键文件：

```text
config/controllers.yaml
config/target_cube.yaml
worlds/empty.sdf
scripts/target_cube_manager.py
```

---

## 7. `so101_bringup`

路径：

```text
ros2_ws/src/so101_bringup/
```

状态：

```text
Active
```

用途：

- Formal simulation launch
- Package resource discovery
- Xacro processing
- Gazebo startup
- Robot spawn
- Camera bridge
- Target cube manager startup
- Controller startup

关键文件：

```text
launch/sim.launch.py
```

正式命令：

```bash
ros2 launch so101_bringup sim.launch.py
```

---

## 8. `hardware/`

路径：

```text
capstone/hardware/
```

状态：

```text
Active documentation
```

用途：

- 真实机械臂硬件映射
- Persistent serial paths
- Persistent camera paths
- USB connection notes
- Machine-specific device checks

结构：

```text
hardware/
├── README.md
└── device_ids/
    └── IDlist.txt
```

### `hardware/README.md`

正式共享硬件说明。

### `hardware/device_ids/IDlist.txt`

本机原始设备记录。

该文件：

- Machine-specific
- 被 `.gitignore` 排除
- 不作为团队共享配置
- 不替代正式 README

---

## 9. `docs/`

路径：

```text
capstone/docs/
```

状态：

```text
Active documentation
```

当前结构：

```text
docs/
├── gazebo_simulation.md
├── project_structure.md
├── real_robot_setup.md
├── target_cube_manager.md
├── troubleshooting.md
└── migration/
    ├── old_cap_robot_assets.txt
    ├── so101_urdf_source.txt
    └── backups/
```

---

## 10. 正式技术文档

### `docs/gazebo_simulation.md`

系统级 Gazebo 说明：

- Package architecture
- World
- Robot
- Cameras
- Controllers
- Target cube
- Build
- Validation

### `docs/target_cube_manager.md`

Target cube 系统详细说明：

- Model
- Spawn region
- Services
- Topics
- Entity ID
- State
- Deletion
- Stale pose suppression

### `docs/real_robot_setup.md`

真实硬件系统说明：

- Leader / Follower
- Cameras
- Persistent paths
- LeRobot environment
- USB bandwidth
- Validation

### `docs/troubleshooting.md`

已确认问题和处理方法：

- ROS 2
- Gazebo
- Xacro
- Controllers
- Cameras
- Target cube
- Real hardware
- Git

### `docs/project_structure.md`

本文件，说明 repository organization。

---

## 11. `docs/migration/`

路径：

```text
capstone/docs/migration/
```

状态：

```text
Historical provenance
```

用途：

- 记录旧项目资源来源
- 记录迁移路径
- 记录模型和 assets 的来源关系

当前文件：

```text
old_cap_robot_assets.txt
so101_urdf_source.txt
```

这些文件是历史记录，不应被当成当前 runtime configuration。

例如 migration 记录中的：

```text
so101.urdf
```

已经演变为当前正式：

```text
so101.urdf.xacro
```

---

## 12. Migration backups

路径：

```text
docs/migration/backups/
```

状态：

```text
Ignored backup area
```

当前包含历史 backup：

```text
so101.urdf.before_path_fix
```

该目录被 `.gitignore` 排除。

后续不应继续依赖 backup 文件保存版本历史。

正式版本历史应使用 Git。

---

## 13. `lerobot/`

路径：

```text
capstone/lerobot/
```

状态：

```text
Active external project tree
```

用途：

- Real robot connection
- Leader-Follower teleoperation
- Real camera input
- Dataset collection
- Imitation learning
- Future policy training / evaluation

该目录来自 LeRobot 项目，并在当前 Capstone 环境中使用。

当前 repository policy：

```text
lerobot/
```

被顶层 `.gitignore` 排除。

因此当前 Capstone Git repository 不直接跟踪该完整目录。

不要修改 LeRobot 上游 README 来充当 Capstone 文档。

Capstone-specific instructions 应放在：

```text
capstone/README.md
docs/real_robot_setup.md
```

---

## 14. `huggingface/`

路径：

```text
capstone/huggingface/
```

状态：

```text
Cache / external data
```

用途：

- Hugging Face cache
- Model downloads
- Dataset cache
- Hub-related files

该目录可能很大，并被 `.gitignore` 排除。

不要把 downloaded model weights 或 cache 提交到 Capstone Git。

需要记录 model identity 时，应记录：

- Repository ID
- Revision / commit
- Configuration
- Local usage instructions

而不是提交完整 cache。

---

## 15. `references/`

路径：

```text
capstone/references/
```

状态：

```text
Read-only reference archive
```

当前重要目录：

```text
references/SO-ARM101_original/
```

该目录保存：

- SO-ARM101 官方教程
- 官方 source materials
- Technical information
- Drawings
- Original reference resources

它不是普通旧项目 backup。

维护原则：

- 保持只读
- 不作为正式 runtime path
- 不直接修改
- 不从中加载正式 ROS 2 assets
- 用于核对官方结构和设计资料

该目录被 `.gitignore` 排除。

---

## 16. `configs/`

路径：

```text
capstone/configs/
```

状态：

```text
Scaffold / future central configuration
```

当前包含规划中的分类：

```text
camera/
experiments/
reward/
robot/
task/
training/
```

预期用途：

- Project-level experiment configuration
- Camera configuration
- Reward settings
- Robot settings
- Task definitions
- Training configuration

当前正式 Gazebo 配置仍位于：

```text
ros2_ws/src/so101_gazebo/config/
```

不要为了“统一目录”而移动已经被 ROS 2 package 正式使用的 YAML。

只有在顶层 training / experiment framework 建立后，再使用 `configs/`。

---

## 17. `data/`

路径：

```text
capstone/data/
```

状态：

```text
Data organization scaffold
```

当前规划包括：

```text
data/
├── metadata/
├── raw/
└── processed/
```

用途：

| Directory | Purpose |
|---|---|
| `metadata/` | Dataset descriptions and manifests |
| `raw/` | Original captures |
| `processed/` | Converted or processed datasets |

当前 `.gitignore` 排除：

```text
data/raw/
data/processed/
```

Dataset metadata 是否提交，应根据文件大小、隐私和复现需求单独决定。

---

## 18. `outputs/`

路径：

```text
capstone/outputs/
```

状态：

```text
Generated outputs
```

预期用途：

- Training runs
- Evaluation results
- Checkpoints
- Plots
- Logs
- Exported policies
- Temporary experiment outputs

该目录被 `.gitignore` 排除。

正式实验结果如果需要长期记录，应考虑：

- 单独的 summary document
- Small metadata file
- External artifact storage
- Hugging Face repository
- Release archive

不要直接提交大量 binary output。

---

## 19. `notebooks/`

路径：

```text
capstone/notebooks/
```

状态：

```text
Scaffold
```

预期用途：

- Data inspection
- Offline analysis
- Plotting
- Experiment exploration
- Debugging notebooks

当前不应把 notebook 作为正式 runtime entry point。

成熟逻辑应移入：

```text
src/
scripts/
tools/
```

Notebook 中如果包含大 output，应在提交前清理。

---

## 20. `scripts/`

路径：

```text
capstone/scripts/
```

状态：

```text
Scaffold
```

预期用途：

- Project-level executable scripts
- Dataset utilities
- Training wrappers
- Evaluation entry points
- Environment setup helpers

不要把 ROS 2 package executable 随意放到顶层 `scripts/`。

ROS 2-specific executable 应放在对应 package 中，例如：

```text
ros2_ws/src/so101_gazebo/scripts/
```

---

## 21. `src/`

路径：

```text
capstone/src/
```

状态：

```text
Scaffold
```

预期用途：

- Capstone-owned Python package
- Shared training code
- Environment wrappers
- Dataset code
- Policy integration
- Non-ROS reusable modules

当前正式 ROS 2 source 不在该目录，而在：

```text
ros2_ws/src/
```

顶层 `src/` 的 package layout 应在 `pyproject.toml` 完善时统一定义。

---

## 22. `tests/`

路径：

```text
capstone/tests/
```

状态：

```text
Scaffold
```

预期用途：

- Unit tests
- Integration tests
- Configuration tests
- Dataset schema tests
- Simulation interface tests

未来建议测试：

- Joint name consistency
- Controller joint order
- Xacro generation
- Required package resources
- Target cube parameter consistency
- Device mapping format
- Dataset schema

当前不要创建没有实际执行环境的空测试模板。

---

## 23. `tools/`

路径：

```text
capstone/tools/
```

状态：

```text
Scaffold
```

预期用途：

- Developer utilities
- Repository maintenance
- Conversion tools
- Validation helpers
- One-off but reusable engineering tools

与 `scripts/` 的建议区别：

- `scripts/`：直接执行项目 workflow
- `tools/`：开发、检查和维护辅助工具

在功能尚未出现前，不必为该目录创建 README 或 placeholder code。

---

## 24. Active、Scaffold 与 Generated 分类

### Active

当前正式使用：

```text
README.md
docs/
hardware/
ros2_ws/
lerobot/
references/
```

### Scaffold

已经规划但尚未形成稳定 workflow：

```text
configs/
data/metadata/
notebooks/
scripts/
src/
tests/
tools/
```

### Generated / ignored

运行时或下载生成：

```text
ros2_ws/build/
ros2_ws/install/
ros2_ws/log/
huggingface/
outputs/
data/raw/
data/processed/
```

---

## 25. Formal runtime paths

### Gazebo simulation

```text
ros2_ws/
```

启动：

```bash
ros2 launch so101_bringup sim.launch.py
```

### Real robot / LeRobot

```text
lerobot/
hardware/
```

### Project documentation

```text
README.md
docs/
hardware/README.md
ros2_ws/README.md
ros2_ws/src/*/README.md
```

### Official reference

```text
references/SO-ARM101_original/
```

---

## 26. 新文件放置规则

### 新 ROS 2 launch

放入对应 package：

```text
ros2_ws/src/<package>/launch/
```

### 新 Gazebo model

放入：

```text
ros2_ws/src/so101_gazebo/models/
```

### 新 Gazebo world

放入：

```text
ros2_ws/src/so101_gazebo/worlds/
```

### 新 robot description

放入：

```text
ros2_ws/src/so101_description/urdf/
```

### 新 mesh

放入：

```text
ros2_ws/src/so101_description/meshes/
```

### 新 ROS 2 manager node

放入负责该功能的 ROS package。

### 新 project-level Python module

未来放入：

```text
src/
```

但应先定义 `pyproject.toml` package layout。

### 新实验配置

非 ROS package-specific 配置未来放入：

```text
configs/
```

### 新技术说明

放入：

```text
docs/
```

---

## 27. 不要跨目录混用

不要：

- 从 `references/` 直接加载正式 mesh
- 从旧项目路径加载 runtime assets
- 把真实 camera path 写入 Gazebo launch
- 把 Gazebo topic 当作真实 camera device
- 把 ROS 2 package script 放到顶层 `scripts/`
- 把 generated output 放入 source directory
- 手动编辑 `install/`
- 用 backup 文件代替 Git history
- 把 LeRobot 上游文档改成 Capstone 项目文档

---

## 28. 文件命名原则

建议：

- Markdown 使用小写 snake_case
- ROS package 使用小写 snake_case
- Python module 使用小写 snake_case
- YAML 名称表达明确功能
- Gazebo model directory 与 model name 一致
- 避免 `final`、`new`、`copy`、`latest` 等模糊名称

不推荐：

```text
file_final.py
file_final2.py
new_world.sdf
test_copy.yaml
working_version.py
```

版本历史由 Git 管理。

---

## 29. 文档层级

建议文档层级：

### Level 1 — Project overview

```text
README.md
```

### Level 2 — Subsystem overview

```text
ros2_ws/README.md
hardware/README.md
```

### Level 3 — Package documentation

```text
ros2_ws/src/so101_description/README.md
ros2_ws/src/so101_gazebo/README.md
ros2_ws/src/so101_bringup/README.md
```

### Level 4 — Detailed technical documents

```text
docs/gazebo_simulation.md
docs/target_cube_manager.md
docs/real_robot_setup.md
docs/troubleshooting.md
docs/project_structure.md
```

### Historical provenance

```text
docs/migration/
```

---

## 30. Git boundaries

当前不应提交：

```text
ros2_ws/build/
ros2_ws/install/
ros2_ws/log/
lerobot/
huggingface/
references/
outputs/
data/raw/
data/processed/
hardware/device_ids/IDlist.txt
docs/migration/backups/
```

以及：

```text
__pycache__/
*.py[cod]
*~
*.save
*.swp
```

---

## 31. Repository inspection

查看顶层：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone"

find . \
  -maxdepth 2 \
  -mindepth 1 \
  -not -path './.git*' \
  -not -path './ros2_ws/build*' \
  -not -path './ros2_ws/install*' \
  -not -path './ros2_ws/log*' \
  | sort
```

查看 ROS source：

```bash
find ros2_ws/src \
  -maxdepth 3 \
  -mindepth 1 \
  | sort
```

查看 tracked files：

```bash
git ls-files
```

查看 ignored files：

```bash
git status --ignored --short
```

该命令可能产生较多输出。

---

## 32. 当前结构中的已知待办

当前仍需后续决定：

- 顶层 `pyproject.toml` 的正式用途
- 顶层 Python package name
- `configs/` 与 training workflow 的对应关系
- Dataset metadata schema
- `scripts/`、`src/`、`tools/` 的明确边界
- Tests framework
- Training output organization
- Model checkpoint storage policy
- LeRobot version / revision record
- Project-level environment reproducibility file

在这些内容明确前，不应为填充目录而创建无意义 placeholder files。

---

## 33. 维护原则

1. 一个功能放在一个明确 subsystem
2. ROS 2 runtime 内容放在 ROS package
3. 真实硬件映射与 Gazebo 配置分离
4. Official reference 保持只读
5. Generated content 不进入 source tree
6. 文档记录正式路径，不依赖聊天历史
7. Migration files 只记录历史
8. Git 保存版本，不使用大量 backup copies
9. 修改前检查现状
10. 修改后运行对应验证
11. 用户确认后再 commit

---

## 34. Git 检查

文档完成前暂不 commit。

最终统一检查：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone"

git status --short
git diff --check
git diff
```

确认：

- 新文档位于正确目录
- 没有修改 official references
- 没有提交 machine-specific files
- 没有 build output
- 没有 cache
- 没有临时 backup
- 没有无关代码修改
