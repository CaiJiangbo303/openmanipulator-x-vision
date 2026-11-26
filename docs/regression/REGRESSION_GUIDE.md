# V4 短回归使用指南

在 Ubuntu 22.04、ROS 2 Humble、Cyclone DDS、colcon 与项目 ROS/Python 依赖已安装的机器上运行。入口位于 tests/regression/run_regression.sh。下面从已应用补丁的目录运行；审查包解压后也可从 integration/ 运行。每次选择全新的输出目录。

```bash
repo="$HOME/openmanipulator-x-vision"
kin="$HOME/manipulator-x/grasp_ws/src/manipulator_grasp/manipulator_grasp"
tool=tests/regression/run_regression.sh
ref=01b7e13bf98ec803bfc4c3ca02469bc0faecb61c
bash "$tool" BASELINE L0 --revision "$ref" --repo "$repo" --contract legacy --output /tmp/v41a-baseline-check
bash "$tool" BASELINE L1 --revision "$ref" --repo "$repo" --contract legacy --kinematics "$kin" --output /tmp/v41a-baseline-ros
bash "$tool" CURRENT ALL --repo "$repo" --contract generation --kinematics "$kin" --output /tmp/v41a-current
```

BASELINE 逐个读取指定 40 位提交的 Git blob。CURRENT 读取 --repo 现场工作树内 inputs.txt 明列的文件，包括其中的未提交编辑。可用 --inputs 替换整份相对路径清单。默认 60 个文件限定本次覆盖面；加包、新增模块或修改构建输入后应审查清单。每次均导出到新目录并从 Humble 基础环境构建八个包，结果写入该次 export/build、install 和 log。sources.json 保存逐文件 SHA256 和当次 Git 提交。历史 4482430 不含本组 V2 接口，使用这份清单时缺文件会报告 BLOCKED。

--contract legacy 验证 HEAD 的旧代际语义；generation 验证 B 工作树候选的源戳和代际传递。接口检查在任何分级下都是前置条件。L0 运行字段、序列化、源码、授权和 StateGate 检查；L1 运行消息链、StateManager 和虚拟控制器；ALL 合并两级。L1/ALL 从可配置的 --kinematics 目录读取三个已核准哈希的文件。签名见 kinematics.json。更换运动学版本需单独核对模型与哈希。

若要核对现场安装，在同一命令追加 --site-setup /path/to/site_env.sh，--repo 应指向相应源码。现场安装与源码相容即通过，不一致返回非零。L0/L1 常规路径不加载现场 install。若依赖缺失，查看 results.json 的 BLOCKED 及 build.log 或各组 output.log。任何 FAIL/BLOCKED 均使入口非零退出。每组结果另有 summary.txt 的总览、前后图快照和控制器原始事件。

graph.py 先检查 localhost 随机 ROS 域为空；运动用例将动作重映射到随机命名空间，等待服务与每次发送均核对唯一虚拟服务器。control.py 在真实控制器的输入、授权、发送处观测，fixture.py 记录动作接收、关节反馈、TCP 与服务端取消。每次发送与接收比较整条 JointTrajectory 的 CDR SHA256。stimulus.py 注入回车前后换代、重复源戳、遮挡及过期关节反馈。runner.py 对构建及测试设时限并检查进程组清理。每个 ROS 用例的 after.log 给出退出后节点和动作端点。L2 历史资产见 EXPERIMENT_INVENTORY.md。

复核单个运动用例可在 L1 命令追加 --case repeated 等用例名。单项复核也会重新导出构建并执行接口前置检查，结果写入新的输出目录。一次失败的完整运行应保留原始判定。

## V4.1a 已验证的范围

CURRENT ALL 通过 17/17 项，BASELINE L0 通过 5/5 项。BASELINE L1 通过 14/15 项；repeated 在完整运行和单项复核中两次因 CLOCK JUMP 失败。V2 的 50 ms 时钟安全保护正常触发，正常运行稳定性验收仍未完全通过。

当前清单覆盖接口、抓取消息链和虚拟控制短回归。完整视觉流水线（相机采集、检测、分割和三维定位）不在本清单覆盖范围内。历史实机版本 4482430 缺少当前 V2 的接口和文件，不能直接使用当前完整 V2 测试清单。
