# 硬件恢复指南

历史实机版本为 4482430（real-vision-tracking-v0.1），当前软件基准为 01b7e13bf98ec803bfc4c3ca02469bc0faecb61c。本阶段仅核对已有配置和 H1.0 记录。以下步骤供重新连接后使用。

## 相机、时钟与外参

1. 在 Windows 保持 Ubuntu-22.04 WSL2 运行，usbipd list 查 D435 的当次 BUSID；历史为 1-7、8086:0b07。设备共享后使用 usbipd attach --wsl --busid <当前BUSID>。WSL 中用 lsusb -d 8086:0b07、lsusb -t 核对枚举和链路，确认 video 组权限、设备无重复占用。
2. 在独立域、localhost 环境按 H1.0 CAMERA_CONFIG.md 单独启动 rs_launch.py，彩色与深度 640×480×30，enable_sync、align_depth.enable 为 true。使用新日志目录。检查三路图像、彩色 CameraInfo、时间差、深度非零比例及 /tf_static。H1.0 的 36 秒记录可作对照，源戳与接收单调时间需分别保留。
3. 使用 sha256sum -c ~/vision_diagnostics/h1_0/calibration/original_sha256.txt 核对六点原始 CSV、标定代码及发布脚本。核对 transforms.json 中 T(world←optical) = T(world←camera_link) × T(camera_link←optical)，内部边取本机对应序列号的 TF。归档 world←camera_link 平移约 (0.24186, 0.02420, 0.05149) m。历史残差 RMS 2.516 mm。用独立参考点复核当前安装后再评价物理精度。

## 两条感知启动路线

先加载已核验的感知 overlay，核对模型路径、实际模块、权重和真实类别。以下每个 ros2 run 独立终端运行。示例用 cup，与 E3.5 的类别一致。项目根目录设为 P；切换路线时先结束旧的感知节点以避免重名发布。

```bash
P="$HOME/openmanipulator-x-vision"
rgb=/camera/camera/color/image_raw
native=false
mask=/object_segmenter/instance_mask
ros2 run object_detector object_detector --ros-args --params-file "$P/vision_ws/src/object_detector/config/yoloe_text.yaml" -p model:="$P/vision_ws/yoloe-26s-seg.pt" -p prompt_classes:=cup -p image_topic:="$rgb" -p enable_native_mask:="$native"
ros2 run object_tracker object_tracker --ros-args -p enable_native_mask:="$native"
"$P/.venv-sam2/bin/python" -m object_segmenter.segmenter_node --ros-args -p image_topic:="$rgb"
ros2 run object_localizer mask_localizer --ros-args --params-file "$P/vision_ws/src/object_localizer/config/realsense.yaml" -p mask_topic:="$mask" -p world_frame:=camera_color_optical_frame
```

Native Mask 路线把 detector 和 tracker 的 native 都设为 true，mask 设为 /object_tracker/native_instance_mask，直接启动 mask_localizer，省去 SAM2 进程。两路均核对 /object_tracker/detections 与掩码归属，深度源为 /camera/camera/aligned_depth_to_color/image_raw，内参为 /camera/camera/color/camera_info。SAM2 直接使用专用 Python 模块以传入真实 RGB 话题；项目中的 02–05 演示脚本含仿真参数。

## 真相机配虚拟臂

先仅录制带源戳的图像、CameraInfo、TF 和感知输出；关闭相机域的发布后，在独占 localhost 域按原时序回放视觉输入。虚拟域继续使用 fixture.py 的关节、硬件反馈、TF 和唯一虚拟动作服务器。当前短回归默认合成输入；接入实测数据需单独替换 stimulus 发布端，检查相机系、TF 所有者与重复发布，禁用合成目标。真实与虚拟域间只传视觉数据。先验证 dry-run 零动作，再用虚拟动作核对目标身份、源龄、反馈与取消；原始录包时钟与回放时钟须显式映射，保留原戳证据。该方案尚未进行现场联调。

## 真实运动前

由操作者核对倒挂安装与夹具固定、工作空间、关节限位和初始姿态；确认硬件电源、线缆、关节/控制器新鲜反馈与实际姿态一致。检查动作图唯一且无外来目标，READY 身份与操作者选定目标一致，授权、丢失暂停、故障闭锁均生效。先独立确认停车/断能方式可触及、有人监护、工作区清空；反馈过期或 STOP NOT VERIFIED 时由独立手段确认停稳。真实运动需单独授权。
