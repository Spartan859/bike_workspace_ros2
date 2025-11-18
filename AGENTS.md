## March_Turn_camera.py → ROS 2 迁移计划

本计划将现有 Python 程序 `March3/March_Turn_camera.py`（自平衡控制 + ODrive + 舵机 + IMU + 相机AI + Flask）迁移到 ROS 2 Humble，并与当前工作区已有包集成：`imu_ros2`, `servo_rs485_ros2`, `balance_controller_ros2`，以及子模块 `odrive_ros2_control`。

---

**目标与范围**
- 目标：在 ROS 2 中实现可启动、可配置、可观测的自平衡控制系统，支持无硬件模拟与实车运行两种模式。
- 范围：
	- 控制算法（角度环/角速度环/飞轮零速校正）参数化并纳入 ROS 参数。
	- ODrive 通过 `ros2_control` 硬件接口接入；舵机通过 `servo_rs485_ros2` 服务调用。
	- IMU 使用 `imu_ros2` 串口节点；相机 AI 独立 ROS 节点对外发布安全状态。
	- Launch 统一编排（含 mock 硬件模式）。

---

**现状梳理（Python 程序）**
- 线程模型：IMU 解码线程 + 控制环线程（~500Hz）+ 可选 Flask Web + 可选可视化线程。
- 设备/库：
	- IMU：串口 + 自定义 `ImuParser`。
	- ODrive：`odrive` Python 库（USB/libusb）控制飞轮与驱动轮（速度模式、加速度斜坡）。
	- 舵机：`Servo_RS485.Servo`，提供角度设置（含速度限制/分步）。
	- 相机：`CameraAI`（MindIR yolov8x）做人/障碍检测，提供安全判断。
- 控制逻辑：三环（角度环/角速度环/飞轮零速校正）+ 动态零点（随速度/转向偏置漂移）+ 安全限幅/超时保护。

---

**目标 ROS 2 架构**
- 节点与插件：
	- `imu_ros2/imu_serial_node.py|cpp`（已存在）：发布 IMU 数据（建议最终对齐 `sensor_msgs/Imu`）。
	- `controller_manager` + `odrive_ros2_control`（已存在）：以 URDF/xacro 绑定 ODrive 硬件（飞轮/驱动轮关节）。
	- `balance_controller_ros2`（已存在，C++ controller 插件）：承载平衡控制算法，读 IMU/状态，写入关节速度命令。
	- `servo_rs485_ros2`（已存在，C++）：通过服务 `SetAngle`, `GetAngle`, `PingServo` 控制转向舵机。
	- `camera_ai_node`（新增，Python rclpy）：封装 `CameraAI`，发布 `detections`、`/safety/status`，提供 `CheckPathSafe` 服务（可选）。
	- `web_ui_node`（可选，Python）：如需保留 Flask，改为只做前端与 ROS 服务/话题交互。

- 接口设计（建议）：
	- 话题：
		- `/imu/data_raw` 或 `/imu/data`：`sensor_msgs/Imu`（若当前为自定义 `ImuData.msg`，可增加桥接节点转换为标准消息）。
		- `/joint_states`：`joint_state_broadcaster` 输出。
		- `/safety/status`：`std_msgs/String`（OK/WARNING/STOP: <reason>）。
		- `/camera/detections`：`vision_msgs/Detection2DArray`（如依赖简化，可用自定义）。
	- 服务/动作：
		- `/servo_rs485/set_angle`：沿用现有 `SetAngle.srv`。
		- `/camera/check_path_safe`：自定义服务（输入目标动作，输出布尔+原因）。
	- 参数：
		- 将 `config_dynzero.yaml` 中 PID/限幅/零点漂移/舵机参数迁移到 `balance_controller_ros2/config/params.yaml`。
		- 摄像头与模型路径、可视化开关等迁移到 `camera_ai_node` 的 YAML。

---

**两条迁移路线**
1) 目标架构（推荐，直接对齐 ros2_control）
	 - 使用现有 `balance_controller_ros2` 插件作为控制算法的唯一实现：
		 - 将 `March_Turn_camera.py` 中的三环控制/动态零点/安全保护迁移到 C++ 插件（当前已有框架，按参数补齐算法与订阅）。
		 - 输入：IMU 话题；输出：飞轮/驱动轮关节速度命令；转向：通过 `servo_rs485_ros2` 服务。
	 - 优点：性能好、架构规范、与 `odrive_ros2_control` 天然集成；缺点：初期 C++ 改造工作量略大。

2) 过渡路径（快速可运行）
	 - 新建 `balance_controller_py`（rclpy 节点）：复用现有 Python 控制逻辑，订阅 IMU，直接用 `odrive` Python 库写速度 + 通过 `servo_rs485_ros2` 服务控舵。
	 - 待验证稳定后，再逐步替换为 C++ `balance_controller_ros2` 插件，移除 Python 直接硬件 IO。
	 - 优点：快速验证；缺点：与 `ros2_control` 并行两套路径、后续需要切换。

> 当前仓库已存在 `balance_controller_ros2` 及 odrive 硬件/URDF 接入，建议沿用“目标架构”直接完善 C++ 控制插件。

---

**落地步骤（按目标架构）**
1. 参数清单迁移
	 - 将下列参数映射到 `balance_controller_ros2/config/params.yaml`：
		 - 角度环：`angle_kp/ki/kd`
		 - 角速度环：`angle_v_kp/ki/kd`
		 - 飞轮零速校正：`flywheel_speed_kp/ki/integral_limit`
		 - 目标/限幅：`flywheel_speed_limit`, `flywheel_accel_limit`, `roll_diff_limit`
		 - 动态零点：`machine_middle_angle_init`, `bike_turn_scale_deg`, `bike_speed_scale_deg`, `middle_angle_recitfy_limit_deg`
		 - 舵机：`servo_center_deg`, `servo_middle_range`, `servo_range_each_side_deg`, `servo_speed_dps`, `servo_step_deg`, `servo_step_time`

2. 控制插件完善（`balance_controller_ros2`）
	 - 在 `update()` 中补齐三环控制与动态零点逻辑；
	 - 订阅 IMU（若使用自定义 `imu_ros2/ImuData`，可在插件或独立节点转换为 `sensor_msgs/Imu` 标准）；
	 - 通过 command interfaces 写入 `flywheel_joint` 与 `drive_joint` 速度；
	 - 通过 `servo_rs485_ros2` 服务设置舵机角度（或另建 steering 小节点按控制输出转发）。

3. 相机 AI 节点（`camera_ai_node`）
	 - rclpy 节点封装 `CameraAI`：
		 - 发布 `/safety/status` 与 `/camera/detections`；
		 - 提供 `CheckPathSafe` 服务供控制层调用（可选：控制器仅订阅状态，简化耦合）。
	 - 参数：模型路径、可视化开关、检测周期、阈值、最小安全距离等。

4. 硬件接入（已具备基础）
	 - ODrive：通过 `odrive_ros2_control` + `balance_controller_ros2/urdf/*.ros2_control.xacro`；
	 - 舵机：`servo_rs485_ros2`；
	 - IMU：`imu_ros2` 串口节点；
	 - Launch 参数化：`use_mock_hardware`（无设备时用 `mock_components`）。

5. 统一启动（bringup）
	 - 扩展/新增 `balance_controller_ros2/launch/balance_controller.launch.py`：
		 - 参数：`use_mock_hardware`、`use_camera`、`visualize`、`params_file`、`urdf_file`；
		 - 流程：`robot_state_publisher` → `controller_manager`（加载 URDF）→ 生成 `/controller_manager` 服务 → spawn `joint_state_broadcaster` 与 `balance_controller`；可选启动 `camera_ai_node`。

6. 监控与诊断
	 - `rqt_graph`, `ros2 topic echo`, `ros2 control list_controllers`；
	 - 记录 ODrive 初始化失败时的降级行为（mock/告警不中断）。

---

**无硬件（Mock）与实车运行**
- 无硬件：
	- 使用 `balance_controller_ros2/urdf/bike_robot_mock.ros2_control.xacro`；
	- 控制器运行但关节命令仅在内存中模拟；配合 `imu_ros2` 可发布仿真 IMU 数据（可新增 small publisher）。

- 实车：
	- ODrive：确认 `lsusb`、`udev` 规则与权限；
	- 连接后 `controller_manager` 不应因 `LIBUSB_ERROR_NO_DEVICE` 崩溃；
	- `servo_rs485_ros2` 串口权限；
	- IMU 串口端口与波特率配置。

---

**构建与启动示例**
- Windows（本地开发，cmd）：
```bat
call C:\dev\ros2_humble\local_setup.bat
colcon build --symlink-install --packages-select imu_ros2 servo_rs485_ros2 balance_controller_ros2
call install\setup.bat
```

- 远程 Linux（zsh，已记录使用 /opt/ros/humble/setup.zsh）：
```bash
source /opt/ros/humble/setup.zsh
colcon build --symlink-install --packages-select imu_ros2 servo_rs485_ros2 balance_controller_ros2
source install/setup.zsh
```

- 启动（Mock 硬件）：
```bash
ros2 launch balance_controller_ros2 balance_controller.launch.py use_mock_hardware:=true
```

- 启动（实车）：
```bash
ros2 launch balance_controller_ros2 balance_controller.launch.py use_mock_hardware:=false
```

---

**测试计划**
- 单元/组件：
	- 控制器：以录制/合成 IMU 序列回放，验证环路输出稳定性与限幅；
	- 相机：以测试帧/离线视频验证 `safety/status` 的状态机与阈值。
- 集成：
	- Mock 硬件：确保 `controller_manager` 与控制器加载/切换正常；
	- 实车：按“ODrive 诊断与 udev”步骤连通，逐项验证飞轮/驱动/舵机动作与安全停机。

---

**时间线（参考）**
- D1：接口对齐与参数迁移；完善/确认 URDF+xacro（含 mock）。
- D2：在 `balance_controller_ros2` 内补齐控制算法（IMU 订阅、命令写入、服务调用）。
- D3：联调 mock；补齐相机节点与安全状态联动。
- D4：实车接入与阈值调参；撰写运行手册。

---

**验收标准**
- Mock 与实车两种模式均可启动与切换；
- 控制器参数可通过 YAML/动态参数调整；
- `ros2_control` 控制链闭合：IMU → 控制器 → 关节命令（驱动/飞轮）/舵机服务；
- 安全状态可拦截推进/触发停机；
- 关键话题/服务在 `rqt_graph` 中可见且文档化。

---

**后续工作**
- 如需：用 `sensor_msgs/Imu` 统一消息类型，增加桥接或改造 `imu_ros2`。
- 将 Flask Web 替换为轻量 ROS Web 前端或保留为独立节点。
- 增加记录/回放工具链（`rosbag2`）用于算法离线调参。

---

如需，我可以：
- 参数映射并补齐 `balance_controller_ros2` 控制算法；
- 新建 `camera_ai_node` 包与其 YAML；
- 改造 Launch 支持 `use_mock_hardware` 与可选相机启动；
- 提供最小化仿真 IMU 发布器用于端到端验证。

---

## servo_rs485_ros2 接口文档（节点/话题/服务/参数）

**概览**
- 包名：`servo_rs485_ros2`
- 可执行节点：`servo_node`（节点名：`/servo_rs485_node`）
- 用途：通过 RS485 驱动方向舵机，提供设置角度与读取角度的服务，并周期性发布当前角度。

**参数（declare_parameter）**
- `port`（string，默认 `/dev/ttyUSB1`）：串口设备路径。
- `servo_id`（int，默认 `1`）：舵机 ID。
- `baudrate`（int，默认 `115200`）：串口波特率。
- `timeout`（double，默认 `0.5`）：串口超时（秒）。

> 说明：这些参数在启动时读取，运行中修改不会自动重连串口。

**话题**
- `/servo_angle`（`std_msgs/Float64`，发布周期约 2 Hz）
	- `data`：当前舵机角度（单位：度）。

**服务**
- `/ping_servo`（`servo_rs485_ros2/srv/PingServo`）
	- Request：无
	- Response：`online`（bool）舵机是否在线

- `/set_angle`（`servo_rs485_ros2/srv/SetAngle`）
	- Request：
		- `degree`（float64）目标角度（单位：度，数学正方向与硬件安装相关）
		- `time_ms`（int32）运动时间（毫秒），用于实现匀速/缓动效果（取决于固件支持）
	- Response：
		- `success`（bool）设置是否成功

- `/get_angle`（`servo_rs485_ros2/srv/GetAngle`）
	 - `/set_angle_with_speed`（`servo_rs485_ros2/srv/SetAngleWithSpeed`）
		 - Request：
			 - `degree`（float64）目标角度（度）
			 - `speed_dps`（float64）期望匀速转动速度（度/秒，<=0 时直接跳转目标角度）
			 - `step_interval_ms`（int32）内部拆分每步的时间间隔（毫秒，<1 自动提升为10ms；建议 30~100ms）
		 - Response：
			 - `success`（bool）指令已执行（动作在服务调用期间完成）
		 - 行为：服务调用阻塞执行，内部按速度拆分为多步调用基础 `set_angle` 写入位置；结束后保证最终精确到达。
		 - 注意：长距离 + 低速度可能阻塞数秒，若需异步可在上层封装 action。
	- Request：无
	- Response：
		- `degree`（float64）当前角度（单位：度）
		- `success`（bool）读取是否成功

**启动示例**
- Linux：
```bash
ros2 launch servo_rs485_ros2 test_servo_launch.py port:=/dev/ttyUSB0 servo_id:=1
```

- Windows（cmd.exe）：
```bat
ros2 launch servo_rs485_ros2 test_servo_launch.py port:=COM5 servo_id:=1
```

- Windows（COM≥10，如遇打开失败可尝试）：
```bat
ros2 launch servo_rs485_ros2 test_servo_launch.py port:"\\\\.\\COM10" servo_id:=1
```

**服务调用示例**
```bash
# 探测舵机在线
ros2 service call /ping_servo servo_rs485_ros2/srv/PingServo {}

# 设置角度：-10 度，时长 500 ms
ros2 service call /set_angle servo_rs485_ros2/srv/SetAngle "{degree: -10.0, time_ms: 500}"

# 匀速平滑旋转到 +15 度，速度 20 度/秒，步间隔 50ms
ros2 service call /set_angle_with_speed servo_rs485_ros2/srv/SetAngleWithSpeed "{degree: 15.0, speed_dps: 20.0, step_interval_ms: 50}"

# 读取当前角度
ros2 service call /get_angle servo_rs485_ros2/srv/GetAngle {}
```

**调试与排错**
- 使用 `ros2 param get /servo_rs485_node port` 校验参数生效。
- 若角度话题无更新，确认节点在运行且串口权限/连线正确（Linux 下 `ls -l /dev/ttyUSB*`）。
- 若服务返回失败或超时，检查 `servo_id` 与布线、终端电阻、供电、电平匹配等。

