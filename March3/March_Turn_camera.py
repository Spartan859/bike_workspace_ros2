from __future__ import annotations

import threading
import time
import os
import yaml
from typing import Optional
from flask import Flask, render_template, request, jsonify
from utils import clamp
import Socket_m
from Servo_RS485 import Servo
from camera_utils_ms import CameraAI

try:
    from Imu import ImuParser
except Exception:
    print("错误：无法导入本地模块 Imu.ImuParser")
    raise


class BalanceCtrlDyn:
    def __init__(self,
                 serial_port: str = 'COM12',
                 baud: int = 115200,
                 axis_drive_index: int = 0,
                 drive_accel: float = 0.5,
                 axis_flywheel_index: int = 1,
                 # 角度环 (外环)
                 angle_kp: float = -7.6,
                 angle_ki: float = 0.0,
                 angle_kd: float = -1.4,
                 # 角速度环 (中环)
                 angle_v_kp: float = 1.4,
                 angle_v_ki: float = 0.0,
                 angle_v_kd: float = 1.1,
                 # 动态零点：机器基准角（静态基准）
                 machine_middle_angle_init: float = -1.6,
                 # 飞轮零速校正环 (最外慢环)
                 flywheel_speed_kp: float = -1.2,
                 flywheel_speed_ki: float = -0.6,
                 flywheel_speed_integral_limit: list[float] = [-200.0, 200.0],
                 flywheel_speed_limit: float = 5.0,
                 flywheel_accel_limit: float = 5.0,
                 roll_diff_limit: float = 0.03,
                 # 舵机几何参数
                 servo_center_deg: float = -10.0,
                 servo_middle_range: float = 2.0,
                 servo_range_each_side_deg: float = 20.0,
                 servo_speed_dps: float = 30.0,  # 舵机转动速度，单位度/秒
                 servo_step_deg: float = 5.0,
                 servo_step_time: int = 50,  # 平滑转动每步间隔时间，单位毫秒
                 # 转向-倾角偏置比例（单位：目标倾角度/舵机角度）
                 bike_turn_scale_deg: float = 0.06,
                 # 速度自适应（动态零点）参数
                 bike_speed_scale_deg: float = 0.002,       # 每次修正量 = current_speed * 该系数
                 middle_angle_rectify_time_ms: int = 100,  # 每隔多少毫秒修正一次
                 middle_angle_recitfy_limit_deg: float = 3.0,  # 基准角的总限幅 ±deg
                 # 网络
                 socket_ip: str = '192.168.251.55',
                 socket_ports: list[int] | None = None,
                 march_velocity: float = 5.0,
                 # 相机安全检测与可视化
                 person_safety_distance: float = 1.5,
                 obstacle_safety_distance: float = 1.0,
                 visualize: bool = False,
                 mindir_path: str = "./yolov8x.mindir",
                 detection_interval: int = 1,
                 ):
        self.port = serial_port
        self.baud = baud
        self.axis_drive_index = axis_drive_index
        self.drive_accel = drive_accel
        self.axis_flywheel_index = axis_flywheel_index
        self.parser = ImuParser()

        self.roll_diff_limit = roll_diff_limit
        self._first_imu_received = 0

        # --- PID 参数 ---
        # 角度环
        self.angle_kp = angle_kp
        self.angle_ki = angle_ki
        self.angle_kd = angle_kd
        self.angle_integral = 0.0

        # 角速度环
        self.angle_v_kp = angle_v_kp
        self.angle_v_ki = angle_v_ki
        self.angle_v_kd = angle_v_kd
        self.angle_v_integral = 0.0
        self.angle_v_last_error = 0.0

        # 飞轮零速校正环
        self.flywheel_speed_kp = flywheel_speed_kp
        self.flywheel_speed_ki = flywheel_speed_ki
        self.flywheel_speed_integral_limit = flywheel_speed_integral_limit
        self.flywheel_speed_integral = 0.0

        self.flywheel_speed_limit = float(flywheel_speed_limit)
        self.flywheel_speed_limit = flywheel_speed_limit

        # 动态零点（机器中值角）
        self.machine_middle_angle = float(machine_middle_angle_init)
        self.middle_angle_recitfy_limit_deg = float(middle_angle_recitfy_limit_deg)
        self.bike_speed_scale_deg = float(bike_speed_scale_deg)
        self._last_rectify_ts = 0.0

        # 舵机与转向-倾角偏置
        self.bike_turn_scale_deg = float(bike_turn_scale_deg)
        self.servo_center_deg = float(servo_center_deg)
        self.servo_middle_range = float(servo_middle_range)
        self.servo_range_each_side_deg = float(servo_range_each_side_deg)
        self.servo_min_deg = self.servo_center_deg - self.servo_range_each_side_deg
        self.servo_max_deg = self.servo_center_deg + self.servo_range_each_side_deg
        self.steer_current_angle = self.servo_center_deg
        self.servo = Servo(servo_id=1)

        self.servo_speed_dps = servo_speed_dps  # 舵机转动速度，单位度/秒
        self.servo_step_deg = servo_step_deg
        self.servo_step_time = servo_step_time  # 平滑转动每步间隔时间，单位毫秒

        # 设定舵机回中
        self.steer_set(self.servo_center_deg)

        # 运行时
        self.stop_event = threading.Event()
        self._imu_thread: Optional[threading.Thread] = None
        self._ctrl_thread: Optional[threading.Thread] = None

        # ODrive
        self.odrive = None
        self.axis_flywheel = None
        self.axis_drive = None
        self.march_velocity = march_velocity
        self.simulate = False

        # IMU缓存
        self.last_eul = (0.0, 0.0, 0.0)
        self.last_acc = (0.0, 0.0, 0.0)
        self.last_gyr = (0.0, 0.0, 0.0)
        self.last_update = 0.0

        # 可选 Socket（示例保留接口，不在本文件发送）
        self.socket_ip = socket_ip
        self.socket_ports = socket_ports if socket_ports is not None else []
        self.flywheel_accel_limit = flywheel_accel_limit
        
        # --- CameraAI 集成 ---
        self.aiCamera = CameraAI(visualize=visualize,mindir_path=mindir_path,detection_interval=detection_interval)
        self.person_safety_distance = person_safety_distance
        self.obstacle_safety_distance = obstacle_safety_distance
        self.movement_state = "Stopped"
        self.safety_status_message = "System Ready"
        self.visualize = visualize
        self.detection_interval = detection_interval
        
        self.drive_input_vel = 0.0  # 当前驱动轮目标速度

    # ---------------------------- ODrive ----------------------------
    def connect_odrive(self, timeout: float = 10.0):
        try:
            import odrive
            from odrive.utils import dump_errors
            from odrive.enums import AxisState, ControlMode
        except Exception:
            print("未检测到 odrive 库或无法导入, 进入模拟模式（不会向电机发送命令）")
            self.simulate = True
            return

        print("查找 ODrive...")
        try:
            od = odrive.find_any(timeout=timeout)
            self.odrive = od
            self.axis_flywheel = getattr(od, f"axis{self.axis_flywheel_index}")
            self.axis_drive = getattr(od, f"axis{self.axis_drive_index}")
            print(f"找到 ODrive, 固件: {od.fw_version_major}.{od.fw_version_minor}.{od.fw_version_revision}")
            try:
                dump_errors(od, clear=True)
            except Exception:
                pass
            # 如果未校准, 尝试校准（会等待一段时间）
            if not self.axis_flywheel.motor.is_calibrated or not self.axis_flywheel.encoder.is_ready:
                print("飞轮电机校准中...")
                from odrive.enums import AxisState
                self.axis_flywheel.requested_state = AxisState.FULL_CALIBRATION_SEQUENCE
                # 等待校准完成或超时
                t0 = time.time()
                while time.time() - t0 < 18.0:
                    if self.axis_flywheel.error != 0:
                        print(f"轴校准失败, 错误: {self.axis_flywheel.error}")
                        break
                    if getattr(self.axis_flywheel, 'current_state', None) == AxisState.IDLE and self.axis_flywheel.encoder.is_ready:
                        break
                    time.sleep(0.5)
            
            if not self.axis_drive.motor.is_calibrated or not self.axis_drive.encoder.is_ready:
                print("后轮电机校准中...")
                from odrive.enums import AxisState
                self.axis_drive.requested_state = AxisState.FULL_CALIBRATION_SEQUENCE
                # 等待校准完成或超时
                t0 = time.time()
                while time.time() - t0 < 18.0:
                    if self.axis_drive.error != 0:
                        print(f"轴校准失败, 错误: {self.axis_drive.error}")
                        break
                    if getattr(self.axis_drive, 'current_state', None) == AxisState.IDLE and self.axis_drive.encoder.is_ready:
                        break
                    time.sleep(0.5)

            from odrive.enums import AxisState, ControlMode, InputMode
            # 尝试进入闭环速度控制
            try:
                self.axis_flywheel.requested_state = AxisState.CLOSED_LOOP_CONTROL
                self.axis_drive.requested_state = AxisState.CLOSED_LOOP_CONTROL
                time.sleep(0.5)
                self.axis_flywheel.controller.config.control_mode = ControlMode.VELOCITY_CONTROL
                self.axis_drive.controller.config.control_mode = ControlMode.VELOCITY_CONTROL
                self.axis_drive.controller.config.vel_ramp_rate = self.drive_accel
                self.axis_drive.controller.config.input_mode = InputMode.VEL_RAMP
                time.sleep(0.5)
            except Exception:
                pass

            print("ODrive 连接并尝试进入闭环控制（若硬件可用）。")
        except Exception as e:
            print(f"无法连接 ODrive: {e}\n进入模拟模式")
            self.simulate = True

    # ---------------------------- 舵机控制 ----------------------------
    def steer_set(self, angle_deg: float):
        angle_deg = max(self.servo_min_deg, min(self.servo_max_deg, angle_deg))
        # self.servo.set_angle_with_speed_limit(angle_deg, speed_dps=self.servo_speed_dps)
        self.servo.set_angle(angle_deg, 1000)

    def steer_center(self):
        self.servo.set_angle(self.servo_center_deg, 1000)
        self.steer_current_angle = self.servo_center_deg

    def steer_left(self):
        target = self.steer_current_angle + self.servo_step_deg
        #self.servo.set_angle(target, 10000)
        self.servo.set_angle_with_speed(target, speed_dps=self.servo_speed_dps, step_interval_ms=self.servo_step_time)
        self.steer_current_angle = target
        # self.servo.set_angle(target)

    def steer_right(self):
        target = self.steer_current_angle - self.servo_step_deg
        #self.servo.set_angle(target, 10000)
        self.servo.set_angle_with_speed(target, speed_dps=self.servo_speed_dps, step_interval_ms=self.servo_step_time)
        self.steer_current_angle = target
        # self.servo.set_angle(target)

    # ---------------------------- 线程控制 ----------------------------
    def start(self):
        self._imu_thread = threading.Thread(target=self._imu_read_loop, daemon=True)
        self._ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._imu_thread.start()
        self._ctrl_thread.start()
        # 启动相机AI
        if not self.aiCamera.start():
            print("\nFailed to start CameraAI service. Please check camera connection and logs.")
        else:
            print("\nCameraAI service is running. Starting main application loop...")
            print(f"Safety protocol: STOP if a person is within {self.person_safety_distance} m or any obstacle within {self.obstacle_safety_distance} m.")
        print("Press Ctrl+C to exit.")

    def stop(self):
        self.stop_event.set()
        #停止相机AI
        try:
            self.aiCamera.stop()
        except Exception:
            pass
        try:
            if not self.simulate and self.axis_flywheel is not None:
                self.axis_flywheel.controller.input_vel = 0.0
                self.axis_drive.controller.input_vel = 0.0
                from odrive.enums import AxisState
                self.axis_flywheel.requested_state = AxisState.IDLE
                self.axis_drive.requested_state = AxisState.IDLE
        except Exception:
            pass

    # ---------------------------- IMU读取 ----------------------------
    def _imu_read_loop(self):
        import serial
        try:
            ser = serial.Serial(self.port, baudrate=self.baud, timeout=0.5)
        except Exception as e:
            print(f"打开串口失败 {self.port}@{self.baud}: {e}")
            self.stop_event.set()
            return

        print(f"串口已打开: {self.port} @ {self.baud}, 开始解码 IMU 数据")
        try:
            while not self.stop_event.is_set():
                data = ser.read(ser.in_waiting or 1)
                for b in data:
                    self.parser.feed(b)
                eul = tuple(self.parser.eul)
                if self._first_imu_received < 1:
                    self.last_eul = eul
                    self._first_imu_received += 1
                elif abs(eul[1] - self.last_eul[1]) > self.roll_diff_limit:
                    self._first_imu_received = 0
                    continue
                else:
                    self.last_eul = eul
                self.last_gyr = tuple(self.parser.gyr)
                self.last_update = time.time()
                time.sleep(0.002)
        except Exception as e:
            print(f"IMU 读取线程异常: {e}")
        finally:
            try:
                ser.close()
            except Exception:
                pass
            self.stop_event.set()

    # ---------------------------- 控制环 ----------------------------
    def _angle_pid(self, current_angle: float, target_angle: float, current_gyro: float) -> float:
        error = current_angle - target_angle
        self.angle_integral += error
        self.angle_integral = max(-30.0, min(30.0, self.angle_integral))
        return self.angle_kp * error + self.angle_ki * self.angle_integral + self.angle_kd * current_gyro

    def _angular_velocity_pid(self, current_gyro: float, target_gyro: float) -> float:
        error = current_gyro - target_gyro
        self.angle_v_integral += error
        self.angle_v_integral = max(-10000.0, min(10000.0, self.angle_v_integral))
        derivative = error - self.angle_v_last_error
        output = self.angle_v_kp * error + self.angle_v_ki * self.angle_v_integral + self.angle_v_kd * derivative
        self.angle_v_last_error = error
        return output

    def _flywheel_zero_speed_pid(self, current_speed: float) -> float:
        target_speed = 0.0
        error = current_speed - target_speed
        self.flywheel_speed_integral += error
        #self.flywheel_speed_integral = max(-300.0, min(300.0, self.flywheel_speed_integral))
        self.flywheel_speed_integral = clamp(self.flywheel_speed_integral, self.flywheel_speed_integral_limit[0], self.flywheel_speed_integral_limit[1])
        # print("self.flywheel_speed_integral:", self.flywheel_speed_integral)
        return error * (self.flywheel_speed_kp / 10.0) + self.flywheel_speed_integral * (self.flywheel_speed_ki / 1000.0)

    def _read_drive_speed(self) -> float:
        # 若有驱动轮编码器速度，可用于动态零点速度漂移；单位不严格，按比例调 bike_speed_scale_deg。
        if not self.simulate and self.axis_drive is not None:
            try:
                return float(self.axis_drive.encoder.vel_estimate)
            except Exception:
                return 0.0
        return 0.0

    def _control_loop(self):
        loop_counter = 0
        pwm_accel = 0.0   # 飞轮速度校正量（慢环）
        pwm_x = 0.0       # 角速度目标（中环）
        loop_interval = 0.002  # 2ms，约 500Hz
        turn_bias = 0.0  # 转向-倾角偏置
        speed_bias = 0.0 # 速度-倾角偏置
        last_vel = 0.0
        steer_angle = 0.0
        while not self.stop_event.is_set():
            t_start = time.time()
            loop_counter = (loop_counter + 1) % 30000

            # IMU 超时保护
            if time.time() - self.last_update > 1.0:
                if not self.simulate and self.axis_flywheel is not None:
                    try:
                        self.axis_flywheel.controller.input_vel = 0.0
                    except Exception:
                        pass
                time.sleep(0.05)
                continue

            roll_angle = float(self.last_eul[1])  # deg
            roll_gyro = float(self.last_gyr[0])   # deg/s（或近似值）

               
            #steer_angle = float(self.steer_current_angle)  # 当前舵机角度 
            # if loop_counter % 40 == 0:
            #     if not (self.servo._move_thread and self.servo._move_thread.is_alive()):
            #         self.servo.set_angle(steer_angle+2,update_current_angle=False)
            # if loop_counter % 40 == 20:
            #     if not (self.servo._move_thread and self.servo._move_thread.is_alive()):
            #         self.servo.set_angle(steer_angle-3,update_current_angle=False)
            
            if loop_counter % 80 == 0:
                # 速度自适应：漂移机器中值角 + 限幅
                steer_angle = float(self.servo.current_degree)  # 当前舵机角度  
                current_speed = abs(self._read_drive_speed())
                speed_bias_direction = 0.0
                if abs(steer_angle - self.servo_center_deg) < self.servo_middle_range:
                    # 靠近中值时，速度偏置为零
                    speed_bias_direction = 0.0
                elif steer_angle > self.servo_center_deg:
                    speed_bias_direction = -1.0
                else:
                    speed_bias_direction = 1.0
                    
                speed_bias = current_speed * current_speed * self.bike_speed_scale_deg * speed_bias_direction
                turn_bias = (steer_angle - self.servo_center_deg) * self.bike_turn_scale_deg

            # 1) 飞轮零速校正（慢环，约 160ms）
            if loop_counter % 80 == 0:
                current_flywheel_speed = 0.0
                if not self.simulate and self.axis_flywheel is not None:
                    try:
                        current_flywheel_speed = self.axis_flywheel.encoder.vel_estimate
                    except Exception:
                        pass
                pwm_accel = self._flywheel_zero_speed_pid(current_flywheel_speed)

            # 2) 角度环（中环，约 30ms）
            if loop_counter % 15 == 0:
                # 动态零点：机器中值角 + 转向偏置 + 慢环校正
                dynamic_zero = self.machine_middle_angle + turn_bias + speed_bias + pwm_accel
                dynamic_zero = clamp(dynamic_zero,
                                     self.machine_middle_angle - self.middle_angle_recitfy_limit_deg,
                                     self.machine_middle_angle + self.middle_angle_recitfy_limit_deg)

                print(f"steer={steer_angle:>3.1f}, roll={roll_angle:>6.2f}°, dy_zero={dynamic_zero:>4.2f}°, spd_bias={speed_bias:>6.4f}, tgt_vel={final_flywheel_speed:>6.2f}, fw_s={self.flywheel_speed_integral:>5.1f}", end='\r', flush=True)
                # print(f"steer_angle={steer_angle:>3.1f}, roll={roll_angle:>6.2f}°, dynamic_zero={dynamic_zero:>6.2f}°, speed_bias={speed_bias:>6.4f}, target_vel={final_flywheel_speed:>6.2f}, flywheel_si={self.flywheel_speed_integral:>6.2f}", end='\r')

                pwm_x = self._angle_pid(roll_angle, dynamic_zero, roll_gyro)

            # 3) 角速度环（快环，每周期）
            final_flywheel_speed = self._angular_velocity_pid(roll_gyro, pwm_x)
            # final_flywheel_speed = max(-self.flywheel_speed_limit, min(self.flywheel_speed_limit, final_flywheel_speed))
            final_flywheel_speed = clamp(final_flywheel_speed, -self.flywheel_speed_limit, self.flywheel_speed_limit)
            final_flywheel_speed = clamp(final_flywheel_speed, last_vel - self.flywheel_accel_limit, last_vel + self.flywheel_accel_limit)  # 限制加减速度
            last_vel = final_flywheel_speed

            # 安全：偏差过大，清积分并停转
            if abs(roll_angle - self.machine_middle_angle) > 3.0:
                final_flywheel_speed = 0.0
                self.angle_integral = 0.0
                self.angle_v_integral = 0.0
                self.flywheel_speed_integral = 0.0
                self.set_stop()
            # final_flywheel_speed = 0.0
            # 写入飞轮
            if self.simulate or self.axis_flywheel is None:
                pass
            else:
                try:
                    self.axis_flywheel.controller.input_vel = float(final_flywheel_speed)
                except Exception as e:
                    print(f"写入速度失败: {e}")
            
            if self.socket_ports:
                try:
                    send_speed = float(self.axis_flywheel.encoder.vel_estimate) if self.axis_flywheel else final_flywheel_speed
                except Exception as e:
                    print(f"读取飞轮速度失败: {e}")
                try:
                    if len(self.socket_ports) > 0:
                        Socket_m.send_param(self.socket_ip, self.socket_ports[0], roll_angle)
                    if len(self.socket_ports) > 1:
                        Socket_m.send_param(self.socket_ip, self.socket_ports[1], send_speed)
                except Exception as e:
                    print(f"\nSocket 发送失败: {e}")
            
            if loop_counter % (self.detection_interval / loop_interval) == 0:
                try:
                    is_path_safe, reason = self.aiCamera.is_safe(
                        person_safe_dist=self.person_safety_distance,
                        obstacle_safe_dist=self.obstacle_safety_distance
                    )
                except Exception as e:
                    is_path_safe, reason = False, f"CameraAI error: {e}"
                    
                self.safety_status_message = reason
                if not is_path_safe and self.drive_input_vel > 0.0:
                    self.set_stop()
                    self.movement_state = "Stopped"
                    print(f"\n[Safety Check] {reason}")
        
            # 频率保持
            t_elapsed = time.time() - t_start
            time.sleep(max(0.0, loop_interval - t_elapsed))

    # ---------------------------- 行进控制 ----------------------------
    def set_forward(self):
        if self.axis_drive: 
            try:
                self.axis_drive.controller.input_vel = self.march_velocity
                self.drive_input_vel = self.march_velocity
            except Exception:
                pass
    
    def set_backward(self):
        if self.axis_drive: 
            try:
                self.axis_drive.controller.input_vel = -self.march_velocity
                self.drive_input_vel = -self.march_velocity
            except Exception:
                pass

    def set_stop(self):
        if self.axis_drive: 
            try:
                # current_speed = self._read_drive_speed()
                # self.axis_drive.controller.input_vel = current_speed/2
                # time.sleep(1.0)
                self.axis_drive.controller.input_vel = 0.0
                self.drive_input_vel = 0.0
            except Exception:
                pass


# ---------------------------- Flask Web ----------------------------
app = Flask(__name__)
ctrl: BalanceCtrlDyn | None = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def move():
    if not ctrl:
        return jsonify(status="error", message="控制器未初始化"), 500
    action = request.json.get('action')
    if action == 'forward':
        #相机安全检测：不安全则拦截并停下
        try:
            is_path_safe, reason = ctrl.aiCamera.is_safe(
                person_safe_dist=ctrl.person_safety_distance,
                obstacle_safe_dist=ctrl.obstacle_safety_distance
            )
        except Exception as e:
            is_path_safe, reason = False, f"CameraAI error: {e}"
            
        ctrl.safety_status_message = reason

        if is_path_safe:
        
            ctrl.set_forward()
            ctrl.movement_state = "Forward"
            ctrl.safety_status_message = "Moving Forward"
            return jsonify(status="ok", message="Moving forward")
        else:
            ctrl.set_forward()
            return jsonify(status="blocked", message=reason)
    elif action == 'backward':
        ctrl.set_backward()
    elif action == 'stop':
        ctrl.set_stop()
    elif action == 'left':
        ctrl.steer_left()
    elif action == 'right':
        ctrl.steer_right()
    return jsonify(status="ok")

def _visualization_loop(controller: BalanceCtrlDyn):
    """可选：实时可视化 CameraAI 检测结果"""
    try:
        import cv2
        import numpy as np
    except Exception:
        print("可视化需要安装 OpenCV 和 NumPy。已跳过。")
        return
    print("可视化线程已启动。在窗口激活时按 'q' 键退出。")
    while not controller.stop_event.is_set():
        frame, heatmap, detections = controller.aiCamera.get_latest_visuals_and_detections()
        if frame is None:
            time.sleep(0.2)
            continue
        display_frame = frame.copy()
        # 绘制检测框与距离
        if detections:
            for person in detections:
                box = person.get('box')
                dist = person.get('distance_m', 0.0)
                if not box:
                    continue
                x, y, w, h = box
                is_close = 0.01 < dist < controller.person_safety_distance
                color = (0, 0, 255) if is_close else (0, 255, 0)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                label = f"{dist:.2f}m"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(display_frame, (x, y - 20), (x + tw, y), color, -1)
                cv2.putText(display_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # 显示安全状态
        status_text = controller.safety_status_message
        color = (0, 0, 255) if str(status_text).startswith('STOP') or str(status_text).startswith('WARNING') else (0, 255, 0)
        (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(display_frame, (5, 5), (15 + tw, 25 + th), (0, 0, 0), -1)
        cv2.putText(display_frame, status_text, (10, 15 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # 显示
        if heatmap is None:
            cv2.imshow('Real-time Detection', display_frame)
        else:
            import numpy as np
            combined_display = np.hstack((display_frame, heatmap))
            cv2.imshow('Real-time Detection', combined_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("在可视化窗口中按下 'q'，正在停止...")
            controller.stop()
            break
        time.sleep(0.03)

def main():
    global ctrl

    default_cfg = {
        "mindir_path": "./yolov8x.mindir",
        "port": "/dev/ttyUSB0",
        "baud": 115200,
        "axis_drive_index": 0,
        "axis_flywheel_index": 1,
        # PID
        "angle_kp": 350.0,
        "angle_ki": 0.0,
        "angle_kd": 0.7,
        "angle_v_kp": 0.5,
        "angle_v_ki": 0.0,
        "angle_v_kd": 0.1,
        "flywheel_speed_kp": 0.1,
        "flywheel_speed_ki": 0.45,
        "flywheel_speed_integral_limit": [-200.0, 200.0],
        "socket_ip": "192.168.220.153",
        "socket_ports": [12344, 12346],
        # 动态零点
        "machine_middle_angle_init": -2.11,
        "bike_turn_scale_deg": 0.06,
        "bike_speed_scale_deg": 0.002,
        "middle_angle_recitfy_limit_deg": 3.0,
        # 舵机
        "servo_center_deg": -10.0,
        "servo_range_each_side_deg": 20.0,
        "servo_speed_dps": 30.0,  # 舵机转动速度，单位度/秒
        "servo_step_time": 50,  # 平滑转动每步间隔时间，单位毫秒
        # 其他
        "flywheel_speed_limit": 20.0,
        "flywheel_accel_limit": 5.0,
        "roll_diff_limit": 0.1,
        "march_velocity": 1.0,
        "drive_accel": 0.5,
        # 相机
        "person_safety_distance": 1.5,
        "obstacle_safety_distance": 1.0,
        "visualize": False,
    }

    cfg_path = os.path.join(os.path.dirname(__file__), "config_dynzero.yaml")
    if not os.path.exists(cfg_path):
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(default_cfg, f, allow_unicode=True, sort_keys=False)
        cfg = default_cfg
        print(f"已创建默认配置: {cfg_path}")
    else:
        with open(cfg_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg = {**default_cfg, **loaded}
        # 保存合并后的配置
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    ctrl = BalanceCtrlDyn(
        serial_port=cfg["port"], baud=cfg["baud"],
        axis_drive_index=cfg["axis_drive_index"], axis_flywheel_index=cfg["axis_flywheel_index"],
        drive_accel=cfg["drive_accel"],
        angle_kp=cfg["angle_kp"], angle_ki=cfg["angle_ki"], angle_kd=cfg["angle_kd"],
        angle_v_kp=cfg["angle_v_kp"], angle_v_ki=cfg["angle_v_ki"], angle_v_kd=cfg["angle_v_kd"],
        flywheel_speed_kp=cfg["flywheel_speed_kp"], flywheel_speed_ki=cfg["flywheel_speed_ki"],
        flywheel_speed_integral_limit=cfg["flywheel_speed_integral_limit"],
        machine_middle_angle_init=cfg["machine_middle_angle_init"],
        bike_turn_scale_deg=cfg["bike_turn_scale_deg"],
        bike_speed_scale_deg=cfg["bike_speed_scale_deg"],
        middle_angle_recitfy_limit_deg=cfg["middle_angle_recitfy_limit_deg"],
        servo_center_deg=cfg["servo_center_deg"],
        servo_range_each_side_deg=cfg["servo_range_each_side_deg"],
        servo_speed_dps=cfg["servo_speed_dps"],
        servo_step_deg=cfg["servo_step_deg"],
        servo_step_time=cfg["servo_step_time"],
        flywheel_speed_limit=cfg["flywheel_speed_limit"],
        flywheel_accel_limit=cfg["flywheel_accel_limit"],
        roll_diff_limit=cfg["roll_diff_limit"],
        march_velocity=cfg["march_velocity"],
        socket_ip=cfg["socket_ip"], socket_ports=cfg["socket_ports"],
        person_safety_distance=cfg["person_safety_distance"],
        obstacle_safety_distance=cfg["obstacle_safety_distance"],
        visualize=cfg["visualize"]
    )
    ctrl.connect_odrive()

    try:
        print("按回车开始控制（Ctrl+C 急停）...")
        input()
        ctrl.start()

        print("Web 服务器正在 http://0.0.0.0:5000 上运行")
        flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000), daemon=True)
        flask_thread.start()
        
        # 可选：启动可视化线程
        if cfg.get("visualize", False):
            vis_thread = threading.Thread(target=_visualization_loop, args=(ctrl,), daemon=True)
            vis_thread.start()

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("检测到 Ctrl+C, 停止...")
    finally:
        ctrl.stop()
        # 关闭可视化窗口（若开启）
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    main()