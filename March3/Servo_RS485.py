import serial
import time
import struct

# --- 协议常量 ---
HEADER = bytes([0x12, 0x4C])
CMD_PING = 0x01
CMD_READ_DATA = 0x02
CMD_WRITE_DATA = 0x03

# --- 舵机寄存器地址 (根据 Command.txt 推断) ---
ADDR_SET_POSITION_TIME = 0x2A  # 写入目标位置和时间的地址
ADDR_GET_POSITION = 0x38       # 读取当前位置的地址

# --- 映射常量 ---
SERVO_MIN_POSITION = 0
SERVO_MAX_POSITION = 4096
SERVO_MIN_DEGREE = -135.0
SERVO_MAX_DEGREE = 135.0

def _calculate_checksum(data):
    """
    计算校验和。
    校验和的计算方法是：将所有数据字节相加，然后按位取反，最后与 0xFF 进行与操作。
    :param data: 一个包含字节值的整数列表或bytes对象。
    :return: 计算出的校验和（一个0-255的整数）。
    """
    return (~sum(data)) & 0xFF

class Servo:
    """
    用于控制和通信的舵机类。
    """
    def __init__(self, port='/dev/ttyUSB1', servo_id=1, baudrate=115200, timeout=0.5):
        """
        初始化舵机对象。
        :param port: 串口名称 (例如 'COM8')。
        :param servo_id: 舵机ID (默认为 1)。
        :param baudrate: 波特率 (默认为 115200)。
        :param timeout: 串口读取超时时间。
        """
        self.port = port
        self.servo_id = servo_id
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.last_angle = 0

    def _send_command(self, command_bytes):
        """
        通过串口发送指令并返回经过清理的响应。
        """
        raw_response = None
        try:
            with serial.Serial(self.port, self.baudrate, timeout=self.timeout) as ser:
                ser.write(command_bytes)
                time.sleep(0.05)
                raw_response = ser.read_all()

                if raw_response:
                    # 健壮性处理：在响应中查找与本机ID匹配的帧头
                    # 舵机响应似乎不带 12 4C，直接以 ID 开头
                    try:
                        start_index = raw_response.index(self.servo_id)
                        # 截取从正确ID开始的有效帧
                        valid_response = raw_response[start_index:]
                        return valid_response
                    except ValueError:
                        return None
                else:
                    return None

        except (serial.SerialException, Exception):
            # 发生任何异常时，静默失败并返回 None
            return None

    def ping(self):
        """
        PING舵机以检查其状态。
        :return: 如果收到响应则返回 True，否则返回 False。
        """
        # 长度字段(LEN) = CMD(1) = 1, 但协议规定为 2
        length = 2
        data_part = bytes([length, CMD_PING])
        checksum_data = bytes([self.servo_id]) + data_part
        checksum = _calculate_checksum(checksum_data)
        command = HEADER + checksum_data + bytes([checksum])
        
        response = self._send_command(command)
        return response is not None

    def set_angle_in_time(self, angle, time_ms = 0):
        """
        在指定时间内将舵机转动到指定位置值。
        :param angle: 目标位置值 (0-4096)。
        :param time_ms: 完成转动所需的时间 (毫秒)。
        """
        # 数据部分: [目标位置(2字节)] + [运行时间(2字节)]
        # 使用大端序 (>)
        data_to_write = struct.pack('>HH', angle, min(time_ms<<2, 0xFFFF))

        # 长度字段(LEN) = CMD(1) + ADDR(1) + DATA(4) + CHK(1) = 7
        length = 3 + len(data_to_write)
        data_part = bytes([length, CMD_WRITE_DATA, ADDR_SET_POSITION_TIME]) + data_to_write
        
        checksum_data = bytes([self.servo_id]) + data_part
        checksum = _calculate_checksum(checksum_data)
        command = HEADER + checksum_data + bytes([checksum])
        
        self._send_command(command)

    def get_angle(self):
        """
        读取舵机的当前位置值。
        :return: 当前位置值 (0-4096)，如果失败则返回 None。
        """
        read_len = 2  # 角度是2个字节
        # 长度字段(LEN) = CMD(1) + ADDR(1) + READ_LEN(1) = 3
        # 但根据协议文档，读取指令的长度字段为4
        length = 4
        data_part = bytes([length, CMD_READ_DATA, ADDR_GET_POSITION, read_len])
        
        checksum_data = bytes([self.servo_id]) + data_part
        checksum = _calculate_checksum(checksum_data)
        command = HEADER + checksum_data + bytes([checksum])
        
        response = self._send_command(command)
        if response is not None:
            print(list(response))
        else:
            print(None)

        # 解析响应: ID LEN CMD [DATA...] CHK
        # 0:ID, 1:LEN, 2:CMD, 3...-2:DATA, -1:CHK
        if response and len(response) >= 6:
            # 数据从第3个字节开始，取2个字节
            data_bytes = response[3:5]
            if len(data_bytes) == 2:
                # 使用大端序 (>)
                value = struct.unpack('>H', data_bytes)[0]
                return value
        
        return None

    def _degree_to_position(self, degree):
        """将角度 (-90 to 90) 转换为位置值 (0-4096)"""
        degree = max(min(degree, SERVO_MAX_DEGREE), SERVO_MIN_DEGREE)
        pos_range = SERVO_MAX_POSITION - SERVO_MIN_POSITION
        deg_range = SERVO_MAX_DEGREE - SERVO_MIN_DEGREE
        position = SERVO_MIN_POSITION + ((degree - SERVO_MIN_DEGREE) / deg_range) * pos_range
        return int(position)

    def _position_to_degree(self, position):
        """将位置值 (0-4096) 转换为角度 (-90 to 90)"""
        position = max(min(position, SERVO_MAX_POSITION), SERVO_MIN_POSITION)
        pos_range = SERVO_MAX_POSITION - SERVO_MIN_POSITION
        deg_range = SERVO_MAX_DEGREE - SERVO_MIN_DEGREE
        degree = SERVO_MIN_DEGREE + ((position - SERVO_MIN_POSITION) / pos_range) * deg_range
        return degree

    def set_angle(self, degree, time_ms=0):
        """
        在指定时间内将舵机转动到指定角度。
        :param degree: 目标角度 (-90 to 90)。
        :param time_ms: 完成转动所需的时间 (毫秒)。
        """
        position = self._degree_to_position(degree)
        self.set_angle_in_time(position, time_ms)
    
    def set_angle_with_speed(self, target_degree, speed_dps=30.0, step_interval_ms=50):
       """
       以恒定速度（度/秒）平滑转动到目标角度。
       通过拆分为多个小步，循环调用 set_angle_in_time 实现。
       :param target_degree: 目标角度 (-135 to 135)。
       :param speed_dps: 转动速度（度/秒）。
       :param step_interval_ms: 每步的时间间隔（毫秒），越小越平滑但通信频繁。
       """
       # 1. 读取当前角度
       current_deg = self.current_degree
       if current_deg is None:
           print("无法读取当前角度，使用上次记录值")
           current_deg = self.last_angle
       
       # 2. 计算角度差与总时间
       angle_diff = target_degree - current_deg
       if abs(angle_diff) < 0.5:
           # 差异过小，直接到位
           self.set_angle(target_degree, time_ms=step_interval_ms)
           return
       
       total_time_s = abs(angle_diff) / speed_dps
       total_time_ms = int(total_time_s * 1000)
       
       # 3. 拆分成若干步
       num_steps = max(1, total_time_ms // step_interval_ms)
       step_deg = angle_diff / num_steps
       
       # 4. 循环执行每步
       for i in range(1, num_steps + 1):
           intermediate_deg = current_deg + step_deg * i
           self.set_angle(intermediate_deg, time_ms=step_interval_ms)
           time.sleep(step_interval_ms / 1000.0)  # 等待该步完成
       
       # 5. 最后确保精确到位
       self.set_angle(target_degree, time_ms=step_interval_ms)
       time.sleep(step_interval_ms / 1000.0)
       self.last_angle = target_degree

    @property
    def current_degree(self):
        """
        读取舵机的当前角度。
        :return: 当前角度 (-90 to 90)，如果失败则返回 None。
        """
        position = self.get_angle()
        # print(position)
        if position is not None:
            self.last_angle = self._position_to_degree(position)
        return self.last_angle


if __name__ == '__main__':
    # --- 这是一个使用示例 ---
    my_servo = Servo(servo_id=1)

    print("\n--- 测试: PING 舵机 ---")
    if my_servo.ping():
        print("舵机 PING 成功，设备在线。")
    else:
        print("舵机 PING 失败，请检查连接或ID。")
        exit()
    my_servo.set_angle(degree=8, time_ms=500)
    time.sleep(1.5)

    print("\n--- 测试: 读取当前角度 ---")
    print(f"舵机当前角度为: {my_servo.current_degree:.2f} 度")
    my_servo.set_angle_with_speed(target_degree=18, speed_dps=5.0, step_interval_ms=100)
    time.sleep(1.5)

    print("\n--- 测试: 再次读取当前角度 ---")
    print(f"舵机当前角度为: {my_servo.current_degree:.2f} 度")
    my_servo.set_angle(degree=-10, time_ms=1000)
    time.sleep(1.5)

    print("\n--- 测试: 再次读取当前角度 ---")
    print(f"舵机当前角度为: {my_servo.current_degree:.2f} 度")