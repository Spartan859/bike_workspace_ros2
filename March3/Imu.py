"""Copy of original Imu parser (kept compatible)."""
import struct
import time
import binascii


class ImuParser:
    def __init__(self):
        self.state = 0
        self.payload_len = 0
        self.crc_calculated = 0
        self.crc_received = 0
        self.data_buffer = bytearray()

        self.acc = [0.0, 0.0, 0.0]
        self.gyr = [0.0, 0.0, 0.0]
        self.eul = [0.0, 0.0, 0.0]

    @staticmethod
    def _crc16_compute(data_bytes, init=0):
        crc = init & 0xFFFF
        for b in data_bytes:
            crc ^= (b << 8) & 0xFFFF
            for _ in range(8):
                temp = (crc << 1) & 0xFFFF
                if crc & 0x8000:
                    temp ^= 0x1021
                crc = temp
        return crc & 0xFFFF

    @staticmethod
    def _crc16_modbus(data_bytes, init=0xFFFF):
        crc = init & 0xFFFF
        for b in data_bytes:
            crc ^= b
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc & 0xFFFF

    def _parse_payload(self, payload: bytes):
        offset = 0
        while offset < len(payload):
            tag = payload[offset]
            if tag == 0xA0 and offset + 7 <= len(payload):
                ax, ay, az = struct.unpack_from('<hhh', payload, offset+1)
                self.acc = [ax / 1000.0, ay / 1000.0, az / 1000.0]
                offset += 7
            elif tag == 0xB0 and offset + 7 <= len(payload):
                gx, gy, gz = struct.unpack_from('<hhh', payload, offset+1)
                self.gyr = [gx / 10.0, gy / 10.0, gz / 10.0]
                offset += 7
            elif tag == 0xD0 and offset + 7 <= len(payload):
                t0, t1, t2 = struct.unpack_from('<hhh', payload, offset+1)
                roll = t0 / 100.0
                pitch = t1 / 100.0
                yaw = t2 / 10.0
                self.eul = [pitch, roll, yaw]
                offset += 7
            elif tag == 0x91 and offset + 76 <= len(payload):
                data = payload[offset:offset+76]
                try:
                    acc = struct.unpack_from('<fff', data, 12)
                    gyr = struct.unpack_from('<fff', data, 24)
                    eul = struct.unpack_from('<fff', data, 48)
                    self.acc = [acc[0], acc[1], acc[2]]
                    self.gyr = [gyr[0], gyr[1], gyr[2]]
                    self.eul = [eul[0], eul[1], eul[2]]
                except struct.error:
                    pass
                offset += 76
            else:
                offset += 1

    def feed(self, byte: int):
        if self.state == 0:
            if byte == 0x5A:
                self.header = bytearray([0x5A])
                self.state = 1
        elif self.state == 1:
            self.type = byte
            self.header.append(byte)
            if self.type == 0xA5:
                self.state = 2
            else:
                self.state = 0
        elif self.state == 2:
            self.payload_len = byte
            self.header.append(byte)
            self.state = 3
        elif self.state == 3:
            self.payload_len |= (byte << 8)
            self.header.append(byte)
            self.state = 4
        elif self.state == 4:
            self.crc_received = byte
            self.state = 5
        elif self.state == 5:
            self.crc_received |= (byte << 8)
            self.data_buffer = bytearray()
            if self.payload_len > 0:
                self.state = 6
            else:
                calc = self._crc16_compute(self.header)
                if calc == self.crc_received:
                    pass
                else:
                    print(f"CRC error: recv={self.crc_received:04X} calc={calc:04X}")
                self.state = 0
        elif self.state == 6:
            self.data_buffer.append(byte)
            if len(self.data_buffer) >= self.payload_len and self.type == 0xA5:
                frame = bytes(self.header) + struct.pack('<H', self.crc_received) + bytes(self.data_buffer)
                calc = self._crc16_compute(self.header + bytes(self.data_buffer), init=0)
                if calc == self.crc_received:
                    self._parse_payload(bytes(self.data_buffer))
                else:
                    calc_ff = self._crc16_compute(self.header + bytes(self.data_buffer), init=0xFFFF)
                    calc_modbus = self._crc16_modbus(self.header + bytes(self.data_buffer))
                    try:
                        print("CRC error:")
                        print(f"  recv CRC   = {self.crc_received:04X}")
                        print(f"  calc(init=0)= {calc:04X}")
                        print(f"  calc(init=0xFFFF)= {calc_ff:04X}")
                        print(f"  calc(modbus)= {calc_modbus:04X}")
                    except Exception:
                        pass
                self.state = 0
