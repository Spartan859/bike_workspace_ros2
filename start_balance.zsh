conda activate mindspore
python March3/Servo_RS485.py
source install/setup.zsh
ros2 launch balance_controller_ros2 bringup.launch.py servo_port:=/dev/ttyUSB0 imu_port:=/dev/ttyUSB1