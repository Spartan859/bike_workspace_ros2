conda activate mindspore
source install/setup.zsh
ros2 launch camera_ai_ros2 camera_ai.launch.py mindir_path:=/root/Workspace/bike_workspace_ros2/yolov8x_lite.mindir enabled:=true