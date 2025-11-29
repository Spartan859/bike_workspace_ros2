# Bike Workspace (ROS 2 Humble)

### 环境配置

#### ROS2安装
```shell
source <(wget -qO- http://fishros.com/install)
```
选择合适的选项进行一键安装，**要求安装ROS2 Humble分支**。

#### conda环境
创建名为 `mindspore` 的 Python 3.10 环境
```shell
conda create -n mindspore python=3.10
conda activate mindspore
# install mindspore 2.6.0
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.6.0/MindSpore/unified/aarch64/mindspore-2.6.0-cp310-cp310-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://repo.huaweicloud.com/repository/pypi/simple
# install mindyolo
pip install mindyolo
# install ros2 related packages
pip install rospkg
pip install -U colcon-common-extensions
```

#### 编译ROS2工作区
克隆本仓库及所有子模块：
```shell
git clone --recurse-submodules https://github.com/Spartan859/bike_workspace_ros2.git
cd bike_workspace_ros2
source /opt/ros/humble/setup.sh
python -m colcon build --symlink-install
```

#### 安装MindSpore Lite
MindSpore Lite官方页面请查阅：[MindSpore Lite](https://mindspore.cn/lite) 
- 下载tar.gz包并解压，同时配置环境变量LITE_HOME,LD_LIBRARY_PATH,PATH
```shell
tar -zxvf mindspore-lite-2.6.0-linux-aarch64.tar.gz 
export LITE_HOME=/[path_to_mindspore_lite_xxx]
export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH
export Convert=$LITE_HOME/tools/converter/converter/converter_lite
```
LITE_HOME为tar.gz解压出的文件夹路径，请设置绝对路径
- 安装whl包
```shell
pip install mindspore_lite-2.6.0-cp310-cp310-linux_aarch64.whl
```

