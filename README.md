# Bike Workspace (ROS 2 Humble)

## 环境配置

### ROS2安装
```shell
source <(wget -qO- http://fishros.com/install)
```
选择合适的选项进行一键安装，**要求安装ROS2 Humble分支**。

### conda环境
创建名为 `mindspore` 的 Python 3.10 环境：

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

### 编译ROS2工作区

