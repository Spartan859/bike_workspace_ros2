# YOLOv8x 基于 COCO2017 的全参微调指南（MindYOLO）

本文在 finetune.md 的流程基础上，给出“基于 COCO2017 的 YOLOv8x 全参数微调”完整操作手册：从数据准备、配置与训练，到评估、导出与部署（对接本仓库 `camera_ai_ros2` 的 MINDIR 推理）。

---
## 1. 数据准备（COCO2017 YOLO 格式）

下载以下资源并组织为 YOLO 目录结构。可使用官方提供的 YOLO 标签包与原始图像：
- COCO2017 标签（YOLO格式）：coco2017labels-segments.zip
- 原始图片：train2017.zip、val2017.zip

解压并整理至如下结构（示例目录 `coco2017_yolo/`）：
```
coco2017_yolo/
    ├─ annotations/
    │   └─ instances_val2017.json         # 可选，仅评测或对齐使用
    ├─ images/
    │   ├─ train2017/                     # train 原图
    │   └─ val2017/                       # val 原图
    ├─ labels/
    │   ├─ train2017/                     # YOLO txt 标签（与 images 同名）
    │   └─ val2017/
    ├─ train2017.txt                      # 每行一个相对路径，如 images/train2017/xxxx.jpg
    ├─ val2017.txt                        # 每行一个相对路径，如 images/val2017/xxxx.jpg
    └─ test-dev2017.txt                   # 可选
```

注：若你使用 COCO JSON 原生标注，也可直接采用 COCO 格式训练；本文以 YOLO 文本标注为例，路径由 dataset yaml 指定。

---
## 2. 编写配置文件（模型 / 数据集 / 超参）

推荐在 MindYOLO 仓库内新建自定义配置，避免修改原始文件。示例：

1) 模型配置 `configs/yolov8/yolov8x_coco.yaml`
```
__BASE__: [
    # 继承官方 yolov8x 结构（路径按实际仓库组织）
    './yolov8x.yaml',
]

# 全参微调：不冻结任何层，nc=80（COCO80 类）
nc: 80
names: [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
    'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'
]

# 训练图像尺寸（与预训练保持一致）
img_size: 640
```

2) 数据配置 `configs/datasets/coco2017_yolo.yaml`
```
dataset_name: coco2017_yolo
train_set: /absolute/path/to/coco2017_yolo/train2017.txt
val_set:   /absolute/path/to/coco2017_yolo/val2017.txt
test_set:  /absolute/path/to/coco2017_yolo/val2017.txt
nc: 80

# 采用 YOLO 文本标签读取；若使用 COCO json，请换用相应 dataset 类与字段
```

3) 超参/训练配置（可合并到模型 yaml 或单独 hyp.yaml）
```
per_batch_size: 32            # 单卡 batch；总 batch = per_batch_size * device_num
epochs: 80
optim: SGD                    # 或 AdamW（小数据集可尝试 AdamW）
lr_init: 0.01                 # 预热后起始学习率（SGD 常用 1e-2）
lrf: 0.01                     # 余弦退火最终比率
weight_decay: 0.0005
warmup_epochs: 3
multi_scale: True             # 多尺度增强

# 数据增强（按需微调）
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
fliplr: 0.5
flipud: 0.0
degrees: 10
mosaic: True                  # 前 70% epoch 使用，后期可降低
```

> 预训练权重：建议使用官方发布的 `yolov8x` 预训练 ckpt（COCO 上预训练）。若继续在 COCO 上“再微调”，可视为继续训练，能对齐你自定义的数据增强/调度策略。

---
## 3. 训练（单机 / 多卡）

进入 MindYOLO 仓库根目录（包含 `tools/train.py`）：

Linux（单卡 GPU/Ascend）
```bash
python tools/train.py \
    --config configs/yolov8/yolov8x_coco.yaml \
    --data   configs/datasets/coco2017_yolo.yaml \
    --weights /path/to/yolov8x_pretrain.ckpt \
    --epochs 80 \
    --batch_size 32 \
    --img_size 640 \
    --optim SGD \
    --warmup_epochs 3 \
    --multi_scale \
    --device 0
```

Linux（多卡分布式，示例 8 卡）
```bash
mpirun -n 8 python tools/train.py \
    --config configs/yolov8/yolov8x_coco.yaml \
    --data   configs/datasets/coco2017_yolo.yaml \
    --weights /path/to/yolov8x_pretrain.ckpt \
    --epochs 100 --batch_size 64 --img_size 640 --multi_scale
```

Windows（可用 GPU 时）
```bat
python tools\train.py ^
    --config configs\yolov8\yolov8x_coco.yaml ^
    --data   configs\datasets\coco2017_yolo.yaml ^
    --weights D:\weights\yolov8x_pretrain.ckpt ^
    --epochs 80 ^
    --batch_size 16 ^
    --img_size 640 ^
    --optim SGD ^
    --multi_scale ^
    --device 0
```

提示与实践：
- 显存不足 → 降低 `batch_size`，或启用混合精度（AMP）/梯度累积（若脚本支持）。
- 继续训练（resume） → 使用 `--weights runs/xxx/last.ckpt --resume True`。
- 全参微调默认不冻结任何层；若需冻结，使用 `freeze` 列表（本指南不建议）。

---
## 4. 评估与指标

独立评测（不训练）：
```bash
python tools/val.py \
    --config configs/yolov8/yolov8x_coco.yaml \
    --data   configs/datasets/coco2017_yolo.yaml \
    --weights runs/exp_yolov8x/best.ckpt \
    --img_size 640
```

重点关注：mAP@0.5、mAP@[0.5:0.95]、各类 PR 曲线与推理时延（可辅助部署选型）。

---
## 5. 导出为 MINDIR 并验证

导出：
```bash
python tools/export.py \
    --config  configs/yolov8/yolov8x_coco.yaml \
    --data    configs/datasets/coco2017_yolo.yaml \
    --weights runs/exp_yolov8x/best.ckpt \
    --format  MINDIR \
    --img_size 640
```

产物通常位于 `runs/exp_yolov8x/weights/` 下，记下 `model.mindir` 路径。可用 `tools/infer.py`/自写脚本对比 ckpt 推理与 MINDIR 推理的一致性（允许微小数值差）。

---
## 6. 部署到本仓库（camera_ai_ros2）

将导出的 `*.mindir` 放入工作区并在 launch 传参：
```bash
ros2 launch camera_ai_ros2 camera_ai.launch.py mindir_path:=/abs/path/to/model.mindir enabled:=true
```
Windows：
```bat
ros2 launch camera_ai_ros2 camera_ai.launch.py mindir_path:=D:\path\to\model.mindir enabled:=true
```

节点会发布：
- `/safety/status`（`std_msgs/String`）
- `/camera/detections`（`vision_msgs/Detection2DArray`）

若仅关心 `person` 类别，可在 `camera_ai_ros2` 内按需过滤或做类别映射。

---
## 7. 常见问题（FAQ）

- 训练不收敛或震荡：
    - 降低 `lr_init`（如 0.01 → 0.005/0.001），或增大 `warmup_epochs`。
    - 检查标签是否与图像一一对应、类别数（nc）与 names 对齐。
- 显存不足：
    - 减小 `batch_size`；多卡分布式；开启 AMP；降低 `img_size`（512）。
- 精度不达标：
    - 增加训练轮数；加强/调优数据增强；检查长尾类别采样；确保评测与训练预处理一致。
- 导出失败/推理不一致：
    - 升级 MindSpore 版本；确认算子支持；对齐预处理/后处理逻辑。

---
## 8. 调参与推荐配置（COCO 场景）

起步推荐（单机 1 卡）：
```
img_size: 640
batch_size: 32
epochs: 80
optim: SGD
lr_init: 0.01
lrf: 0.01
warmup_epochs: 3
weight_decay: 0.0005
multi_scale: true
mosaic: true  # 前期开启，后 20% epoch 降低权重或关闭
```

速度优先：
- 若实测时延过高，考虑 YOLOv8m/s 或 AMP/量化（Lite int8 校准）。

---
以上流程覆盖了 COCO2017 上的 YOLOv8x 全参微调到导出部署的闭环。若需要，我可以提供一份可直接运行的最小化配置模板（含路径占位）与一键脚本（train/val/export）。
