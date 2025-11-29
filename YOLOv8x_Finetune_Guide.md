### YOLOv8x 全参微调（基于Mindspore）
自行车的避障和跟随功能需要能够在性能有限的OrangePi AIpro上高效运行的目标检测模型。为此，训练和推理需要使用不同的框架，即功能完备的MindSpore和具有轻量化AI推理加速能力的MindSpore-Lite。
本部分实验基于MindSpore和昇腾CANN，在coco2017数据集上对YOLOv8x模型进行全参微调训练，得到可供Mindspore-Lite加载的yolov8x_lite.mindir，部署到camera_ai_node节点进行高速推理。

#### 数据准备
下载coco2017 YOLO格式 [coco2017labels-segments](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip) 以及coco2017 原始图片 [train2017](http://images.cocodataset.org/zips/train2017.zip) , [val2017](http://images.cocodataset.org/zips/val2017.zip) ，然后将coco2017 原始图片放到coco2017 YOLO格式 images目录下：
```
└─ coco2017_yolo
    ├─ annotations
    │    └─ instances_val2017.json
    ├─ images
    │    ├─ train2017   # coco2017 原始图片
    │    └─ val2017     # coco2017 原始图片
    ├─ labels
    │    ├─ train2017
    │    └─ val2017
    ├─ train2017.txt
    ├─ val2017.txt
    └─ test-dev2017.txt
```

#### 编写配置文件

新建 `configs/yolov8/yolov8x_coco.yaml`
```yaml
__BASE__: [
    # 继承官方 yolov8x 结构（路径按实际仓库组织）
    './yolov8x.yaml',
]

# 训练图像尺寸（与预训练保持一致）
img_size: 640

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

data:
    dataset_name: coco2017_yolo
    train_set: /absolute/path/to/coco2017_yolo/train2017.txt
    val_set:   /absolute/path/to/coco2017_yolo/val2017.txt
    test_set:  /absolute/path/to/coco2017_yolo/val2017.txt
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
```

#### 训练
```bash
python train.py --config configs/yolov8/yolov8x_coco.yaml 
```

#### 权重转换与部署
先把ckpt转换为mindspore专用的mindir权重，便于NPU推理。
```shell
python ./deploy/export.py --config ./configs/yolov8/yolov8x_coco.yaml --weight /path_to_ckpt/WEIGHT.ckpt --file_format MINDIR --device_target Ascend
```
为了加快推理时加载模型的速度，把MindSpore mindir文件转换成MindSpore Lite mindir文件，直接使用lite mindir文件进行推理。
```shell
$Convert --fmk=MINDIR --modelFile=./yolov8x.mindir --outputFile=./yolov8x_lite  --saveType=MINDIR --optimize=ascend_oriented
```
把yolov8x_lite.mindir放到bike_workspace_ros2目录下以便camera_ai_ros2读取。
