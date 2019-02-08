# download the tf yolov3 repo
git clone https://github.com/mystic123/tensorflow-yolo-v3.git
cd tensorflow-yolo-v3
#download weights from darknet
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
wget https://github.com/pjreddie/darknet/blob/master/data/coco.names
#convert weights to TF model
python3 ./convert_weights_pb.py --class_names coco.names   --weights_file yolov3.weights --data_format NHWC
python3 ./convert_weights_pb.py --class_names coco.names   --weights_file yolov3-tiny.weights --data_format NHWC --tiny
#convert model to openVINO
python3 $DLDT_PATH/model_optimizer/mo.py --framework tf --input_model frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config /home/vinod/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3.json --batch 1
 python3 $DLDT_PATH/model_optimizer/mo.py --framework tf --input_model frozen_darknet_yolov3_tiny_model.pb --tensorflow_use_custom_operations_config yolo_v3_tiny.json --batch 1

#yolo_v3_tiny.json contents: 
: '
[
  {
    "id": "TFYOLOV3",
    "match_kind": "general",
    "custom_attributes": {
      "classes": 80,
      "anchors": [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319],
      "coords": 4,
      "num": 6,
      "mask": [0, 1, 2],
      "entry_points": ["detector/yolo-v3-tiny/Reshape", "detector/yolo-v3-tiny/Reshape_4"]
    }
  }
]
'
: '
# yolo_v3.json contents: 

[
  {
    "id": "TFYOLOV3",
    "match_kind": "general",
    "custom_attributes": {
      "classes": 80,
      "coords": 4,
      "num": 9,
      "mask": [0, 1, 2],
      "entry_points": ["detector/yolo-v3/Reshape", "detector/yolo-v3/Reshape_4", "detector/yolo-v3/Reshape_8"]
    }
  }
]
'
