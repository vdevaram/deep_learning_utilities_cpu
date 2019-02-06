#Need to use dark flow to convert darknet mdoels to TF
git clone https://github.com/thtrieu/darkflow.git
cd darkflow
pip install Cython
pip install -e . --user
#get yolo weights, cfg, labels
wget https://pjreddie.com/media/files/yolov2.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
#convert darknet model to TF model
python3 ./flow --model yolov2.cfg --load yolov2.weights --savepb --labels coco.names
#match the extension file parameters yolo_v1_v2.json to yolov2.cfg params under [region] header
python3 $DLDT_PATH/model_optimizer/mo.py --framework tf --input_model built_graph/yolov2.pb --output_dir  ./   --batch 1  --tensorflow_use_custom_operations_config yolo_v1_v2.json
#run sample code 
??
