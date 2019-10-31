# This file gives step by step instructions to download install openVINO and use for running object detection models on IA
#Assumed to have ubuntu or CentOS installed along with Python3.
#################### installation and setup #######################
#download and install openVINO R3.1
cd /tmp
wget http://registrationcenter-download.intel.com/akdlm/irc_nas/16057/l_openvino_toolkit_p_2019.3.376.tgz
tar -zxvf l_openvino_toolkit_p_2019.3.376.tgz
cd l_openvino_toolkit_p_2019.3.376/
sudo ./install_openvino_dependencies.sh
./install.sh
# if the openVINO installed in root mode then installation path will be /opt. otherwise it will be in $HOME folder. Choose below command accordingly
export VINO_HOME=$HOME  or export VINO_HOME=/opt
# Add openVINO libs to PATH and LD_LIBRARY_PATH. Add below line to .bashrc file if you want to enable by default
source $VINO_HOME/intel/openvino/bin/setupvars.sh
# create virtual environment and enable for python3 as below for ubuntu. instructions are different for centOS.
virtualenv -p python3 ~/.vino_env
# Add below line to .bashrc file if you want to enable by default
source ~/.vino_env/bin/activate
# Add dependency framework libs
cd $VINO_HOME/intel/openvino/deployment_tools/model_optimizer/
pip install -r requirements_tf.txt --upgrade
mkdir $HOME/VINO
export VINO_WORKSPACE=$HOME/VINO
#Build samples and demos
mkdir -p $VINO_WORKSPACE/samples
rm -rf $VINO_WORKSPACE/samples/*
cd $VINO_WORKSPACE/samples
cmake $VINO_HOME/intel/openvino/deployment_tools/inference_engine/samples/
make -j
mkdir -p $VINO_WORKSPACE/demos
rm -rf $VINO_WORKSPACE/demos/*
cd $VINO_WORKSPACE/demos
cmake $VINO_HOME/intel/openvino/deployment_tools/inference_engine/demos/
#################### installation and setup complete #######################
#### initialize environment #####
export VINO_HOME=$HOME  or export VINO_HOME=/opt
export VINO_WORKSPACE=$HOME/VINO
source $VINO_HOME/intel/openvino/bin/setupvars.sh
source ~/.vino_env/bin/activate
#FRCNN for Object detection
mkdir -p $VINO_WORKSPACE/models
cd $VINO_WORKSPACE/models
tar -zxvf faster_rcnn_resnet50_coco_2018_01_28.tar.gz
cd faster_rcnn_resnet50_coco_2018_01_28
#FP32 conversion 
python3 $VINO_HOME/intel/openvino/deployment_tools/model_optimizer/mo.py --framework tf --input_model frozen_inference_graph.pb     --output_dir  ./fp32/ --output=detection_boxes,detection_scores,num_detections --tensorflow_use_custom_operations_config $VINO_HOME/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config
#FP16 Conversion
python3 $VINO_HOME/intel/openvino/deployment_tools/model_optimizer/mo.py --framework tf --input_model frozen_inference_graph.pb     --output_dir  ./fp16/ --output=detection_boxes,detection_scores,num_detections --tensorflow_use_custom_operations_config $VINO_HOME/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --data_type FP16
#INT8 conversion
## TBD
# Benchmark 
$VINO_WORKSPACE/samples/intel64/Release/benchmark_app -m $VINO_WORKSPACE/models/faster_rcnn_resnet50_coco_2018_01_28/fp32/frozen_inference_graph.xml
# Python Demo sample usage 
 $VINO_HOME/intel/openvino/deployment_tools/inference_engine/demos/python_demos/object_detection_demo_ssd_async/object_detection_demo_ssd_async.py -i "cam" -m $VINO_WORKSPACE/models/faster_rcnn_resnet50_coco_2018_01_28/fp32/frozen_inference_graph.xml -d CPU  -l $VINO_WORKSPACE/samples/intel64/Release/lib/libcpu_extension.so

