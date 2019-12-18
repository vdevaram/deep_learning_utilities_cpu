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
make -j
#################### installation and setup complete #######################
#### Initialize environment ##### This step is needed each time new terminal is opened. Store in ~/.bashrc if needed ####
export VINO_HOME=$HOME  or export VINO_HOME=/opt
export VINO_WORKSPACE=$HOME/VINO
source $VINO_HOME/intel/openvino/bin/setupvars.sh
source ~/.vino_env/bin/activate
# create workspace for conversion of object detection models
mkdir -p $VINO_WORKSPACE/models
cd $VINO_WORKSPACE/models

########### SSD Mobilenet for Object detection  ###############
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -zxvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
rm ssd_mobilenet_v2_coco_2018_03_29.tar.gz
cd ssd_mobilenet_v2_coco_2018_03_29

#FP32 conversion 
python3 $VINO_HOME/intel/openvino/deployment_tools/model_optimizer/mo.py --framework tf --input_model frozen_inference_graph.pb     --output_dir  ./fp32/ --output=detection_boxes,detection_scores,num_detections --tensorflow_use_custom_operations_config $VINO_HOME/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config

#FP16 Conversion
python3 $VINO_HOME/intel/openvino/deployment_tools/model_optimizer/mo.py --framework tf --input_model frozen_inference_graph.pb     --output_dir  ./fp16/ --output=detection_boxes,detection_scores,num_detections --tensorflow_use_custom_operations_config $VINO_HOME/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --data_type FP16

#INT8 calibration requires trained model along with validation data
#install required libs
pip install shapely yamlloader scipy nibabel tqdm xmltodict pillow  sklearn py-cpuinfo --upgrade
# validation dataset download
mkdir -p $VINO_WORKSPACE/coco_data
cd $VINO_WORKSPACE/coco_data
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
# annotation Conversion
python $VINO_HOME/intel/openvino/deployment_tools/tools/accuracy_checker_tool/convert_annotation.py mscoco_detection --annotation_file $VINO_WORKSPACE/coco_data/annotations/instances_val2017.json -o ~/coco_data/ -a coco.pickle -m coco.json --has_background True --use_full_label_map  True  --sort_annotations True --images_dir $VINO_WORKSPACE/coco_data/val2017 -ss 2000
# setup calibration 
cd $VINO_WORKSPACE/models/ssd_mobilenet_v2_coco_2018_03_29
# copy calibration configuration files from standard release path and modify attributes as per settings
# Copy dataset_definitions.yml from "$VINO_HOME/intel/openvino/deployment_tools/open_model_zoo/tools/accuracy_checker"
# Modify the attributes as per your dataset. For example in this case modify "ms_coco_detection_91_classes dataset" section with attributes as "annotation with .pickle file path " and dataset_meta with .json file path"
cp $VINO_HOME/intel/openvino/deployment_tools/open_model_zoo/tools/accuracy_checker/dataset_definitions.yml .
# Copy <topology.yml> file from "$VINO_HOME/intel/openvino/deployment_tools/open_model_zoo/tools/accuracy_checker/configs" and modify launcher section FP32 with right paths for .xml and .bin file relative paths and change the input size and any other attribute. Keep only one launcher either FP32 or FP16. 
cp $VINO_HOME/intel/openvino/deployment_tools/open_model_zoo/tools/accuracy_checker/configs/ssd_mobilenet_v2_coco.yml .
# calibration
python $VINO_HOME/intel/openvino/deployment_tools/tools/calibration_tool/calibrate.py -c $VINO_WORKSPACE/models/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco.yml  -d $VINO_WORKSPACE/models/ssd_mobilenet_v2_coco_2018_03_29/dataset_definitions.yml -M $VINO_HOME/intel/openvino/deployment_tools/model_optimizer --tf_custom_op_config_dir $VINO_WORKSPACE/models/ssd_mobilenet_v2_coco_2018_03_29/ --models $VINO_WORKSPACE/models/ssd_mobilenet_v2_coco_2018_03_29/ --source $VINO_WORKSPACE/coco_data/ --annotations  $VINO_WORKSPACE/coco_data/ -e $VINO_WORKSPACE/samples/intel64/Release/lib  -cfc  -thboundary 99 --progress bar  --batch_size 1 -o ./int8
# Benchmark 
$VINO_WORKSPACE/samples/intel64/Release/benchmark_app -m $VINO_WORKSPACE/models/ssd_mobilenet_v2_coco_2018_03_29/fp32/frozen_inference_graph.xml -niter 100  -nireq <multiples of cores> -nstreams <multiples of cores>
#FP32 inference
$VINO_HOME/intel/openvino/deployment_tools/inference_engine/demos/python_demos/object_detection_demo_ssd_async/object_detection_demo_ssd_async.py -i "cam" -m $VINO_WORKSPACE/models/ssd_mobilenet_v2_coco_2018_03_29/fp32/frozen_inference_graph.xml -d CPU  -l $VINO_WORKSPACE/samples/intel64/Release/lib/libcpu_extension.so
# INT8 inference
$VINO_HOME/intel/openvino/deployment_tools/inference_engine/demos/python_demos/object_detection_demo_ssd_async/object_detection_demo_ssd_async.py -i "cam" -m $VINO_WORKSPACE/models/ssd_mobilenet_v2_coco_2018_03_29/int8/frozen_inference_graph_i8.xml -d CPU  -l $VINO_WORKSPACE/samples/intel64/Release/lib/libcpu_extension.so
########### SSD Mobilenet for Object detection  end ###############
