# This file gives step by step instructions to download install openVINO and use for running Face recognition models on IA
# Assumed to have ubuntu or CentOS installed along with Python3. Skip installation step if OpenVINO already installed.
# ################### installation and setup #######################
# download and install openVINO 2019R4
cd /tmp
#wget http://registrationcenter-download.intel.com/akdlm/irc_nas/xxxx/l_openvino_toolkit_p_2019.4.xxx.tgz
#tar -zxvf l_openvino_toolkit_p_2019.4.xxx.tgz
#cd l_openvino_toolkit_p_2019.4.xxx/
sudo ./install_openvino_dependencies.sh
./install.sh
# if the openVINO installed in root mode then installation path will be /opt. otherwise it will be in $HOME folder. Choose below command accordingly
export VINO_HOME=$HOME  or export VINO_HOME=/opt
# Add openVINO libs to PATH and LD_LIBRARY_PATH. Add below line to .bashrc file if you want to enable by default
source $VINO_HOME/intel/openvino/bin/setupvars.sh
# create virtual environment and enable for python3 as below for ubuntu. instructions are different for centOS.  
# deploy package runs with python 3.6 or above. For ubuntu 16.04, follow these instructions to install python3.6 https://vsupalov.com/developing-with-python3-6-on-ubuntu-16-04/
virtualenv -p python3.6 ~/.vino_env
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
cmake $VINO_HOME/intel/openvino/deployment_tools/inference_engine/samples/cpp
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
##############################################################################
#### Example procedure for deployment package with frcnn example ######
# model conversion
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz
tar -zxvf faster_rcnn_resnet50_coco_2018_01_28.tar.gz
rm faster_rcnn_resnet50_coco_2018_01_28.tar.gz
cd faster_rcnn_resnet50_coco_2018_01_28
#FP32 conversion 
python3 $VINO_HOME/intel/openvino/deployment_tools/model_optimizer/mo.py --framework tf --input_model frozen_inference_graph.pb     --output_dir  ./fp32/ --output=detection_boxes,detection_scores,num_detections --tensorflow_use_custom_operations_config $VINO_HOME/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config
### procedure for packaging ######################
# Move all dependency and user files to separate folder i.e opencv libs,IRs, samples/demo files
mkdir -p $VINO_WORKSPACE/deploy_dir
# IR files
mv $VINO_WORKSPACE/models/faster_rcnn_resnet50_coco_2018_01_28/fp32 $VINO_WORKSPACE/deploy_dir
# opencv libs
cp -rf $VINO_HOME/intel/openvino/opencv/lib $VINO_WORKSPACE/deploy_dir/
# samples libs
cp $VINO_WORKSPACE/samples/intel64/Release/lib/libformat_reader.so $VINO_WORKSPACE/deploy_dir/lib/
# demo binaries
cp $VINO_WORKSPACE/demos/intel64/Release/object_detection_demo_ssd_async $VINO_WORKSPACE/deploy_dir/
# run package command. optional to use UI. instructions are at https://docs.openvinotoolkit.org/latest/_docs_install_guides_deployment_manager_tool.html
python $VINO_HOME/intel/openvino/deployment_tools/tools/deployment_manager/deployment_manager.py --targets cpu --output_dir $VINO_WORKSPACE/ --archive_name deploy_pkg_frcnn  --user_data  $VINO_WORKSPACE/deploy_dir/
# now deployment package is ready at $VINO_WORKSPACE/deploy_pkg_frcnn.tar.gz
# Copy to target host
scp $VINO_WORKSPACE/deploy_pkg_frcnn.tar.gz <username>@<target_ip>:
exit
####### Below instructions on target device. Make sure same OS version and Python version on target ##########
# Coonect to target device
ssh <username>@<target_ip>
# assume workspace to be TARGET_WORKSPACE
export TARGET_WORKSPACE=$HOME/VINO
mkdir -p $TARGET_WORKSPACE
mv $HOME/deploy_pkg_frcnn.tar.gz $TARGET_WORKSPACE
cd $TARGET_WORKSPACE
tar -zxvf deploy_pkg_frcnn.tar.gz
# folders "bin  deploy_dir  deployment_tools  install_dependencies" will be seen in the deploy_pkg_frcnn
# Activate openVINO environment
source $TARGET_WORKSPACE/deploy_pkg_frcnn/bin/setupvars.sh
# add libs to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TARGET_WORKSPACE/deploy_pkg_frcnn/deploy_dir/lib:$LD_LIBRARY_PATH
# Now everything is ready to lunch the FRCNN demo
$TARGET_WORKSPACE/deploy_pkg_frcnn/deploy_dir/object_detection_demo_ssd_async -i <videopath/cam> -m $TARGET_WORKSPACE/deploy_pkg_frcnn/deploy_dir/fp32/frozen_inference_graph.xml
# you should be able to see the objectdetection demo running
# All above procedure not required for further applications. Just copy the new deploy_dir
###################################end ###########################################
