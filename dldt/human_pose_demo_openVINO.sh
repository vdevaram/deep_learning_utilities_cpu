# This file gives step by step instructions to download install openVINO and use for running human pose models on IA
#Assumed to have ubuntu or CentOS installed along with Python3. Skip installation step if openVINO already installed.
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

########### Human pose detection  ###############
# Download and use the IR models generated with the openVINO tool
mkdir -p human_pose/fp32
mkdir -p human_pose/fp16
mkdir -p human_pose/int8
cd $VINO_WORKSPACE/models/human_pose/fp32
wget https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml
wget https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/human-pose-estimation-0001/FP32/human-pose-estimation-0001.bin
cd $VINO_WORKSPACE/models/human_pose/fp16
wget https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml
wget https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/human-pose-estimation-0001/FP16/human-pose-estimation-0001.bin
cd $VINO_WORKSPACE/models/human_pose/int8
wget https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/human-pose-estimation-0001/INT8/human-pose-estimation-0001.xml
wget https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/human-pose-estimation-0001/INT8/human-pose-estimation-0001.bin
 
Benchmark 
$VINO_WORKSPACE/samples/intel64/Release/benchmark_app -m $VINO_WORKSPACE/models/human_pose/fp32/human-pose-estimation-0001.xml -niter 100  -nireq <multiples of cores> -nstreams <multiples of cores> -d CPU
$VINO_WORKSPACE/samples/intel64/Release/benchmark_app -m $VINO_WORKSPACE/models/human_pose/fp16/human-pose-estimation-0001.xml -niter 100  -nireq <multiples of cores> -nstreams <multiples of cores> -d GPU
$VINO_WORKSPACE/samples/intel64/Release/benchmark_app -m $VINO_WORKSPACE/models/human_pose/int8/human-pose-estimation-0001.xml -niter 100  -nireq <multiples of cores> -nstreams <multiples of cores> -d CPU

#FP32 inference
$VINO_WORKSPACE/demos/intel64/Release/human_pose_estimation_demo -m ~/dldt/models/human_pose/fp32/human-pose-estimation-0001.xml -i < image or video or "cam"> 
# INT8 inference
$VINO_WORKSPACE/demos/intel64/Release/human_pose_estimation_demo -m ~/dldt/models/human_pose/int8/human-pose-estimation-0001.xml -i < image or video or "cam"> 
########### human pose detection  end ###############
