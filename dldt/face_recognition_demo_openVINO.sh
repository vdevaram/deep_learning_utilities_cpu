# This file gives step by step instructions to download install openVINO and use for running Face recognition models on IA
# Assumed to have ubuntu or CentOS installed along with Python3. Skip installation step if OpenVINO already installed.
# ################### installation and setup #######################
# download and install openVINO R3.1
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

########### Face Recognition detection  ###############
# Download and use the IR models generated with the openVINO tool
mkdir -p fr/fp32
cd $VINO_WORKSPACE/models/fr/fp32
wget https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.bin
wget https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml
wget https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/face-detection-retail-0004/FP32/face-detection-retail-0004.bin
wget https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/face-detection-retail-0004/FP32/face-detection-retail-0004.xml
wget https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml
wget https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.bin
#copy demo script files
cp $VINO_HOME/intel/openvino/deployment_tools/inference_engine/demos/python_demos/face_recognition_demo $VINO_WORKSPACE/ -rf

# Run Demo for Face Recognition with camera connected to the system
# Give the file path to '-i' option in the below command to pass a Video file.
# You can see the newly identified faces on the Screen. After adding name you can see the Name on further frames.
python $VINO_WORKSPACE/face_recognition_demo/face_recognition_demo.py -i 0 --run_detector -m_fd $VINO_WORKSPACE/models/fr/fp32/face-detection-adas-0001.xml -m_reid $VINO_WORKSPACE/models/fr/fp32/face-reidentification-retail-0095.xml  -m_lm $VINO_WORKSPACE/models/fr/fp32/landmarks-regression-retail-0009.xml -l ~/VINO/samples/intel64/Release/lib/libcpu_extension.so -fg $VINO_WORKSPACE/face_dir --allow_grow
