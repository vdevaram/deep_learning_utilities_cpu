# This file gives step by step instructions to download install openVINO and use for running Face recognition models on IA
# Assumed to have ubuntu or CentOS installed along with Python3. Skip installation step if OpenVINO already installed.
# ################### installation and setup #######################
# download and install openVINO R3.1
cd /tmp
#wget http://registrationcenter-download.intel.com/akdlm/irc_nas/16057/l_openvino_toolkit_p_2019.3.376.tgz
#tar -zxvf l_openvino_toolkit_p_2019.3.376.tgz
#cd l_openvino_toolkit_p_2019.3.376/
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
##############################################################################
