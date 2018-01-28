###########################################################################
# This is a sample installation file for mxnet with mkl library on Bare metal
# Please contact vinod.devarampati@intel.com for any clarifications
# Installation instructions
# 1. Asumes that you are using intel CPUs
# 2. Latest version of CentOS/RHEL is installed (preferably minimal install)  
# 	- CentOS 7.4 (1708) is recommended
# 3. It installs the master branch. 
# 4. This installation requires interenet connection.
# 5. Please set the necessary proxy settings before installation
########################################################################## 
# Install the basic libs for mxnet
sudo yum install -y epel-release
sudo yum install -y git wget numactl opencv-devel curl python python-pip \
		    python-devel graphviz scipy gcc gcc-c++ make openssl-devel \
                    cmake kernel-devel
# clone repository 
git clone --recursive https://github.com/ykim362/mxnet.git mkldnn_mxnet
cd mkldnn_mxnet
git checkout add-mkldnn
git submodule update --recursive
make -j $(nproc) USE_MKLDNN=1 MKLDNN_ROOT=$PWD/external/mkldnn USE_BLAS=mkl USE_DIST_KVSTORE=1
pip install --user -e python/

#check the installation
export MKLDNN_ROOT=$PWD/external/mkldnn
export LD_LIBRARY_PATH=$MKLDNN_ROOT/lib:$LD_LIBRARY_PATH
python -c "import mxnet; print(mxnet.__version__)"
