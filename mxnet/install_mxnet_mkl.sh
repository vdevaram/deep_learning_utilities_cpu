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
sudo yum install -y git wget numactl opencv-devel curl atlas-devel \
		    python python-pip python-devel graphviz scipy  \
                    gcc gcc-c++ make openssl-devel cmake kernel-devel
#clone the repository 
git clone --recursive https://github.com/apache/incubator-mxnet.git mkl_mxnet
cd mkl_mxnet
git submodule update --recursive
mkdir external
wget https://github.com/01org/mkl-dnn/releases/download/v0.12/mklml_lnx_2018.0.1.20171227.tgz
tar -zxvf mklml_lnx_2018.0.1.20171227.tgz
rm mklml_lnx_2018.0.1.20171227.tgz
mv mklml_lnx_2018.0.1.20171227 external/mkl
make -j $(nproc) USE_OPENCV=1 USE_MKL2017=1 USE_MKL2017_EXPERIMENTAL=1 MKLML_ROOT=$PWD/external/mkl USE_BLAS=atlas USE_DIST_KVSTORE=1
pip install --user --upgrade -e python/

# check the installation
export MKLML_ROOT=$PWD/external/mkl
export LD_LIBRARY_PATH=$MKLML_ROOT/lib:$LD_LIBRARY_PATH
python -c "import mxnet; print(mxnet.__version__)"
