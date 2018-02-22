#########################################################################################
# This is a sample script file to build TF  with mkl/mkldnn library on Intel CPUs
# Please contact vinod.devarampati@intel.com for any clarifications
# Instructions to fill the variable are given in comments
# make sure you are using TF1.5 or later to use these scripts
# Fill the below variables :
#          GCC_PATH
#          TF_VERSION
# Note : Instructions for input to configure file (tf.cfg) may vary depending on TF version 
##########################################################################################

sudo yum install -y epel-release
sudo yum install -y \
        git \
        make \
        tar \
        wget \
        numactl \
        python-devel \
        python-pip \
        python-wheel \
        numpy \
        libibverbs-devel
wget https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-7/vbatts-bazel-epel-7.repo
sudo mv vbatts-bazel-epel-7.repo /etc/yum.repos.d/
sudo yum install -y bazel
sudo yum clean all
export TF_VERSION=r1.5
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git fetch
git checkout $TF_VERSION
export GCC_PATH=/workspace/gcc
export PATH=$GCC_PATH/bin:$PATH
export LD_LIBRARY_PATH=$GCC_PATH/lib64:$LD_LIBRARY_PATH
cat <<EOF >tf.cfg
/usr/bin/python
/usr/lib/python2.7/site-packages
y
n
n
n
n
n
n
n
n
n
-march=native
n
EOF
cat tf.cfg | ./configure
bazel build --config=mkl //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package  ./
pip install --user --upgrade ./tensorflow-*.whl