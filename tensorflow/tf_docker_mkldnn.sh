###########################################################################
# This is a sample Dockerfile for TF with mkl library on a docker
# Please contact vinod.devarampati@intel.com for any clarifications
# Installation instructions
# 1. Asumes that you are using intel CPUs
# 2. Assumes that you have installed the docker on your system
# 3. It installs the TF wheel already generated with MKL. 
#   -- please use https://raw.githubusercontent.com/vdevaram/deep_learning_utilities_cpu/master/tensorflow/tf_build.sh for building optimized mkl wheel 
# 4. This installation requires internet connection.
##########################################################################

# use centos 7
FROM centos:7.4.1708
RUN yum install -y epel-release
# install required libs
RUN yum install -y \
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

RUN yum clean all
# install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py
# copy the MKL integrated TF wheel
WORKDIR /workspace
ADD tensorflow-<version>-cp27-cp27mu-linux_x86_64.whl .
#install tf wheel
RUN pip install --no-cache-dir tensorflow-<version>-cp27-cp27mu-linux_x86_64.whl
