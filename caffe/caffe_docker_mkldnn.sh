###########################################################################
# This is a sample installation file for caffe with mkldnn library on a docker
# Please contact vinod.devarampati@intel.com for any clarifications
# Installation instructions
# 1. Asumes that you are using intel CPUs
# 2. Assumes that you have installed the docker on your system
# 3. It installs the master branch. 
# 4. This installation requires internet connection.
# 5. Please uncomment and set the necessary proxy settings before installation
##########################################################################

# use centos 7.4
# use centos 7
FROM centos:7.4.1708
# proxy settings
#ENV http_proxy http://<site>:<port>
#ENV https_proxy https://<sit>:<port>
RUN yum install -y epel-release

# install required libs for caffe
RUN yum install -y \
	git \
	make \
	tar \
	wget \
	numactl \
	protobuf-devel \
	leveldb-devel \
	snappy-devel \
	opencv-devel \
	boost-devel \
	hdf5-devel \
	gflags-devel \
	glog-devel \
	lmdb-devel \
	python-devel  \
	python-pip \
	gcc-c++ \
	libibverbs-devel\
	cmake

RUN yum clean all

#RUN pip install --upgrade \
#	numpy  \
#	protobuf \
#	matplotlib  \
#	scikit-image

WORKDIR /workspace


RUN git clone https://github.com/intel/caffe.git; \
    cd caffe; \
    cp Makefile.config.example Makefile.config; \
    sed -i 's/# USE_MLSL/ USE_MLSL/g' Makefile.config; \
    make all -j"$(nproc)";\
    source ./external/mlsl/l_mlsl_2017.1.016/intel64/bin/mlslvars.sh;\
    make all -j"$(nproc)";

CMD source ./external/mlsl/l_mlsl_2017.1.016/intel64/bin/mlslvars.sh

