###########################################################################
# This is a sample installation file for mxnet with mkl library on a docker
# Please contact vinod.devarampati@intel.com for any clarifications
# Installation instructions
# 1. Asumes that you are using intel CPUs
# 2. Assumes that you have installed the docker on your system
# 3. It installs the master branch. 
# 4. This installation requires interenet connection.
# 5. Please uncomment and set the necessary proxy settings before installation
##########################################################################

# use centos 7.4
FROM centos:7.4.1708
# proxy settings
#ENV http_proxy http://<site>:<port>
#ENV https_proxy https:<site>:<port>
RUN yum install -y epel-release
# install required libs
RUN yum install -y \
	git \
	make \
	cmake \
	tar \
	wget \
	numactl \
	python-devel \
	python-pip \
	python-wheel \
	numpy \
	libibverbs-devel \
	opencv-devel \
	curl \
	atlas-devel \
	graphviz \
	scipy \
	gcc-c++ \
	openssl-devel\
	kernel-devel

RUN yum clean all
# install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py
# copy required libs
WORKDIR /workspace
RUN git clone --recursive https://github.com/apache/incubator-mxnet.git mkl_mxnet
RUN cd mkl_mxnet
RUN git submodule update --recursive
RUN mkdir external
RUN wget https://github.com/01org/mkl-dnn/releases/download/v0.12/mklml_lnx_2018.0.1.20171227.tgz
RUN tar -zxvf mklml_lnx_2018.0.1.20171227.tgz
RUN rm mklml_lnx_2018.0.1.20171227.tgz
RUN mv mklml_lnx_2018.0.1.20171227 external/mkl
RUN make -j $(nproc) USE_OPENCV=1 USE_MKL2017=1 USE_MKL2017_EXPERIMENTAL=1 MKLML_ROOT=$PWD/external/mkl USE_BLAS=atlas USE_DIST_KVSTORE=1
RUN pip install --user --upgrade -e python/
# check the installation
ENV MKLDNN_ROOT=$PWD/external/mkldnn
ENV LD_LIBRARY_PATH=$MKLDNN_ROOT/lib:$LD_LIBRARY_PATH
CMD python -c "import mxnet; print(mxnet.__version__)"
