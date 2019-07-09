FROM ubuntu:18.04
ARG INSTALL_DIR=/opt/intel/openvino
ARG TEMP_DIR=/tmp/
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    libgtk-3-dev \
    python3-pip \
    wget \
    cpio \
    sudo \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p $TEMP_DIR
WORKDIR $TEMP_DIR
ADD l_openvino_toolkit_p_2019.1.094 .
RUN ./install_openvino_dependencies.sh
RUN sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg
WORKDIR /
RUN    rm -rf $TEMP_DIR
# build Inference Engine samples
RUN mkdir samples
WORKDIR samples
RUN /bin/bash -c "pip3 install setuptools"
RUN /bin/bash -c "pip3 install jupyter"
RUN /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh && \
                  cmake $INSTALL_DIR/deployment_tools/inference_engine/samples && \
                  make -j"
CMD jupyter notebook --ip 0.0.0.0 --allow-root  --no-browser
