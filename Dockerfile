FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime


RUN apt-get update 

RUN apt-get install -y \
    build-essential wget

RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py &&\
         python3 get-pip.py &&\
         rm -f get-pip.py

RUN pip3 install \
        jupyter \
        torch==1.4.0 \
        torchvision==0.5.0 \
        pandas==1.0.2 \
        matplotlib==3.2.0 \
        opencv-contrib-python==4.2.0.32 \
        onnxruntime==1.1.0

RUN apt-get install -y \
        libsm6=2:1.2.2-1 \ 
        libxext6=2:1.3.3-1

# ###############################################################

ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTPS_PROXY
ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/16345/l_openvino_toolkit_p_2020.1.023_online.tgz
ARG INSTALL_DIR=/opt/intel/openvino
ARG TEMP_DIR=/tmp/openvino_installer

RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        wget \
        cpio \
        sudo \
        lsb-release && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && \
    wget -c $DOWNLOAD_LINK && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR
RUN $INSTALL_DIR/install_dependencies/install_openvino_dependencies.sh


RUN cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites && \
    sudo ./install_prerequisites.sh

RUN pip3 install networkx==2.3
RUN pip3 install protobuf==3.6.1

RUN /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh && cd /opt/intel/openvino/deployment_tools/demo && ./demo_squeezenet_download_convert_run.sh"

RUN echo 'source /opt/intel/openvino/bin/setupvars.sh' >> ~/.bashrc

RUN pip install imgaug

WORKDIR /app/
COPY . /app/
