FROM nvidia/cudagl:11.1-base-ubuntu20.04

SHELL ["/bin/bash", "-c"]

# Install packages without prompting the user to answer any questions
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list


ARG USR_NAME=shun-hat
#####################################################
# Install common apt packages
#####################################################
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && apt-get install -y \
    ### utility
    locales \
    xterm \
    dbus-x11 \
    terminator \
    sudo \
    ### tools
    unzip \
    lsb-release \
    curl \
    ffmpeg \
    net-tools \
    software-properties-common \
    subversion \
    libssl-dev \
    ### Development tools
    build-essential \
    htop \
    git \
    vim \
    gedit \
    gdb \
    valgrind \
    ## track ik
    libnlopt-cxx-dev \
    swig \
    libgflags-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


##########################################################
### System dependencies
##########################################################

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    parallel \
    wget \
    xpra \
    xserver-xorg-dev \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


#####################################################
# Set locale & time zone
#####################################################
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV TZ=Asia/Tokyo


COPY ./azure/files/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./azure/files/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY ./environment.yml /opt/environment.yml

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

##########################################################
### gsutil
##########################################################
RUN curl -sSL https://sdk.cloud.google.com | bash

ENV PATH $PATH:/root/google-cloud-sdk/bin

##########################################################
### MuJoCo
##########################################################
# Note: ~ is an alias for /root
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
RUN ln -s /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}
COPY ./azure/files/mjkey.txt /root/.mujoco

##########################################################
### Example Python Installation
##########################################################
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc

RUN conda update -y --name base conda && conda clean --all -y

RUN git config --global url."https://".insteadOf git://
RUN conda env create -f /opt/environment.yml
ENV PATH /opt/conda/envs/diffuser/bin:$PATH

##########################################################
### gym sometimes has this patchelf issue
##########################################################
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

RUN echo "source activate /opt/conda/envs/diffuser && export PYTHONPATH=$PYTHONPATH:/home/code && export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
RUN source ~/.bashrc

##########################################################
### mount for repo
##########################################################

RUN mkdir /home/code
RUN mkdir /home/logs


#####################################################
# Run scripts (commands)
#####################################################

### bashrc settings to activate conda environment
RUN echo "conda activate diffuser" >> /root/.bashrc

### terminator window settings
COPY assets/config /

### user group settings
COPY assets/entrypoint_setup.sh /
ENTRYPOINT ["/entrypoint_setup.sh"]

# Run terminator
CMD ["terminator"]

