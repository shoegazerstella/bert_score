FROM ubuntu:18.04

########################################  BASE SYSTEM
# set noninteractive installation
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y apt-utils
RUN apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    curl \
    unzip \
    supervisor \
    tzdata \
    cron \
    git

# set local timezone
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

######################################## PYTHON3
RUN apt-get install -y \
    python3 \
    python3-pip \
    python3-distutils \
    python3-setuptools \
    python-matplotlib \
    python-numpy \
    python-scipy \
    python-yaml \
    python-six

RUN mkdir src
COPY src src

RUN pip3 install pip --upgrade
RUN pip3 install -r src/requirements.txt

##################################### FASTBPE
RUN git clone https://github.com/glample/fastBPE.git && \
cd fastBPE && \
python3.6 setup.py install

################################################ CLEAN-UP
RUN apt-get remove -y \
    libyaml-dev \
    libfftw3-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libavresample-dev \
    libsamplerate0-dev \
    libtag1-dev \
    python-numpy-dev && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR src