
#
# Dockerfile to build latest OpenCV with Python2, Python3 and Java binding support.
#
#FROM ubuntu:14.04

# check host CUDA version
FROM nvidia/cuda:7.5-cudnn5-devel

RUN mkdir OpenCV && cd OpenCV

RUN apt-get update && apt-get install -y \
  build-essential \
  checkinstall \
  cmake \
  pkg-config \
  yasm \
  libtiff5-dev \
  libjpeg-dev \
  libjasper-dev \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libdc1394-22-dev \
  libxine-dev \
  libgstreamer0.10-dev \
  libgstreamer-plugins-base0.10-dev \
  libv4l-dev \
  python-dev \
  python-numpy \
  python-pip \
  libtbb-dev \
  libeigen3-dev \
  libqt4-dev \
  libgtk2.0-dev \
  # Doesn't work libfaac-dev \
  libmp3lame-dev \
  libopencore-amrnb-dev \
  libopencore-amrwb-dev \
  libtheora-dev \
  libvorbis-dev \
  libxvidcore-dev \
  x264 \
  v4l-utils \
  # Doesn't work ffmpeg \
  libgtk2.0-dev \
  # zlib1g-dev \
  # libavcodec-dev \
  unzip \
  wget

RUN cd /opt && \
  wget https://github.com/Itseez/opencv/archive/3.1.0.zip -O opencv-3.1.0.zip -nv && \
  unzip opencv-3.1.0.zip && \
  cd opencv-3.1.0 && \
  rm -rf build && \
  mkdir build && \
  cd build && \
  cmake -D CUDA_ARCH_BIN=3.2 \
    -D CUDA_ARCH_PTX=3.2 \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_TBB=ON \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D WITH_V4L=ON \
    -D BUILD_TIFF=ON \
    -D WITH_QT=ON \
    # -D USE_GStreamer=ON \
    -D WITH_OPENGL=ON .. && \
  make -j4 && \
  make install && \
  echo "/usr/local/lib" | sudo tee -a /etc/ld.so.conf.d/opencv.conf && \
  ldconfig
RUN cp /opt/opencv-3.1.0/build/lib/cv2.so /usr/lib/python2.7/dist-packages/cv2.so


# install pip
RUN apt-get install python-pip


# install python modules
RUN pip install -U "setuptools"
RUN pip install -U "cython"
RUN pip install -U "numpy<1.12"
RUN pip install -U "ipython"
RUN pip install -U "hacking"
RUN pip install -U "nose"
RUN pip install -U "mock"
RUN pip install -U "coverage"
RUN pip install -U "chainer"
RUN apt-get -y install python-imaging






