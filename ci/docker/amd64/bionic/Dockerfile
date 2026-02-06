FROM ubuntu:18.04

# Avoid prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies for Python build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        build-essential \
        libssl-dev \
        curl \
        ca-certificates \
        git \
        ninja-build \
        ccache \
        libeigen3-dev \
        file \
        && rm -rf /var/lib/apt/lists/*


# install cmake 3.31.0
RUN wget --quiet --no-check-certificate -P /tmp/ https://cmake.org/files/v3.31/cmake-3.31.0-linux-x86_64.sh \
        && mkdir /opt/cmake \
        && sh /tmp/cmake-3.31.0-linux-x86_64.sh --prefix=/opt/cmake --skip-license \
        && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
        && ln -s /opt/cmake/bin/cpack /usr/local/bin/cpack \
        && rm /tmp/cmake-3.31.0-linux-x86_64.sh \
        && cmake --version 



COPY external/data-bus /tmp/data-bus
RUN /tmp/data-bus/scripts/install_cyclonedds.sh
RUN /tmp/data-bus/scripts/install_cyclonedds_cxx.sh

COPY scripts/install_build_dependencies.sh /tmp/install_build_dependencies.sh
RUN /tmp/install_build_dependencies.sh