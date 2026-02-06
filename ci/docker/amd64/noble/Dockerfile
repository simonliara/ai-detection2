FROM ubuntu:24.04

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
        libeigen3-dev \
        file \
        clang-format \
        ccache \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        cmake \
        python3-numpy \
        libcxxopts-dev \
        libspdlog-dev \
        && rm -rf /var/lib/apt/lists/*


COPY external/data-bus /tmp/data-bus
RUN /tmp/data-bus/scripts/install_cyclonedds.sh
RUN /tmp/data-bus/scripts/install_cyclonedds_cxx.sh

COPY scripts/install_build_dependencies.sh /tmp/install_build_dependencies.sh
RUN /tmp/install_build_dependencies.sh