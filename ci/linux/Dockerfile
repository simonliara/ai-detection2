FROM ubuntu:22.04 as base
LABEL Author="ARA Robotics <dev@ara-uas.com>"
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install --no-install-recommends -y \
    # build tools
    build-essential \
    python3-pip \
    python3-dev \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    git \
    zip \
    cmake \ 
    && rm -rf /var/lib/apt/lists/*

COPY ../external/data-bus /tmp/data-bus
RUN /tmp/data-bus/scripts/install_cyclonedds.sh
