FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ARG NSYS_URL=https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_4/
ARG NSYS_PKG=NsightSystems-linux-cli-public-2024.4.1.61-3431596.deb

RUN apt-get update && \
    apt-get install -y \
        git \
        vim \
        htop \
        wget \
        zsh \
        build-essential \
        python3 \
        gnupg \
        python3-pip && \
    apt-get update && apt install -y wget libglib2.0-0 && \
    wget ${NSYS_URL}${NSYS_PKG} && dpkg -i $NSYS_PKG && rm $NSYS_PKG && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Copy the entire project directory into the Docker image
COPY . .

RUN python3 -m pip install --no-cache-dir  --upgrade pip && \
    python3 -m pip install --no-cache-dir "jax[cuda]" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN python3 -m pip install --no-cache-dir jupyter

ENV PYTHONUNBUFFERED=1

ARG SCRATCH_VOLUME=/scratch
ENV SCRATCH_VOLUME=/scratch
RUN echo creating ${SCRATCH_VOLUME} && mkdir -p ${SCRATCH_VOLUME}
VOLUME ${SCRATCH_VOLUME}

RUN pip install .
