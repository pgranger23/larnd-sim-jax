FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y \
        git \
        vim \
        htop \
        wget \
        build-essential \
        python3 \
        python3-pip && \
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
