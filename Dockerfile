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

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "jax[cuda]" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install PyTorch and torchvision
RUN python3 -m pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN python3 -m pip install jupyter

ENV PYTHONUNBUFFERED=1

ARG SCRATCH_VOLUME=/scratch
ENV SCRATCH_VOLUME=/scratch
RUN echo creating ${SCRATCH_VOLUME} && mkdir -p ${SCRATCH_VOLUME}
VOLUME ${SCRATCH_VOLUME}

ADD requirements.txt /work/requirements.txt

RUN mkdir -p /tmp && wget -q --no-check-certificate -P /tmp https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.4.tar.bz2 && \
    tar -x -f /tmp/openmpi-4.0.4.tar.bz2 -C /tmp -j && \
    cd /tmp/openmpi-4.0.4 && ./configure --prefix=/usr/local/openmpi --disable-getpwuid \
    --with-slurm --with-cuda && \
    make -j4 && \
    make -j4 install && \
    rm -rf /tmp/openmpi-4.0.4.tar.bz2 /tmp/openmpi-4.0.4
ENV PATH=/usr/local/openmpi/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH

RUN pip install --no-cache-dir -r requirements.txt
