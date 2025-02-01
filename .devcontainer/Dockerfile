FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

ARG SSH_PASSWORD=P2sswrd
ENV DEBIAN_FRONTEND=noninteractive
ENV USER=developer
ENV SSH_PASSWORD=${SSH_PASSWORD}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Установка базовых зависимостей
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        wget \
        software-properties-common \
        ca-certificates \
        gnupg \
        curl \
        sudo \
        git \
        vim \
        gdb \
        bash-completion \
        build-essential \
        libopenblas-dev \
        openssh-server \
    && apt-get clean

# Установка Python 3.10 и настройка как версии по умолчанию
RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3.10 /usr/bin/pip3 \
    && ln -sf /usr/bin/pip3.10 /usr/bin/pip

# Установка Node.js 20.x
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g npm@latest \
    && apt-get clean

# Настройка пользователя и SSH
RUN useradd -m -s /bin/bash ${USER} \
    && echo "${USER}:${SSH_PASSWORD}" | chpasswd \
    && mkdir /run/sshd \
    && chmod 0755 /run/sshd \
    && usermod -aG sudo ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /workspace
COPY requirements.txt .

# Установка Python-зависимостей
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir --extra-index-url=https://pypi.nvidia.com "cudf-cu11==24.12.*" "dask-cudf-cu11==24.12.*" "cuml-cu11==24.12.*" "cugraph-cu11==24.12.*" "nx-cugraph-cu11==24.12.*" "cuspatial-cu11==24.12.*" "cuproj-cu11==24.12.*" "cuxfilter-cu11==24.12.*" "cucim-cu11==24.12.*" "pylibraft-cu11==24.12.*" "raft-dask-cu11==24.12.*" "cuvs-cu11==24.12.*" \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu118.html \
    && pip install --no-cache-dir jupyter

COPY --chown=${USER}:${USER} . .

# Настройка прав доступа
RUN chown -R ${USER}:${USER} /workspace

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
