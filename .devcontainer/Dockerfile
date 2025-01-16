# Базовый образ Ubuntu с CUDA
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Установка переменных сборки
ARG SSH_PASSWORD=P2sswrd

# Установка переменных окружения
ENV PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive
ENV USER=developer
ENV SSH_PASSWORD=${SSH_PASSWORD}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Выбор зеркала
RUN sed -i 's/http:\/\/archive\.ubuntu\.com\/ubuntu/http:\/\/mirror.yandex.ru\/ubuntu\//g' /etc/apt/sources.list

# Удаление дубликатов
RUN sed -i '/^deb http:\/\/mirror.yandex.ru\/ubuntu\/ focal/d' /etc/apt/sources.list

# Добавление репозиториев
RUN echo "deb http://mirror.yandex.ru/ubuntu/ focal main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo "deb http://mirror.yandex.ru/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo "deb http://security.ubuntu.com/ubuntu focal-security main restricted universe multiverse" >> /etc/apt/sources.list

# Обновление пакетов
RUN apt-get update -y

# Очистка apt-кэша
RUN apt-get clean

# Установка wget
RUN apt-get install -y wget

# Добавление репозитория deadsnakes для установки конкретной версии Python
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa

# Обновление пакетов после добавления PPA
RUN apt-get update -y

# Установка Python 3.10
RUN apt-get install -y python3.10 python3.10-dev

# Установка Python 3.10 как основной версии для python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --config python3

# Установка Python 3.10 как основной версии для python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 2
RUN update-alternatives --config python

# Установка pip для Python 3.10
RUN python3.10 -m ensurepip
RUN python3.10 -m pip install --upgrade pip

# Рабочая директория
WORKDIR /workspace

# Копирование проекта
COPY . /workspace/

# Изменение владельца директории /workspace и ее содержимого на пользователя USER
RUN chown -R ${USER}:${USER} /workspace

# Вывод списка файлов
RUN find /workspace/ -print

# Установка PyTorch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Установка RAPIDS (cuml)
RUN pip install --extra-index-url=https://pypi.nvidia.com "cudf-cu11==24.12.*" "dask-cudf-cu11==24.12.*" "cuml-cu11==24.12.*" "cugraph-cu11==24.12.*" "nx-cugraph-cu11==24.12.*" "cuspatial-cu11==24.12.*" "cuproj-cu11==24.12.*" "cuxfilter-cu11==24.12.*" "cucim-cu11==24.12.*" "pylibraft-cu11==24.12.*" "raft-dask-cu11==24.12.*" "cuvs-cu11==24.12.*" "nx-cugraph-cu11==24.12.*"

# Установка основных python библиотек
RUN pip install -r /workspace/requirements.txt

# Установка pyg
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu118.html

# Установка jupyter
RUN pip install jupyter

# Установка nodejs и npm
RUN apt-get install -y nodejs
RUN apt-get install -y npm

# Дополнительные утилиты
RUN apt-get install -y vim gdb wget git bash-completion

# Установка openssh-server и пользователя
RUN apt-get install -y openssh-server
RUN useradd -m -s /bin/bash ${USER}

# Назначение пароля через переменную окружения
RUN echo "${USER}:${SSH_PASSWORD}" | chpasswd
RUN mkdir /run/sshd
RUN chmod 0755 /run/sshd

# Добавление пользователя  в группу sudo и разрешение sudo без пароля
RUN usermod -aG sudo ${USER}
RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Открытие порта 22 (SSH)
EXPOSE 22

# Запуск sshd
CMD ["/usr/sbin/sshd", "-D"]