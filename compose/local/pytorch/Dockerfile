# syntax = docker/dockerfile:1.4.3
# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV PIP_CACHE_DIR=/var/cache/buildkit/pip
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Create user
ARG USERNAME=impaakt
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME 


# Set timezone
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

#Enable BuildKit  cache
RUN mkdir -p $PIP_CACHE_DIR
RUN rm -f /etc/apt/apt.conf.d/docker-clean

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
  apt-get update \ 
  && apt-get install -yqq --no-install-recommends \
  build-essential \
  streamer1.0-liba \
  libnss3-tools \
  libatk-bridge2.0-0 \
  libcups2-dev \
  libxkbcommon-x11-0 \
  libxcomposite-dev \
  libxrandr2 \
  libgbm-dev \
  libgtk-3-0 \
  && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
  && rm -rf /var/lib/apt/lists/*


# Requirements are installed here to ensure they will be cached.
# COPY ./requirements.txt /requirements.txt
COPY ./compose/local/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY ./compose/local/entrypoint /entrypoint
RUN sed -i 's/\r$//g' /entrypoint
RUN chmod +x /entrypoint

WORKDIR /app

ENTRYPOINT ["/entrypoint"]

USER $USERNAME

RUN python -m nltk.downloader punkt