# syntax = docker/dockerfile:1.4.3
FROM python:3.14-slim

ENV PIP_CACHE_DIR=/var/cache/buildkit/pip
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create user
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

RUN mkdir -p $PIP_CACHE_DIR
RUN rm -f /etc/apt/apt.conf.d/docker-clean

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
  apt-get update \
  && apt-get install -yqq --no-install-recommends \
  build-essential \
  libpq-dev \
  && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
  && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
  pip install --upgrade pip setuptools

COPY ./compose/local/requirements.txt /requirements.txt
RUN --mount=type=cache,target=$PIP_CACHE_DIR \
  pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu && \
  pip install -r /requirements.txt

COPY ./compose/local/entrypoint /entrypoint
RUN sed -i 's/\r$//g' /entrypoint
RUN chmod +x /entrypoint

WORKDIR /app

ENTRYPOINT ["/entrypoint"]

USER $USERNAME

RUN python -m nltk.downloader punkt