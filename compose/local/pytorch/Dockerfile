# syntax = docker/dockerfile:1.4.3
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_NO_CACHE=1
ENV UV_FROZEN=1
ENV UV_PROJECT_ENVIRONMENT=/venv
ENV PATH=/venv/bin:$PATH

# Create user
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -yqq --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

COPY ./compose/local/pyproject.toml ./compose/local/uv.lock /build/
WORKDIR /build
RUN uv sync --no-install-project

COPY ./compose/local/entrypoint /entrypoint
RUN sed -i 's/\r$//g' /entrypoint
RUN chmod +x /entrypoint

WORKDIR /app

USER $USERNAME

RUN python -m nltk.downloader punkt

ENTRYPOINT ["/entrypoint"]
