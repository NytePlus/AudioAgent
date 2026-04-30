FROM python:3.11-slim-bullseye

WORKDIR /build

COPY . ./
RUN pip install -e .
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace