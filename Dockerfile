
# dockerfile
FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# code is bind-mounted by compose; keep this for image-only runs
COPY src /app/src

ENV PYTHONUNBUFFERED=1
