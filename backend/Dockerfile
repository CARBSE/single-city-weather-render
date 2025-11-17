# backend/Dockerfile
FROM python:3.10.13-slim

# avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# install small build deps (kept minimal)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      libatlas-base-dev \
      gfortran \
      libffi-dev \
      libssl-dev \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements and install
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy backend source
COPY backend /app

# default data dir inside container (fetch_s3_data will save here)
ENV CARBSE_DATA_DIR=/tmp/data_private
ENV FORCE_S3_DOWNLOAD=true

# Render will set AWS credentials and S3_BUCKET_NAME as environment variables
# Use PORT provided by Render
ENV PORT=10000

# start command:
# 1) fetch data from S3 into CARBSE_DATA_DIR
# 2) then run gunicorn on $PORT with one worker (safe for free/small instances)
CMD ["bash", "-lc", "python fetch_s3_data.py && exec gunicorn app:app -b 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120"]
