# Base Ubuntu Image
FROM ubuntu:22.04

# Needed for geopandas poetry installation
RUN apt-get update && apt-get install -y python3-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
     libgeos-dev \
     && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN  apt-get update && apt-get install -y --no-install-recommends \
     libgdal-dev \
     && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV GDAL_CONFIG=/usr/bin/gdal-config

# Update and install necessary compilers/interpreters
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3.10 \
    python3-pip \
    g++ \
    curl \
    make \
    git \
    gnupg \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*

# Create a symbolic link for Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set up working directory
WORKDIR /app

# mjh, Added the following line to install
RUN git clone https://github.com/iamgreggarcia/codesherpa.git
RUN mv codesherpa/* /app/

COPY ./code-interpreter/pyproject.toml /app/
RUN rm ./poetry.lock

RUN pip install --no-cache-dir poetry

# Create virtual environment and install dependencies
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-dev --no-root

RUN /app/.venv/bin/pip install --no-cache-dir pytest pytest-cov pytest-asyncio
RUN /app/.venv/bin/pip install --no-cache-dir google-cloud-secret-manager 

RUN mkdir -p /app/static/images
RUN mkdir -p /app/static/uploads

ENV PYTHONPATH=/app

ENV PATH="/app/.venv/bin:${PATH}"

#COPY . /app/

#CMD ["python", "-c", "import server.main; server.main.start()"]
CMD ["python", "-c", "import localserver.main; localserver.main.start()"]
