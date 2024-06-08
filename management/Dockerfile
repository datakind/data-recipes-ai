FROM python:3.11.4

# Needed for geopandas poetry installation
RUN apt-get update && apt-get install -y python3-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
     libgeos-dev \
     && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN  apt-get update && apt-get install -y --no-install-recommends \
     libgdal-dev \
     && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV GDAL_CONFIG=/usr/bin/gdal-config

# Copy application files and install dependencies
COPY ./management/ /app
COPY ./utils/ /app/utils
COPY ./templates /app/templates
COPY requirements.txt /app
RUN pip install --upgrade pip
WORKDIR /app
RUN pip install -r requirements.txt 
RUN pip install autogenstudio

# Set the path
ENV PATH="/home/app/.local/bin:${PATH}"

# set python path
ENV PYTHONPATH="/home/app/.local/bin:/app:${PYTHONPATH}"

RUN autogenstudio version

EXPOSE 8081

ENTRYPOINT [ "autogenstudio", "ui", "--host", "0.0.0.0", "--port", "8081", "--appdir", "/app" ]


