FROM python:3.11.4

WORKDIR /app

COPY ./requirements.txt ./

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install --no-cache-dir -r requirements.txt

COPY ./ingestion .
COPY ./utils ./utils

CMD [ "tail", "-f" , "/dev/null" ]