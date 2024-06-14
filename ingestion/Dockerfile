FROM python:3.11.4

WORKDIR /app

COPY ./requirements.txt ./

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install -r requirements.txt

# Override main requirements.txt version for legacy geopandas
RUN pip uninstall pandas
RUN pip install pandas==1.5.3 

COPY ./ingestion .
COPY ./utils ./utils

CMD [ "tail", "-f" , "/dev/null" ]