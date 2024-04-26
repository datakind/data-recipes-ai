FROM debian:bookworm-slim

RUN echo "Hi!"

ARG DATA_DB_CONN_STRING
ENV DATA_DB_CONN_STRING=$DATA_DB_CONN_STRING

# Setup Nginx and Supervisor
RUN apt-get update && apt-get install -y procps nginx supervisor && \
    rm -rf /var/lib/apt/lists/*

COPY actions/config/nginx.conf /etc/nginx/nginx.conf
COPY actions/config/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Setup user and permissions
RUN useradd -m as-user
RUN chown -R as-user:as-user /var/log /run /var/lib/nginx

# Setup workspace
RUN mkdir -p /action-server/datadir /action-server/actions
RUN chown -R as-user:as-user /action-server

WORKDIR /action-server/actions

# Setup Action Server
ADD https://downloads.robocorp.com/action-server/releases/latest/linux64/action-server /usr/local/bin/action-server
RUN chmod +x /usr/local/bin/action-server

COPY ./actions .
RUN chown -R as-user:as-user /action-server

USER as-user

ARG DATA_DB_CONN_STRING
ENV DATA_DB_CONN_STRING=$DATA_DB_CONN_STRING

# Load all actions
# RUN action-server import --datadir=/action-server/datadir
# Load individually
RUN action-server import --dir=./actions_plugins/postgres-universal --datadir=/action-server/datadir
RUN action-server import --dir=./actions_plugins/recipe-server --datadir=/action-server/datadir

RUN echo "{\"dsn\": \"$DATA_DB_CONN_STRING\"}" > ./actions_plugins/postgres-universal/postgres_connection.json

EXPOSE 8080

CMD ["/usr/bin/supervisord"]
#CMD ["python", "./actions_plugins/postgres-universal/main.py", "--dsn", "$DATA_DB_CONN_STRING]